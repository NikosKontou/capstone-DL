"""
Sequence data pipeline for the time-series spin model.

Unit of a training example = one session's spins, in order. Padding is
required because sessions have variable length; the padding mask is
combined with the existing terminal-spin mask (next_spin_amount is
NULL on the last spin of a session, same reasoning as the row-level
model) so that both padded steps and terminal steps are excluded from
the next_bet loss/metrics.
"""
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import config as cfg


class SpinFeaturePreprocessor:
    """
    Fits per-column medians/means/stds on the training split. Structurally
    identical to the row-level model's preprocessor -- scaling is still
    done per-row, sequencing happens on top of it afterward.
    """

    def __init__(self):
        self.medians_ = {}
        self.means_ = None
        self.stds_ = None
        self.feature_names_ = None

    def fit(self, df_train: pd.DataFrame):
        for col in cfg.ADD_MISSING_INDICATORS:
            self.medians_[col] = df_train[col].median()

        numeric_matrix = self._build_numeric_matrix(df_train)
        self.feature_names_ = numeric_matrix.columns.tolist()
        self.means_ = numeric_matrix.mean(axis=0).values.astype(np.float32)
        stds = numeric_matrix.std(axis=0)
        stds = stds.where(stds > 0, 1.0)
        self.stds_ = stds.values.astype(np.float32)
        return self

    def _build_numeric_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "hour_of_day" in df.columns:
            df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24.0)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24.0)
        if "day_of_week" in df.columns:
            df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
            df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

        out = df[cfg.NUMERIC_FEATURES].copy()
        for col in cfg.ADD_MISSING_INDICATORS:
            missing = out[col].isna()
            out[f"{col}_was_missing"] = missing.astype(np.float32)
            out[col] = out[col].fillna(self.medians_[col])

        out = out.fillna(0.0)
        return out

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        numeric_matrix = self._build_numeric_matrix(df)
        numeric_matrix = numeric_matrix[self.feature_names_]
        scaled = (numeric_matrix.values.astype(np.float32) - self.means_) / self.stds_
        onehot = df[cfg.ONEHOT_FEATURES].fillna(0.0).values.astype(np.float32)
        return np.concatenate([scaled, onehot], axis=1)

    @property
    def n_features(self) -> int:
        return len(self.feature_names_) + len(cfg.ONEHOT_FEATURES)

    @property
    def all_feature_names(self):
        return self.feature_names_ + cfg.ONEHOT_FEATURES


def _pick_accounts_to_keep(fpath: str, size_pct: float, seed: int) -> set:
    accounts = pq.read_table(fpath, columns=[cfg.ACCOUNT_COL])[cfg.ACCOUNT_COL].to_numpy()
    unique_accounts = np.unique(accounts)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_accounts)
    n_keep = max(1, round(len(unique_accounts) * size_pct / 100))
    return set(unique_accounts[:n_keep].tolist())


def load_splits(data_dir: str = None, train_file: str = None,
                 valid_file: str = None, test_file: str = None,
                 size_pct: float = None, seed: int = None):
    data_dir = data_dir or cfg.DATA_DIR
    filenames = {
        "train": train_file or cfg.TRAIN_FILE,
        "valid": valid_file or cfg.VALID_FILE,
        "test": test_file or cfg.TEST_FILE,
    }

    apply_sampling = size_pct is not None and size_pct < 100
    seed = cfg.RANDOM_SEED if seed is None else seed
    splits = {}

    for name, fname in filenames.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Expected {name} split at {fpath} -- not found.")

        if apply_sampling:
            keep_accounts = _pick_accounts_to_keep(fpath, size_pct, seed)
            dataset = pa_ds.dataset(fpath, format="parquet")
            filter_expr = pa_ds.field(cfg.ACCOUNT_COL).isin(list(keep_accounts))
            d = dataset.to_table(filter=filter_expr).to_pandas()
        else:
            d = pd.read_parquet(fpath)

        if cfg.SPLIT_COL in d.columns:
            bad = d[cfg.SPLIT_COL].astype(str).ne(name)
            if bad.any():
                raise ValueError("Split mismatch detected.")

        # Sort within account/session so sequence order is correct.
        # Sequences must be ordered by time; sessionid groups rows into
        # sequences, account is only needed if sequences were ever
        # built across sessions (not the case here, but kept as a
        # stable outer sort key for determinism).
        d = d.sort_values([cfg.ACCOUNT_COL, cfg.SESSION_COL, cfg.TIME_COL]).reset_index(drop=True)
        splits[name] = d

    return splits


def _sequences_from_df(df: pd.DataFrame, X: np.ndarray):
    """
    Slice the row-level feature matrix X (already preprocessed, aligned
    row-for-row with df) into a list of per-session sequences, applying
    MAX_SEQ_LEN truncation (keep the most recent spins) and MIN_SEQ_LEN
    filtering.

    Returns a list of dicts, one per kept session, each holding the
    feature sequence and per-step targets/masks as numpy arrays of
    shape (T, ...).
    """
    y_spins_left_all = np.log1p(df[cfg.TARGET_SPINS_LEFT].values.astype(np.float32))
    next_bet_mask_all = (~df[cfg.TARGET_NEXT_BET].isna()).values.astype(np.float32)
    y_next_bet_all = np.log1p(df[cfg.TARGET_NEXT_BET].fillna(0.0).values.astype(np.float32))

    sequences = []
    # groupby on a pre-sorted frame preserves within-group row order.
    group_keys = df.groupby([cfg.ACCOUNT_COL, cfg.SESSION_COL], sort=False).indices
    for _, idx in group_keys.items():
        idx = np.asarray(idx)
        if len(idx) < cfg.MIN_SEQ_LEN:
            continue
        if len(idx) > cfg.MAX_SEQ_LEN:
            idx = idx[-cfg.MAX_SEQ_LEN:]  # keep most recent spins

        sequences.append({
            "x": X[idx],
            "y_spins_left": y_spins_left_all[idx],
            "y_next_bet": y_next_bet_all[idx],
            "next_bet_mask": next_bet_mask_all[idx],
        })
    return sequences


class SpinSequenceDataset(Dataset):
    """
    Each item is one session: variable-length (T, F) feature sequence
    plus per-step targets/masks of shape (T,). Padding to a common
    length within a batch happens in collate_fn, not here, so no
    wasted computation/memory from padding to the longest sequence in
    the whole dataset.
    """

    def __init__(self, sequences: list):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            "x": torch.from_numpy(seq["x"]).float(),
            "y_spins_left": torch.from_numpy(seq["y_spins_left"]).float(),
            "y_next_bet": torch.from_numpy(seq["y_next_bet"]).float(),
            "next_bet_mask": torch.from_numpy(seq["next_bet_mask"]).float(),
            "length": seq["x"].shape[0],
        }


def collate_sequences(batch):
    """
    Pads a list of variable-length session dicts into batch tensors of
    shape (B, T_max, ...), plus a (B, T_max) padding_mask (1 = real
    step, 0 = padding) and the original lengths (needed for
    pack_padded_sequence).
    """
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    x = pad_sequence([b["x"] for b in batch], batch_first=True)  # (B, T_max, F)
    y_spins_left = pad_sequence([b["y_spins_left"] for b in batch], batch_first=True)
    y_next_bet = pad_sequence([b["y_next_bet"] for b in batch], batch_first=True)
    next_bet_mask = pad_sequence([b["next_bet_mask"] for b in batch], batch_first=True)

    T_max = x.shape[1]
    padding_mask = (torch.arange(T_max).unsqueeze(0) < lengths.unsqueeze(1)).float()  # (B, T_max)

    # Combined mask for the next-bet loss: only real, non-terminal steps count.
    combined_next_bet_mask = next_bet_mask * padding_mask

    return {
        "x": x,
        "y_spins_left": y_spins_left.unsqueeze(-1),      # (B, T_max, 1)
        "y_next_bet": y_next_bet.unsqueeze(-1),           # (B, T_max, 1)
        "next_bet_mask": combined_next_bet_mask.unsqueeze(-1),  # (B, T_max, 1)
        "padding_mask": padding_mask.unsqueeze(-1),       # (B, T_max, 1): all-steps mask (for spins_left loss)
        "lengths": lengths,
    }


def build_datasets(splits: dict):
    pre = SpinFeaturePreprocessor().fit(splits["train"])
    datasets = {}

    for name, d in splits.items():
        X = pre.transform(d)
        sequences = _sequences_from_df(d, X)
        datasets[name] = SpinSequenceDataset(sequences)

    return datasets, pre


def build_dataloaders(datasets: dict, batch_size: int = None):
    batch_size = batch_size or cfg.BATCH_SIZE

    # drop_last=True on train: BatchNorm isn't used here (LSTM +
    # LayerNorm-free heads), so a size-1 trailing batch wouldn't crash,
    # but keeping drop_last=True avoids a batch with a single very
    # short sequence dominating a gradient step disproportionately.
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                             drop_last=True, collate_fn=collate_sequences),
        "valid": DataLoader(datasets["valid"], batch_size=batch_size, shuffle=False,
                             collate_fn=collate_sequences),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False,
                            collate_fn=collate_sequences),
    }
    return loaders
