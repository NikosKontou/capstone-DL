import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
import config as cfg

class SpinFeaturePreprocessor:
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
        self.stds_ = numeric_matrix.std(axis=0).replace(0, 1.0).values.astype(np.float32)
        return self

    def _build_numeric_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Apply cyclic encoding
        if 'hour_of_day' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24.0)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24.0)
        if 'day_of_week' in df.columns:
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)

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


class SpinDataset(Dataset):
    def __init__(self, X: np.ndarray, y_spins_left: np.ndarray,
                 y_next_bet: np.ndarray, next_bet_mask: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y_spins_left = torch.from_numpy(y_spins_left).float().unsqueeze(1)
        self.y_next_bet = torch.from_numpy(y_next_bet).float().unsqueeze(1)
        self.next_bet_mask = torch.from_numpy(next_bet_mask).float().unsqueeze(1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "y_spins_left": self.y_spins_left[idx],
            "y_next_bet": self.y_next_bet[idx],
            "next_bet_mask": self.next_bet_mask[idx],
        }

def _pick_accounts_to_keep(fpath: str, size_pct: float, seed: int) -> set:
    accounts = pq.read_table(fpath, columns=[cfg.ACCOUNT_COL])[cfg.ACCOUNT_COL].to_numpy()
    unique_accounts = np.unique(accounts)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_accounts)
    n_keep = max(1, round(len(unique_accounts) * size_pct / 100))
    return set(unique_accounts[:n_keep].tolist()), len(unique_accounts), n_keep

def load_splits(data_dir: str = None, train_file: str = None,
                 valid_file: str = None, test_file: str = None,
                 size_pct: float = None, seed: int = None):
    data_dir = data_dir or cfg.DATA_DIR
    filenames = {
        "train": train_file or cfg.TRAIN_FILE,
        "valid": valid_file or cfg.VALID_FILE,
        "test": test_file or cfg.TEST_FILE,
    }

    # Removed required_cols checking logic here temporarily as we drop raw hour/day
    # but still expect them from upstream data. Kept it simple.

    apply_sampling = size_pct is not None and size_pct < 100
    seed = cfg.RANDOM_SEED if seed is None else seed
    splits = {}

    for name, fname in filenames.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Expected {name} split at {fpath} -- not found.")

        if apply_sampling:
            keep_accounts, n_total_accounts, n_keep_accounts = _pick_accounts_to_keep(
                fpath, size_pct, seed
            )
            dataset = pa_ds.dataset(fpath, format="parquet")
            filter_expr = pa_ds.field(cfg.ACCOUNT_COL).isin(list(keep_accounts))
            table = dataset.to_table(filter=filter_expr)
            d = table.to_pandas()
        else:
            d = pd.read_parquet(fpath)

        if cfg.SPLIT_COL in d.columns:
            bad = d[cfg.SPLIT_COL].astype(str).ne(name)
            if bad.any():
                raise ValueError("Split mismatch detected.")

        splits[name] = d.reset_index(drop=True)

    return splits


def build_datasets(splits: dict):
    pre = SpinFeaturePreprocessor().fit(splits["train"])
    datasets = {}

    for name, d in splits.items():
        X = pre.transform(d)

        # Log1p transforms
        y_spins_left = np.log1p(d[cfg.TARGET_SPINS_LEFT].values.astype(np.float32))
        next_bet_mask = (~d[cfg.TARGET_NEXT_BET].isna()).values.astype(np.float32)
        y_next_bet = np.log1p(d[cfg.TARGET_NEXT_BET].fillna(0.0).values.astype(np.float32))

        datasets[name] = SpinDataset(X, y_spins_left, y_next_bet, next_bet_mask)

    return datasets, pre

def build_dataloaders(datasets: dict, batch_size: int = None):
    batch_size = batch_size or cfg.BATCH_SIZE
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, drop_last=False),
        "valid": DataLoader(datasets["valid"], batch_size=batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False),
    }
    return loaders
