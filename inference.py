"""
Serving contract change vs the row-level model: SpinPredictor.predict
now takes the full spin history of a session-in-progress (a DataFrame
of consecutive spins, oldest first) rather than a single row, because
the LSTM needs the preceding steps in its hidden state to produce a
meaningful prediction at the current step. It returns a prediction at
the *last* row of the given history.

If a caller only has a single spin (session start), that's still valid
input -- lengths=1 -- but predictions at the very first step of a
session are naturally less informed than later ones, same as the
model would be during training.
"""
import numpy as np
import pandas as pd
import torch

import config as cfg
from model import build_model


class SpinPredictor:
    def __init__(self, checkpoint_path: str, device: str = None):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.n_features = ckpt["n_features"]
        self.feature_names = ckpt["feature_names"]
        self.medians = ckpt["preprocessor_medians"]
        self.means = np.array(ckpt["preprocessor_means"], dtype=np.float32)
        self.stds = np.array(ckpt["preprocessor_stds"], dtype=np.float32)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(self.n_features)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self._base_numeric = [c for c in cfg.NUMERIC_FEATURES if not c.endswith(("_sin", "_cos"))]

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()

        if "hour_of_day" in df.columns:
            df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24.0)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24.0)
        if "day_of_week" in df.columns:
            df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
            df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

        numeric_cols = [c for c in cfg.NUMERIC_FEATURES if c in df.columns]
        out = df[numeric_cols].copy()
        for col in cfg.ADD_MISSING_INDICATORS:
            if col not in out.columns:
                continue
            missing = out[col].isna()
            out[f"{col}_was_missing"] = missing.astype(np.float32)
            out[col] = out[col].fillna(self.medians[col])

        out = out.fillna(0.0)

        missing_cols = [c for c in self.feature_names if c not in out.columns and c in cfg.NUMERIC_FEATURES + [f"{c}_was_missing" for c in cfg.ADD_MISSING_INDICATORS]]
        # Strict check: every expected numeric feature name must be present.
        numeric_feature_names = [c for c in self.feature_names if c not in cfg.ONEHOT_FEATURES]
        missing_numeric = [c for c in numeric_feature_names if c not in out.columns]
        if missing_numeric:
            raise ValueError(f"Missing expected numeric features at inference time: {missing_numeric}")
        out = out[numeric_feature_names]

        n_numeric = len(numeric_feature_names)
        scaled = (out.values.astype(np.float32) - self.means[:n_numeric]) / self.stds[:n_numeric]

        missing_onehot = [c for c in cfg.ONEHOT_FEATURES if c not in df.columns]
        if missing_onehot:
            raise ValueError(f"Missing expected one-hot features at inference time: {missing_onehot}")
        onehot = df[cfg.ONEHOT_FEATURES].fillna(0.0).values.astype(np.float32)

        return np.concatenate([scaled, onehot], axis=1)

    @torch.no_grad()
    def predict(self, session_history_df: pd.DataFrame):
        """
        session_history_df: consecutive spins for ONE session, sorted
        oldest-first (e.g. by starttime / spin_number_in_session).
        Returns predictions (spins_left, next_bet) for the LAST row in
        the given history -- i.e. "given everything up to and
        including the most recent spin, what happens next".
        """
        if len(session_history_df) == 0:
            raise ValueError("session_history_df must contain at least one row.")

        X = self._build_features(session_history_df)  # (T, F)
        x_t = torch.from_numpy(X).float().unsqueeze(0).to(self.device)  # (1, T, F)
        lengths = torch.tensor([X.shape[0]], dtype=torch.long)

        pred_spins_left, pred_next_bet = self.model(x_t, lengths)  # (1, T, 1) each

        last_spins_left = pred_spins_left[0, -1, 0].item()
        last_next_bet = pred_next_bet[0, -1, 0].item()

        return np.expm1(last_spins_left), np.expm1(last_next_bet)

    @torch.no_grad()
    def predict_all_steps(self, session_history_df: pd.DataFrame):
        """
        Same as predict(), but returns predictions at every timestep in
        the given history rather than just the last one -- useful for
        offline analysis of how predictions evolve across a session.
        """
        X = self._build_features(session_history_df)
        x_t = torch.from_numpy(X).float().unsqueeze(0).to(self.device)
        lengths = torch.tensor([X.shape[0]], dtype=torch.long)

        pred_spins_left, pred_next_bet = self.model(x_t, lengths)
        spins_left = np.expm1(pred_spins_left[0, :, 0].cpu().numpy())
        next_bet = np.expm1(pred_next_bet[0, :, 0].cpu().numpy())
        return spins_left, next_bet
