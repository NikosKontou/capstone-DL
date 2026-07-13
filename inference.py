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
        self._base_numeric = [c for c in cfg.NUMERIC_FEATURES]

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()

        # Apply cyclical encoding matching preprocessing
        if 'hour_of_day' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24.0)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24.0)
        if 'day_of_week' in df.columns:
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)

        out = df[self._base_numeric].copy()
        for col in cfg.ADD_MISSING_INDICATORS:
            missing = out[col].isna()
            out[f"{col}_was_missing"] = missing.astype(np.float32)
            out[col] = out[col].fillna(self.medians[col])

        out = out.fillna(0.0)
        out = out[[c for c in self.feature_names if c in out.columns]]

        scaled = (out.values.astype(np.float32) - self.means[:out.shape[1]]) / self.stds[:out.shape[1]]
        onehot = df[cfg.ONEHOT_FEATURES].fillna(0.0).values.astype(np.float32)
        return np.concatenate([scaled, onehot], axis=1)

    @torch.no_grad()
    def predict(self, df: pd.DataFrame):
        X = self._build_features(df)
        x_t = torch.from_numpy(X).float().to(self.device)
        pred_spins_left, pred_next_bet = self.model(x_t)

        # Revert log1p transforms
        return (
            np.expm1(pred_spins_left.cpu().numpy().ravel()),
            np.expm1(pred_next_bet.cpu().numpy().ravel()),
        )
