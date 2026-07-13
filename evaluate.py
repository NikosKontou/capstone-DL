import numpy as np
import torch

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds_spins_left, preds_next_bet = [], []
    true_spins_left, true_next_bet, masks = [], [], []

    for batch in loader:
        x = batch["x"].to(device)
        p_spins, p_bet = model(x)

        preds_spins_left.append(p_spins.cpu().numpy())
        preds_next_bet.append(p_bet.cpu().numpy())
        true_spins_left.append(batch["y_spins_left"].numpy())
        true_next_bet.append(batch["y_next_bet"].numpy())
        masks.append(batch["next_bet_mask"].numpy())

    return (
        np.concatenate(preds_spins_left).ravel(),
        np.concatenate(preds_next_bet).ravel(),
        np.concatenate(true_spins_left).ravel(),
        np.concatenate(true_next_bet).ravel(),
        np.concatenate(masks).ravel().astype(bool),
    )

def regression_metrics(y_true, y_pred, name):
    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    within_1 = np.mean(np.abs(err) <= 1.0)
    return {
        f"{name}_MAE": float(mae),
        f"{name}_RMSE": float(rmse),
        f"{name}_R2": float(r2),
        f"{name}_within_1_unit_pct": float(within_1 * 100),
    }

def evaluate_model(model, loader, device):
    p_spins, p_bet, y_spins, y_bet, mask = predict(model, loader, device)

    # Expm1 transform to return values to original scale for accurate metric reporting
    p_spins = np.expm1(p_spins)
    p_bet = np.expm1(p_bet)
    y_spins = np.expm1(y_spins)
    y_bet = np.expm1(y_bet)

    metrics = {}
    metrics.update(regression_metrics(y_spins, p_spins, "spins_left_in_session"))
    metrics.update(regression_metrics(y_bet[mask], p_bet[mask], "next_spin_amount"))

    baseline_spins_mae = float(np.mean(np.abs(y_spins - np.median(y_spins))))
    baseline_bet_mae = float(np.mean(np.abs(y_bet[mask] - np.median(y_bet[mask]))))
    metrics["spins_left_in_session_baseline_MAE_median"] = baseline_spins_mae
    metrics["next_spin_amount_baseline_MAE_median"] = baseline_bet_mae

    plot_data = {
        "spins_true": y_spins, "spins_pred": p_spins,
        "bet_true": y_bet[mask], "bet_pred": p_bet[mask]
    }

    return metrics, plot_data
