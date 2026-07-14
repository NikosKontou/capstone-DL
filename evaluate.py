import numpy as np
import torch


@torch.no_grad()
def predict(model, loader, device):
    """
    Runs the model over every batch and returns flattened (all valid
    steps, across all sessions) predictions/targets plus a same-shape
    boolean mask for each head. "Valid" for spins_left is any real
    (non-padding) step; for next_bet it's additionally non-terminal.

    Also returns last-step arrays for comparison against the row-level
    baseline: for spins_left, the last real (non-padding) timestep of
    each session; for next_bet, the last timestep with a VALID target
    (i.e. the last non-terminal spin), since the literal last real
    timestep of a session is always the terminal spin and therefore
    never has a next_bet target by construction.
    """
    model.eval()
    preds_spins_left, preds_next_bet = [], []
    true_spins_left, true_next_bet = [], []
    padding_masks, next_bet_masks = [], []

    last_pred_spins_left, last_true_spins_left = [], []
    last_pred_next_bet, last_true_next_bet = [], []

    for batch in loader:
        x = batch["x"].to(device)
        lengths = batch["lengths"]
        p_spins, p_bet = model(x, lengths)

        preds_spins_left.append(p_spins.cpu().numpy())
        preds_next_bet.append(p_bet.cpu().numpy())
        true_spins_left.append(batch["y_spins_left"].numpy())
        true_next_bet.append(batch["y_next_bet"].numpy())
        padding_masks.append(batch["padding_mask"].numpy())
        next_bet_masks.append(batch["next_bet_mask"].numpy())

        # Last real timestep per session: index (length - 1) along dim 1.
        # Used for the spins_left "last step" comparison.
        lengths_np = lengths.numpy()
        batch_idx = np.arange(len(lengths_np))
        last_idx = lengths_np - 1

        p_spins_np = p_spins.cpu().numpy()
        p_bet_np = p_bet.cpu().numpy()
        y_spins_np = batch["y_spins_left"].numpy()
        y_bet_np = batch["y_next_bet"].numpy()
        next_bet_mask_np = batch["next_bet_mask"].numpy()[:, :, 0]  # (B, T)

        last_pred_spins_left.append(p_spins_np[batch_idx, last_idx, 0])
        last_true_spins_left.append(y_spins_np[batch_idx, last_idx, 0])

        # The literal last real step of a session is always the terminal
        # spin (next_bet_mask is 0 there by construction -- there is no
        # "next spin" to predict). The comparable "last step" for the
        # next_bet head is instead the LAST step with a valid target,
        # i.e. the second-to-last real step of the session (or earlier,
        # if masking removed more than just the terminal step).
        for b in range(len(lengths_np)):
            valid_steps = np.where(next_bet_mask_np[b, :lengths_np[b]] > 0)[0]
            if len(valid_steps) == 0:
                continue  # session had no valid next_bet target at all (e.g. length 1)
            t = valid_steps[-1]
            last_pred_next_bet.append(p_bet_np[b, t, 0])
            last_true_next_bet.append(y_bet_np[b, t, 0])

    flat = {
        "pred_spins_left": np.concatenate([a.ravel() for a in preds_spins_left]),
        "pred_next_bet": np.concatenate([a.ravel() for a in preds_next_bet]),
        "true_spins_left": np.concatenate([a.ravel() for a in true_spins_left]),
        "true_next_bet": np.concatenate([a.ravel() for a in true_next_bet]),
        "padding_mask": np.concatenate([a.ravel() for a in padding_masks]).astype(bool),
        "next_bet_mask": np.concatenate([a.ravel() for a in next_bet_masks]).astype(bool),
    }
    last_step = {
        "pred_spins_left": np.concatenate(last_pred_spins_left),
        "true_spins_left": np.concatenate(last_true_spins_left),
        "pred_next_bet": np.array(last_pred_next_bet, dtype=np.float32),
        "true_next_bet": np.array(last_true_next_bet, dtype=np.float32),
    }
    return flat, last_step


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
    flat, last_step = predict(model, loader, device)

    # Per-step metrics (every valid timestep in every session)
    p_spins = np.expm1(flat["pred_spins_left"][flat["padding_mask"]])
    y_spins = np.expm1(flat["true_spins_left"][flat["padding_mask"]])
    p_bet = np.expm1(flat["pred_next_bet"][flat["next_bet_mask"]])
    y_bet = np.expm1(flat["true_next_bet"][flat["next_bet_mask"]])

    metrics = {"per_step": {}, "last_step": {}}
    metrics["per_step"].update(regression_metrics(y_spins, p_spins, "spins_left_in_session"))
    metrics["per_step"].update(regression_metrics(y_bet, p_bet, "next_spin_amount"))
    metrics["per_step"]["spins_left_in_session_baseline_MAE_median"] = float(
        np.mean(np.abs(y_spins - np.median(y_spins)))
    )
    metrics["per_step"]["next_spin_amount_baseline_MAE_median"] = float(
        np.mean(np.abs(y_bet - np.median(y_bet)))
    )

    # Last-real-timestep-only metrics for spins_left, and
    # last-valid-target-step-only metrics for next_bet, comparable to
    # the row-level model.
    p_spins_last = np.expm1(last_step["pred_spins_left"])
    y_spins_last = np.expm1(last_step["true_spins_left"])
    p_bet_last = np.expm1(last_step["pred_next_bet"])
    y_bet_last = np.expm1(last_step["true_next_bet"])

    metrics["last_step"].update(regression_metrics(y_spins_last, p_spins_last, "spins_left_in_session"))
    if len(y_bet_last) > 0:
        metrics["last_step"].update(regression_metrics(y_bet_last, p_bet_last, "next_spin_amount"))

    plot_data = {
        "spins_true": y_spins, "spins_pred": p_spins,
        "bet_true": y_bet, "bet_pred": p_bet,
    }

    return metrics, plot_data
