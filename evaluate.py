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


@torch.no_grad()
def permutation_feature_importance(model, loader, device, feature_names: list,
                                     n_repeats: int = 3, seed: int = 42,
                                     max_batches: int = 20):
    """
    Model-agnostic feature importance for the sequence model (there's no
    built-in importance like a tree model has for an LSTM). For each
    feature column, shuffle that column's values across the batch
    dimension at every valid timestep, independently per batch,
    independently per repeat -- this breaks the feature's association
    with the target while leaving every other feature's values, and
    the temporal structure of the sequence, untouched. Importance is
    how much validation loss gets worse when that feature is
    shuffled: a feature the model actually relies on will hurt more
    when scrambled than a feature it ignores.

    Computed separately for each head (spins_left, next_bet) since a
    feature can matter to one task and not the other. Reuses the same
    masked-MSE loss as training so this is measuring exactly what the
    model was optimized against, not a proxy metric.

    Shuffling is done PER BATCH rather than once globally: this keeps
    memory bounded (we never materialize a full shuffled dataset) and
    still gives each feature many independent permutations across
    n_repeats x n_batches, which is what the importance scores are
    averaged over.

    Cost note: this runs n_features x n_repeats extra forward passes
    per batch (~40 features x 3 repeats = 120x a normal eval pass), so
    it's capped at max_batches batches by default rather than running
    over the full loader -- pass a validation loader with a smaller
    batch_size, or raise max_batches, if more precision is needed.

    Returns a dict: {feature_name: {"spins_left_importance": float,
    "next_bet_importance": float}}, where importance is the mean
    increase in masked MSE loss (post-shuffle minus baseline) across
    all repeats and batches. Higher = more important.
    """
    from trainer import masked_mse  # local import: avoids a circular import at module load time

    model.eval()
    rng = np.random.default_rng(seed)
    n_features = len(feature_names)

    baseline_spins_loss = 0.0
    baseline_bet_loss = 0.0
    shuffled_spins_loss = np.zeros(n_features)
    shuffled_bet_loss = np.zeros(n_features)
    n_batches = 0

    # Cache a bounded number of batches so the same baseline forward
    # pass and the same shuffled inputs are reused across every
    # feature column, and so cost stays predictable regardless of
    # dataset size.
    batches = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batches.append(batch)

    for batch in batches:
        x = batch["x"].to(device)
        lengths = batch["lengths"]
        y_spins_left = batch["y_spins_left"].to(device)
        y_next_bet = batch["y_next_bet"].to(device)
        next_bet_mask = batch["next_bet_mask"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        pred_spins_left, pred_next_bet = model(x, lengths)
        baseline_spins_loss += masked_mse(pred_spins_left, y_spins_left, padding_mask).item()
        baseline_bet_loss += masked_mse(pred_next_bet, y_next_bet, next_bet_mask).item()
        n_batches += 1

        B = x.shape[0]
        for f in range(n_features):
            for _ in range(n_repeats):
                x_shuffled = x.clone()
                perm = torch.from_numpy(rng.permutation(B))
                # Shuffle this feature's values across sessions in the
                # batch; padded positions get shuffled too but never
                # affect the loss since padding_mask/next_bet_mask
                # exclude them regardless.
                x_shuffled[:, :, f] = x[perm, :, f]

                p_spins, p_bet = model(x_shuffled, lengths)
                shuffled_spins_loss[f] += masked_mse(p_spins, y_spins_left, padding_mask).item()
                shuffled_bet_loss[f] += masked_mse(p_bet, y_next_bet, next_bet_mask).item()

    baseline_spins_loss /= n_batches
    baseline_bet_loss /= n_batches
    shuffled_spins_loss /= (n_batches * n_repeats)
    shuffled_bet_loss /= (n_batches * n_repeats)

    importance = {}
    for f, name in enumerate(feature_names):
        importance[name] = {
            "spins_left_importance": float(shuffled_spins_loss[f] - baseline_spins_loss),
            "next_bet_importance": float(shuffled_bet_loss[f] - baseline_bet_loss),
        }
    return importance


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
