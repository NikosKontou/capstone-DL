"""
Training loop for the sequence (time-series) multi-task spin model.

Loss = w1 * masked_MSE(spins_left_pred, spins_left_true, padding_mask)
     + w2 * masked_MSE(next_bet_pred, next_bet_true, padding_mask & next_bet_mask)

Two masks are needed now (unlike the row-level model, which only
needed one): padding_mask excludes padded timesteps from *both* heads,
and next_bet_mask additionally excludes terminal-spin timesteps (no
ground truth "next spin") from the next_bet head only. These are
combined once in the data loader's collate_fn into
`next_bet_mask` (already padding-aware) and `padding_mask` (for the
spins_left head), so this file just consumes them directly.
"""
import copy
import json
import os
import time

import torch
import torch.nn as nn

import config as cfg


def masked_mse(pred, target, mask):
    sq_err = (pred - target) ** 2 * mask
    denom = mask.sum().clamp(min=1.0)
    return sq_err.sum() / denom


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss, total_spins_loss, total_bet_loss, n_batches = 0.0, 0.0, 0.0, 0

    for batch in loader:
        x = batch["x"].to(device)
        lengths = batch["lengths"]  # stays on CPU, pack_padded_sequence requires it
        y_spins_left = batch["y_spins_left"].to(device)
        y_next_bet = batch["y_next_bet"].to(device)
        next_bet_mask = batch["next_bet_mask"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        with torch.set_grad_enabled(train):
            pred_spins_left, pred_next_bet = model(x, lengths)
            loss_spins = masked_mse(pred_spins_left, y_spins_left, padding_mask)
            loss_bet = masked_mse(pred_next_bet, y_next_bet, next_bet_mask)
            loss = (cfg.LOSS_WEIGHT_SPINS_LEFT * loss_spins
                    + cfg.LOSS_WEIGHT_NEXT_BET * loss_bet)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
                optimizer.step()

        total_loss += loss.item()
        total_spins_loss += loss_spins.item()
        total_bet_loss += loss_bet.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "spins_left_mse": total_spins_loss / n_batches,
        "next_bet_mse": total_bet_loss / n_batches,
    }


def train_model(model, loaders, output_dir: str = None):
    output_dir = output_dir or cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE,
                                  weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.LR_SCHEDULER_FACTOR,
        patience=cfg.LR_SCHEDULER_PATIENCE
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    print(f"Training on device: {device}")
    t0 = time.time()

    for epoch in range(1, cfg.EPOCHS + 1):
        train_metrics = run_epoch(model, loaders["train"], optimizer, device, train=True)
        val_metrics = run_epoch(model, loaders["valid"], optimizer, device, train=False)
        scheduler.step(val_metrics["loss"])

        history.append({"epoch": epoch, "train": train_metrics, "valid": val_metrics,
                         "lr": optimizer.param_groups[0]["lr"]})

        print(f"[{epoch:03d}] train_loss={train_metrics['loss']:.4f} "
              f"(spins={train_metrics['spins_left_mse']:.4f}, bet={train_metrics['next_bet_mse']:.4f}) "
              f"| val_loss={val_metrics['loss']:.4f} "
              f"(spins={val_metrics['spins_left_mse']:.4f}, bet={val_metrics['next_bet_mse']:.4f}) "
              f"| lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_metrics["loss"] < best_val_loss - 1e-5:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for "
                      f"{cfg.EARLY_STOPPING_PATIENCE} epochs).")
                break

    elapsed = time.time() - t0
    print(f"Training finished in {elapsed:.1f}s. Best val_loss={best_val_loss:.4f}")

    model.load_state_dict(best_state)

    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return model, history
