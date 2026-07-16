import os
import numpy as np
import matplotlib.pyplot as plt

def plot_residuals(y_true, y_pred, name: str, out_dir: str):
    """
    Histogram of (predicted - true) on the test set. Complements the
    scatter plot: a scatter can look reasonable on the diagonal while
    still hiding a systematic bias that only shows up as an
    off-center residual distribution. A histogram centered at 0 with
    a tight, roughly symmetric spread indicates well-calibrated
    predictions; a skew indicates the model consistently over- or
    under-predicts.
    """
    residuals = y_pred - y_true
    plt.figure(figsize=(7, 4.5))
    plt.hist(residuals, bins=60, color="#1f77b4", edgecolor="none")
    plt.axvline(0, color="red", linewidth=1, linestyle="--", label="zero error")
    plt.axvline(np.median(residuals), color="black", linewidth=1,
                linestyle=":", label=f"median = {np.median(residuals):.2f}")
    plt.xlabel("Predicted - True")
    plt.ylabel("Count")
    plt.title(f"Residual Distribution: {name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(os.path.join(out_dir, f"{name}_residuals.png"), bbox_inches="tight")
    plt.close()


def plot_per_step_vs_last_step(metrics: dict, out_dir: str):
    """
    Bar chart comparing per_step vs last_step MAE/RMSE for both heads,
    read directly from the test_metrics dict evaluate_model already
    produces. The README names last_step as "the metric to compare
    against a row-level baseline," but until now that comparison only
    existed as two separate JSON blocks -- this plot puts the two
    side by side so the gap (or lack of one) between "typical mid-
    session accuracy" and "accuracy right before the moment that
    matters" is visible at a glance, without cross-referencing numbers
    by hand.
    """
    heads = [
        ("spins_left_in_session", "Spins Left in Session"),
        ("next_spin_amount", "Next Bet Amount"),
    ]
    metric_keys = ["MAE", "RMSE"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (head_key, head_label) in zip(axes, heads):
        per_step_vals = [metrics["per_step"].get(f"{head_key}_{m}", np.nan) for m in metric_keys]
        last_step_vals = [metrics["last_step"].get(f"{head_key}_{m}", np.nan) for m in metric_keys]

        x = np.arange(len(metric_keys))
        width = 0.35
        ax.bar(x - width / 2, per_step_vals, width, label="per_step", color="#1f77b4")
        ax.bar(x + width / 2, last_step_vals, width, label="last_step", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_keys)
        ax.set_title(head_label)
        ax.set_ylabel("Error")
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Per-step vs Last-step Test Error")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_step_vs_last_step.png"), bbox_inches="tight")
    plt.close(fig)


def plot_error_vs_true_value(y_true, y_pred, name: str, out_dir: str, n_bins: int = 15):
    """
    Mean absolute error as a function of the true target value,
    binned into n_bins equal-width bins across the observed range.
    Neither the scatter nor the residual histogram shows whether
    error is uniform across the target's range or concentrated at one
    end -- this is the plot that would reveal, e.g., "spins_left
    predictions are fine early in a session but degrade badly as a
    session gets very long," which a single aggregate MAE number
    cannot distinguish from uniformly-mediocre performance.
    """
    bin_edges = np.linspace(y_true.min(), y_true.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(y_true, bin_edges) - 1, 0, n_bins - 1)

    bin_centers, bin_mae, bin_counts = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        bin_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
        bin_mae.append(np.mean(np.abs(y_pred[mask] - y_true[mask])))
        bin_counts.append(mask.sum())

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(bin_centers, bin_mae, marker="o", color="#1f77b4", label="MAE")
    ax1.set_xlabel(f"True {name} (bin center)")
    ax1.set_ylabel("MAE", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.bar(bin_centers, bin_counts, width=(bin_edges[1] - bin_edges[0]) * 0.8,
            alpha=0.15, color="gray", label="sample count")
    ax2.set_ylabel("Sample count", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    plt.title(f"Error vs True Value: {name}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_error_by_magnitude.png"), bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(history: list, out_dir: str):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train"]["loss"] for h in history]
    val_loss = [h["valid"]["loss"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(os.path.join(out_dir, "learning_curves.png"), bbox_inches="tight")
    plt.close()

def plot_predictions(y_true, y_pred, name: str, out_dir: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=2)

    # Ideal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")

    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"True vs Predicted: {name}")

    plt.savefig(os.path.join(out_dir, f"{name}_scatter.png"), bbox_inches="tight")
    plt.close()

def plot_feature_importance(importance: dict, out_dir: str, top_n: int = 20,
                             dominance_ratio: float = 3.0):
    """
    importance: output of evaluate.permutation_feature_importance --
    {feature_name: {"spins_left_importance": float, "next_bet_importance": float}}

    Draws one horizontal bar chart per prediction head, sorted by that
    head's importance descending, showing only the top_n features so
    the plot stays readable when there are dozens of one-hot columns.
    A feature with near-zero or negative importance means shuffling it
    barely changed (or even slightly helped) validation loss -- i.e.
    the model isn't relying on it much for that head.

    Handles the common case where one feature's importance dwarfs
    every other one (e.g. betbaseamount for next-bet-amount): on a
    single linear axis, a bar 10x the size of the runner-up squashes
    every other bar down to a sliver a pixel or two wide, making the
    plot useless for comparing the remaining features to each other.
    When the top feature exceeds `dominance_ratio` times the second-
    place score, the plot is instead split into two side-by-side
    panels sharing one y-axis: a narrow panel on the left showing the
    dominant bar at its true, undistorted scale, and a wider panel on
    the right zoomed in to just past the second-place score, with a
    diagonal break mark on the boundary flagging that the x-axis is
    discontinuous. This keeps the dominant feature's true magnitude
    visible (so its outsized importance isn't hidden) while making
    every other bar's relative size legible again.
    """
    features = list(importance.keys())

    for head_key, head_label, fname in [
        ("spins_left_importance", "Spins Left in Session", "spins_left"),
        ("next_bet_importance", "Next Bet Amount", "next_bet"),
    ]:
        scores = np.array([importance[f][head_key] for f in features])
        order = np.argsort(scores)[::-1][:top_n]
        top_features = [features[i] for i in order]
        top_scores = scores[order]
        n = len(top_features)
        y_pos = np.arange(n)
        colors = ["#d62728" if s < 0 else "#1f77b4" for s in top_scores]

        is_dominant = (
            n >= 2 and top_scores[0] > 0 and top_scores[1] > 0
            and top_scores[0] >= dominance_ratio * top_scores[1]
        )

        if not is_dominant:
            plt.figure(figsize=(8, max(4, 0.35 * n)))
            plt.barh(y_pos, top_scores, color=colors)
            plt.yticks(y_pos, top_features, fontsize=8)
            plt.gca().invert_yaxis()
            plt.axvline(0, color="black", linewidth=0.8)
            plt.xlabel("Increase in masked MSE loss when shuffled")
            plt.title(f"Permutation Feature Importance: {head_label}")
            plt.grid(True, axis="x", linestyle="--", alpha=0.5)
            plt.savefig(os.path.join(out_dir, f"feature_importance_{fname}.png"),
                        bbox_inches="tight")
            plt.close()
            continue

        # Split-axis version: left panel = full scale (just wide enough
        # for the dominant bar's label), right panel = zoomed view of
        # everything else, capped just past the second-place score.
        zoom_max = top_scores[1] * 1.25
        fig, (ax_left, ax_right) = plt.subplots(
            1, 2, sharey=True, figsize=(9, max(4, 0.35 * n)),
            gridspec_kw={"width_ratios": [1, 3], "wspace": 0.08},
        )

        ax_left.barh(y_pos, top_scores, color=colors)
        ax_left.set_xlim(0, top_scores[0] * 1.08)
        ax_left.set_xticks([0, round(top_scores[0], 3)])
        ax_left.invert_yaxis()
        ax_left.set_yticks(y_pos)
        ax_left.set_yticklabels(top_features, fontsize=8)
        ax_left.grid(True, axis="x", linestyle="--", alpha=0.5)
        ax_left.spines["right"].set_visible(False)

        ax_right.barh(y_pos, top_scores, color=colors)
        ax_right.set_xlim(0, zoom_max)
        # ax_right.invert_yaxis()
        ax_right.axvline(0, color="black", linewidth=0.8)
        ax_right.tick_params(left=False, labelleft=False)
        ax_right.grid(True, axis="x", linestyle="--", alpha=0.5)
        ax_right.spines["left"].set_visible(False)
        # label the truncated dominant bar so its true value isn't lost
        # off the right edge of the zoomed panel
        ax_right.text(zoom_max * 0.99, y_pos[0], f"  {top_scores[0]:.3f} \u2192",
                       va="center", ha="right", fontsize=7, color="#1f77b4")

        # diagonal break marks on the seam between the two panels,
        # the standard convention for flagging a discontinuous axis
        d = 0.4
        break_kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8,
                             linestyle="none", color="k", mec="k", mew=1, clip_on=False)
        ax_left.plot([1], [0], transform=ax_left.transAxes, **break_kwargs)
        ax_left.plot([1], [1], transform=ax_left.transAxes, **break_kwargs)
        ax_right.plot([0], [0], transform=ax_right.transAxes, **break_kwargs)
        ax_right.plot([0], [1], transform=ax_right.transAxes, **break_kwargs)

        fig.supxlabel("Increase in masked MSE loss when shuffled", fontsize=10)
        fig.suptitle(f"Permutation Feature Importance: {head_label}\n"
                     f"(axis break: {features[order[0]]} is "
                     f"{top_scores[0] / top_scores[1]:.1f}\u00d7 the runner-up)",
                     fontsize=11)
        fig.savefig(os.path.join(out_dir, f"feature_importance_{fname}.png"),
                    bbox_inches="tight")
        plt.close(fig)
