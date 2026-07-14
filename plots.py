import os
import numpy as np
import matplotlib.pyplot as plt

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

def plot_feature_importance(importance: dict, out_dir: str, top_n: int = 20):
    """
    importance: output of evaluate.permutation_feature_importance --
    {feature_name: {"spins_left_importance": float, "next_bet_importance": float}}

    Draws one horizontal bar chart per prediction head, each sorted by
    that head's importance descending, showing only the top_n features
    so the plot stays readable when there are dozens of one-hot
    columns. A feature with near-zero or negative importance means
    shuffling it barely changed (or even slightly helped) validation
    loss -- i.e. the model isn't relying on it much for that head.
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

        plt.figure(figsize=(8, max(4, 0.35 * len(top_features))))
        colors = ["#d62728" if s < 0 else "#1f77b4" for s in top_scores]
        y_pos = np.arange(len(top_features))
        plt.barh(y_pos, top_scores, color=colors)
        plt.yticks(y_pos, top_features, fontsize=8)
        plt.gca().invert_yaxis()  # highest importance at the top
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("Increase in masked MSE loss when shuffled")
        plt.title(f"Permutation Feature Importance: {head_label}")
        plt.grid(True, axis="x", linestyle="--", alpha=0.5)

        plt.savefig(os.path.join(out_dir, f"feature_importance_{fname}.png"), bbox_inches="tight")
        plt.close()
