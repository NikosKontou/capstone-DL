import os
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
