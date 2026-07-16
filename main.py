"""
Orchestrator: load data -> build session sequences -> train -> evaluate -> save artifacts.

Usage:
    python main.py [--data-dir DIR] [--epochs N] [--size PCT]

    --size PCT   Keep only PCT% of the dataset, by randomly sampling whole
                 accounts (not rows), so sessions stay intact. Omit to
                 use the full dataset.
"""
import argparse
import json
import os
from datetime import datetime

import torch

import config as cfg
from data_loader import load_splits, build_datasets, build_dataloaders
from model import build_model
from trainer import train_model
from evaluate import evaluate_model, permutation_feature_importance
from plots import (
    plot_learning_curves, plot_predictions, plot_feature_importance,
    plot_residuals, plot_per_step_vs_last_step, plot_error_vs_true_value,
)


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=cfg.DATA_DIR,
                         help="Directory containing train/valid/test parquet files")
    parser.add_argument("--train-file", default=cfg.TRAIN_FILE)
    parser.add_argument("--valid-file", default=cfg.VALID_FILE)
    parser.add_argument("--test-file", default=cfg.TEST_FILE)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--output-dir", default=cfg.OUTPUT_DIR)
    parser.add_argument("--size", type=float, default=None,
                         help="Percent of accounts to keep, e.g. 10 for 10%%. "
                              "Omit to use the full dataset.")
    args = parser.parse_args()

    cfg.EPOCHS = args.epochs

    timestamp = datetime.now().strftime("%d%m%y-%H-%M-%S")
    run_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    set_seed(cfg.RANDOM_SEED)

    print(f"Loading data from {args.data_dir} "
          f"({args.train_file} / {args.valid_file} / {args.test_file}) ...")
    if args.size is not None:
        print(f"Subsampling to ~{args.size}% of accounts (per split, filtered during read) ...")
    splits = load_splits(args.data_dir, args.train_file, args.valid_file, args.test_file,
                          size_pct=args.size, seed=cfg.RANDOM_SEED)
    for name, d in splits.items():
        print(f"  {name}: {len(d):,} rows")

    print("Building preprocessed session sequences ...")
    datasets, preprocessor = build_datasets(splits)
    for name, ds in datasets.items():
        print(f"  {name}: {len(ds):,} sessions")
    loaders = build_dataloaders(datasets)

    n_features = preprocessor.n_features
    print(f"Feature count after preprocessing: {n_features}")

    model = build_model(n_features)
    print(model)

    model, history = train_model(model, loaders, output_dir=run_dir)

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print("Evaluating on test split ...")

    test_metrics, plot_data = evaluate_model(model, loaders["test"], device)
    print(json.dumps(test_metrics, indent=2))

    print("Computing permutation feature importance on the validation split ...")
    # Uses the validation loader, not test: keeps test purely for the
    # final reported metrics, and validation is typically the smaller
    # split anyway which keeps this (relatively expensive) computation
    # fast. See evaluate.permutation_feature_importance's docstring
    # for the cost tradeoff and max_batches cap.
    feature_importance = permutation_feature_importance(
        model, loaders["valid"], device, feature_names=preprocessor.all_feature_names
    )

    plot_learning_curves(history, run_dir)
    plot_predictions(plot_data["spins_true"], plot_data["spins_pred"], "spins_left", run_dir)
    plot_predictions(plot_data["bet_true"], plot_data["bet_pred"], "next_bet", run_dir)
    plot_residuals(plot_data["spins_true"], plot_data["spins_pred"], "spins_left", run_dir)
    plot_residuals(plot_data["bet_true"], plot_data["bet_pred"], "next_bet", run_dir)
    plot_error_vs_true_value(plot_data["spins_true"], plot_data["spins_pred"], "spins_left", run_dir)
    plot_error_vs_true_value(plot_data["bet_true"], plot_data["bet_pred"], "next_bet", run_dir)
    plot_per_step_vs_last_step(test_metrics, run_dir)
    plot_feature_importance(feature_importance, run_dir)

    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    with open(os.path.join(run_dir, "feature_importance.json"), "w") as f:
        json.dump(feature_importance, f, indent=2)

    torch.save({
        "model_state_dict": model.state_dict(),
        "n_features": n_features,
        "feature_names": preprocessor.all_feature_names,
        "preprocessor_medians": preprocessor.medians_,
        "preprocessor_means": preprocessor.means_.tolist(),
        "preprocessor_stds": preprocessor.stds_.tolist(),
    }, os.path.join(run_dir, "spin_model.pt"))

    print(f"Saved model, metrics, and plots to {run_dir}")


if __name__ == "__main__":
    main()
