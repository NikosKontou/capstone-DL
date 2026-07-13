# Per-Spin Multi-Task Deep Learning Model

Predicts, for every spin row in `full_merged_sorted_data.parquet`:

1. **`spins_left_in_session`** — regression, count of remaining spins in the session
2. **`next_spin_amount`** — regression, bet size of the following spin (NULL/undefined on the terminal spin of each session)

This slots into the existing pipeline described in the technical summary — same split-aware feature set, same `split_name` train/valid/test partitioning — as a third, per-spin sibling to the `will_return_7d` classifier and the `time_to_next_session_day` regressor.

## Why a multi-task MLP

Every row already carries the lag-1 spin features (`prev_bet`, `prev_win`, `prev_rtp_bucket_id`) and expanding-window player-history features computed in SQL, so the within-session temporal signal is already flattened into the feature vector — a plain tabular network is the right first model, not a sequence model. Section 8 (future work) already earmarks LSTM/Transformer upgrades if longer within-session dependencies turn out to matter; this architecture is the natural stepping stone.

The two targets share a trunk because they're correlated (a player about to leave typically also isn't placing another bet) — joint training lets the shared representation pick up on that.

## Files

| File | Purpose |
|---|---|
| `config.py` | All tunable constants — paths, feature lists, architecture, training hyperparameters |
| `data_loader.py` | Preprocessing (train-only fit scalers/medians), missing-indicator construction, `Dataset`/`DataLoader` |
| `model.py` | `SpinMultiTaskNet` — shared MLP trunk + two Softplus regression heads |
| `trainer.py` | Training loop: masked loss, early stopping, `ReduceLROnPlateau`, gradient clipping |
| `evaluate.py` | MAE/RMSE/R²/within-1-unit metrics, masked correctly for `next_spin_amount` |
| `main.py` | Orchestrator — run this to train end-to-end |
| `inference.py` | `SpinPredictor` class for scoring new spin rows from a saved checkpoint |
| `make_sample_data.py` | Generates synthetic data matching your schema, for smoke-testing only |

## Key design decisions

- **Masked loss for `next_spin_amount`.** The terminal spin of every session has no "next spin" (NULL in the source data). Rather than dropping those rows — which would throw away the single most informative row per session for `spins_left_in_session` (where the target is exactly 0) — the row is kept, and a `next_bet_mask` tensor zeroes out its contribution to the `next_spin_amount` loss only. Both heads therefore see full data for their own task.
- **Missing-value indicators, not just imputation.** `prev_bet`, `prev_win`, `prev_rtp_bucket_id`, and `days_since_last_session` are `NaN` on the first spin of a session by construction (no prior spin exists). These are median-imputed *and* paired with a `_was_missing` binary flag, so the network can distinguish "first spin of session" from "observed value happens to equal the median."
- **Train-only preprocessing fit.** Scaler means/stds and imputation medians are fit on `split_name == 'train'` only and applied unchanged to `valid`/`test`, consistent with the leakage-avoidance design already used for the player-history window features in the SQL layer.
- **Softplus output heads.** Both targets are non-negative counts/amounts. Softplus keeps predictions ≥ 0 without the dead-gradient problem a ReLU output can have.
- **One-hot columns passed through as-is.** `game_id`, `game_provider_id`, `currency_id`, `device_id` are already one-hot encoded upstream in SQL (per Section 3.6), so they bypass the numeric scaler and go straight into the input vector.

## Running on the real dataset

Train/valid/test live in **three separate parquet files** (not one file with a `split_name` column to filter on), matching your actual pipeline output.

1. Point `config.DATA_DIR` (or `--data-dir`) at the directory holding `train.parquet`, `valid.parquet`, `test.parquet` — or pass `--train-file` / `--valid-file` / `--test-file` if they're named differently. Each file must contain every column in `config.NUMERIC_FEATURES`, `config.ONEHOT_FEATURES`, `spins_left_in_session`, and `next_spin_amount`.
2. `python main.py --data-dir /path/to/splits --epochs 60`
3. Artifacts land in `outputs/`: `spin_model.pt` (weights + preprocessing state), `history.json` (per-epoch losses), `test_metrics.json`.
4. Score new rows with `inference.SpinPredictor("outputs/spin_model.pt").predict(df)`.

### Working with a subset of the data

The full dataset can be large enough that you want to iterate on a smaller slice first. `--size PCT` keeps only `PCT`% of the data:

```bash
python main.py --data-dir /path/to/splits --size 10   # ~10% of the data
```

Subsampling is done **by account, not by row**, independently within each split, and the filtering happens *during* the parquet read rather than after loading the full file into memory. An account's spins span one or more full sessions, and slicing rows directly would cut a session mid-stream — corrupting `spins_left_in_session` (which counts down to the true end of the session) and the `prev_*` lag features for whatever row happened to land at the cut. Sampling whole accounts keeps every kept session fully intact, and train/valid/test each keep their original relative proportions since the sampling happens per split.

Under the hood: only the `account` column is read first (a cheap columnar read) to decide which accounts to keep, then the rest of each file is read via a PyArrow dataset scan with an `account.isin(keep_accounts)` filter pushed down to the reader -- row groups containing no kept accounts are skipped, and only matching rows are ever materialized into a DataFrame. The full unfiltered file is never held in memory at once, which matters once the real dataset no longer fits comfortably in RAM.

Omit `--size` to train on the full dataset.

If your files also happen to carry a `split_name` column, `load_splits()` checks it against the file each row came from and raises an error on any mismatch — this catches the class of bug where a file gets loaded into the wrong slot (e.g. `valid.parquet` accidentally pointed to by `--test-file`). It is not used to filter rows.

## Smoke test

`make_sample_data.py` generates synthetic `train.parquet` / `valid.parquet` / `test.parquet` with the same columns/dtypes so you can verify the pipeline runs before pointing it at the full 2.4M-row dataset:

```bash
python make_sample_data.py     # writes train.parquet, valid.parquet, test.parquet
python main.py --epochs 15     # trains on the synthetic data
python inference.py            # scores 10 rows and prints predictions vs ground truth
```

On synthetic data (session length generated independently of features, so this understates real performance): `spins_left_in_session` MAE ≈ 6.6 vs a 7.7 median baseline, `next_spin_amount` R² ≈ 0.97 (bet amount is highly self-correlated within a session in the synthetic generator). Expect materially better `spins_left_in_session` performance on the real data, where session length correlates with `player_avg_round_duration`, `hour_of_day`, `bet_change_direction_id`, and other genuine behavioural signal that the synthetic generator doesn't encode.

## Tuning knobs worth trying first

- `LOSS_WEIGHT_SPINS_LEFT` / `LOSS_WEIGHT_NEXT_BET` in `config.py` — rebalance if one head dominates the shared trunk's gradients (watch the per-head MSE printed each epoch).
- `HIDDEN_DIMS` — current `[256, 128, 64]` is a reasonable default for ~45 input features; widen if underfitting on the full 2.4M rows.
- Consider log1p-transforming `next_spin_amount` before training (mirroring the `log1p(days)` treatment already used for `time_to_next_session_day` in Section 5.2) if bet-amount distributions turn out to be as right-skewed as the session-gap target.
