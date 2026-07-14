# Spin Sequence Model

Predicts, for a player's in-progress slot session, **how many spins are
left in the session** and **the size of the next bet** — as a running
time series that updates after every spin, not a single snapshot
prediction.

This is a sequence model: it reads a session's spins in order (bet
size, RTP outcome, timing, etc.) and carries what it's seen forward
through an LSTM, rather than relying on hand-engineered lag features
computed one step at a time.

---

## Pipeline overview

```
build_features.sql          DuckDB feature pipeline (raw events -> parquet)
        |
        v
config.py       -- all paths, feature lists, and hyperparameters
data_loader.py  -- groups rows into per-session sequences, preprocesses,
                    pads/collates batches for the LSTM
model.py        -- SpinMultiTaskNet: LSTM + two prediction heads
trainer.py      -- training loop, masked multi-task loss
evaluate.py     -- test-set metrics (per-step and last-step)
inference.py    -- SpinPredictor: serve predictions on a live session
plots.py        -- learning curves + prediction scatter plots
main.py         -- orchestrates all of the above end to end
```

Run a full training job with:

```bash
python main.py
python main.py --size 10        # quick run on ~10% of accounts
python main.py --epochs 30 --data-dir /path/to/parquet/dir
```

Each run writes a timestamped folder under `OUTPUT_DIR` (see
`config.py`) containing `spin_model.pt` (the trained checkpoint),
`history.json`, `test_metrics.json`, and the plots.

---

## What "sequence model" means here, concretely

**Unit of training data:** one *session* (all of a player's spins from
session start to session end, or the most recent `MAX_SEQ_LEN=64`
spins for very long sessions), not one spin. Each session is a matrix
of shape `(T, F)` — `T` spins, `F` features per spin.

**No lag features.** The old row-level model computed `prev_bet`,
`prev_win`, `prev_rtp_bucket_id`, `bet_change_direction_id` upstream in
SQL as hand-crafted "memory." Those are gone — an LSTM sees the raw
per-step sequence directly and learns any such relationship itself, so
computing them separately was both redundant and slower to build.
`inter_spin_gap_sec` is the one exception kept, since it's a genuine
computed gap the model can't reconstruct from the other retained
columns.

**Causal by construction.** The LSTM is unidirectional
(`bidirectional=False` in `model.py`) — its prediction at spin `t` is
mathematically a function of spins `1..t` only, never spins after `t`.
The player-level aggregate features (`player_avg_bet`,
`player_round_count`, etc.) are also computed upstream using `ROWS
BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`, so they only reflect
history strictly before the current row. No future leakage anywhere in
the stack.

**Per-step predictions, per-step loss.** The model predicts
`spins_left` and `next_bet` at *every* timestep in a session, not just
the last one. Training backpropagates a loss at every valid timestep
in every session in a batch — a single 20-spin session contributes up
to 20 training signals for `spins_left` and up to 19 for `next_bet`
(all but the terminal spin, which has no "next spin" to predict).

**Padding and masking.** Batches contain sessions of different
lengths, so shorter ones are zero-padded up to the longest session in
the batch. Two masks (`padding_mask`, `next_bet_mask`) ensure padded
and terminal-spin positions never contribute to the loss —
`pack_padded_sequence`/`pad_packed_sequence` also keep padding out of
the LSTM's recurrence entirely, not just out of the loss.

---

## Serving / inference

`inference.py`'s `SpinPredictor` has a different contract than a
row-level model would: `predict()` takes a **session's spin history so
far** (a DataFrame of consecutive spins, oldest first), not a single
row, because the LSTM needs that history in its hidden state to
produce a meaningful prediction.

```python
from inference import SpinPredictor

predictor = SpinPredictor("path/to/spin_model.pt")

# session_so_far: DataFrame of this session's spins, oldest first
spins_left, next_bet = predictor.predict(session_so_far)

# Optional: predictions at every step of the session, not just the last
spins_left_seq, next_bet_seq = predictor.predict_all_steps(session_so_far)
```

A single-spin history (session just started) is valid input — the
LSTM simply has no prior hidden state to draw on yet, same as during
training.

---

## Evaluation

`evaluate.py` reports two metric sets in `test_metrics.json`:

- **`per_step`** — computed over every valid timestep across every
  test session. Uses the most data, reflects the model's typical
  accuracy at an arbitrary point in a session.
- **`last_step`** — computed only at the last real timestep of each
  session (for `spins_left`) or the last timestep with a valid target
  (for `next_bet`, since the literal last spin of a session never has
  a "next spin"). This is the metric to compare against a row-level
  baseline model, since it's the closest match to "one prediction per
  session, as late as possible."

---

## Feature pipeline (`build_features.sql`)

Builds `train/valid/test_features_fix.parquet` from raw spin events in
DuckDB. Notable behavior:

- Sessions containing a very long round (>500s) that **isn't** the
  final spin of the session are dropped entirely (likely a logging
  gap or AFK period, not real play).
- `next_session_bucket_id` buckets time-to-next-session into same-day
  / next-day / within-a-week / within-two-weeks / within-a-month /
  churned, mirroring the style of `rtp_bucket_id`.
- Session-recency window functions (`LAG`/`LEAD` over
  `session_start`) break ties on `sessionid`, so two sessions with an
  identical start time for the same account resolve deterministically
  instead of producing arbitrary `NULL`s.
- `is_first_session` / `is_last_session` flags make it explicit when
  `days_since_last_session` or `time_to_next_session_*` are `NULL` by
  genuine absence of a prior/next session, rather than leaving that to
  be silently inferred downstream.
- Output is ordered by `account, starttime` (not just `starttime`),
  since the Python pipeline needs rows grouped by account, in time
  order, to build session sequences.

---

## Known limitations / things to check before trusting results

- `MAX_SEQ_LEN=64` truncates long sessions to their most recent 64
  spins; very long sessions lose their earliest spins from the input
  window entirely.
- `MIN_SEQ_LEN=2` drops single-spin sessions, since there's no
  meaningful sequence to learn from.
- The `last_step` metrics are the fairest comparison point against the
  old row-level model's numbers — run both and compare before
  concluding the sequence model is actually an improvement.
- This has only been validated end-to-end against synthetic data
  matching the parquet schema, not the real dataset. Run `main.py` on
  a small `--size` sample of the real data first.
