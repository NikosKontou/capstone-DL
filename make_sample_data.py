"""
Generates a small synthetic dataset matching the real schema so the training
pipeline can be smoke-tested end-to-end. Replace with the real parquet export
from build_ml_sessions.sql for actual training.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

CATEGORICAL_ONEHOT_GROUPS = {
    "currency": ["currency_1", "currency_2", "currency_3"],
    "device": ["device_1", "device_2", "device_3"],
    "game_id": [f"game_id_{i}" for i in range(1, 12)],
    "game_provider_id": ["game_provider_id_2", "game_provider_id_5", "game_provider_id_10"],
}

NUMERIC_FEATURES = [
    "betbaseamount", "round_duration_sec", "win_over_bet_ratio",
    "spin_number_in_session", "prev_bet", "prev_win", "inter_spin_gap_sec",
    "rtp_bucket_id", "prev_rtp_bucket_id", "bet_change_direction_id",
    "days_since_last_session", "hour_of_day", "period_of_day", "day_of_week",
    "is_weekend", "player_round_count", "player_avg_bet",
    "player_avg_win_over_bet", "player_pct_no_win", "player_avg_round_duration",
    "player_distinct_sessions",
]

TARGETS = ["next_spin_amount", "spins_left_in_session"]

def make_session(account, sessionid, n_spins, split_name):
    rows = []
    bet = rng.uniform(0.1, 10.0)
    for i in range(n_spins):
        spin_num = i + 1
        spins_left = n_spins - spin_num  # 0 on the terminal spin
        win = max(0.0, rng.normal(bet * 0.4, bet * 0.5))
        row = {
            "account": account,
            "sessionid": sessionid,
            "split_name": split_name,
            "betbaseamount": bet,
            "wonbaseamount": win,
            "round_duration_sec": min(rng.exponential(20), 500),
            "win_over_bet_ratio": win / bet if bet > 0 else 0.0,
            "spin_number_in_session": spin_num,
            "prev_bet": bet if i > 0 else np.nan,
            "prev_win": win if i > 0 else np.nan,
            "inter_spin_gap_sec": rng.exponential(4),
            "rtp_bucket_id": rng.integers(0, 12),
            "prev_rtp_bucket_id": rng.integers(0, 12) if i > 0 else np.nan,
            "bet_change_direction_id": rng.integers(0, 3),
            "days_since_last_session": rng.exponential(1.5) if i == 0 else np.nan,
            "hour_of_day": rng.integers(0, 24),
            "period_of_day": rng.integers(0, 4),
            "day_of_week": rng.integers(0, 7),
            "is_weekend": rng.integers(0, 2),
            "player_round_count": i,
            "player_avg_bet": bet,
            "player_avg_win_over_bet": rng.uniform(0, 1),
            "player_pct_no_win": rng.uniform(0, 1),
            "player_avg_round_duration": rng.uniform(0, 5),
            "player_distinct_sessions": rng.integers(0, 10),
            # targets
            "next_spin_amount": bet if spin_num < n_spins else np.nan,  # NULL on last spin
            "spins_left_in_session": spins_left,
        }
        for group, cols in CATEGORICAL_ONEHOT_GROUPS.items():
            chosen = rng.integers(0, len(cols))
            for j, c in enumerate(cols):
                row[c] = 1 if j == chosen else 0
        rows.append(row)
    return rows

def main(n_sessions=4000, out_dir="."):
    all_rows = []
    splits = rng.choice(["train", "valid", "test"], size=n_sessions, p=[0.72, 0.15, 0.13])
    for s in range(n_sessions):
        n_spins = rng.integers(1, 40)
        all_rows.extend(make_session(account=rng.integers(1, 800), sessionid=100000 + s,
                                      n_spins=n_spins, split_name=splits[s]))
    df = pd.DataFrame(all_rows)

    # Write as three separate files -- matches the real on-disk layout
    # (train.parquet / valid.parquet / test.parquet), not one file with a
    # split_name column to filter on.
    filenames = {"train": "train.parquet", "valid": "valid.parquet", "test": "test.parquet"}
    for split_name, fname in filenames.items():
        subset = df[df["split_name"] == split_name].reset_index(drop=True)
        path = f"{out_dir}/{fname}"
        subset.to_parquet(path, index=False)
        print(f"Wrote {len(subset)} spin rows to {path}")

if __name__ == "__main__":
    main()

