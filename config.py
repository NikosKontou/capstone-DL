import os

DATA_DIR = "/Top/ACG/Year_2/trimester 3/capstone/data/ml/generic/fixed/"
TRAIN_FILE = "train_features_fix.parquet"
VALID_FILE = "valid_features_fix.parquet"
TEST_FILE = "test_features_fix.parquet"
OUTPUT_DIR = "/Top/ACG/Year_2/trimester 3/capstone/code/DL/outputs/"

SPLIT_COL = "split_name"
ACCOUNT_COL = "account"
SESSION_COL = "sessionid"

NUMERIC_FEATURES = [
    "betbaseamount", "round_duration_sec", "win_over_bet_ratio",
    "spin_number_in_session", "prev_bet", "prev_win", "inter_spin_gap_sec",
    "rtp_bucket_id", "prev_rtp_bucket_id", "bet_change_direction_id",
    "days_since_last_session", "period_of_day",
    "is_weekend", "player_round_count", "player_avg_bet",
    "player_avg_win_over_bet", "player_pct_no_win", "player_avg_round_duration",
    "player_distinct_sessions",
    "hour_sin", "hour_cos", "day_sin", "day_cos"
]

ONEHOT_FEATURES = (
    [f"currency_{i}" for i in (1, 2, 3)]
    + [f"device_{i}" for i in (1, 2, 3)]
    + [f"game_id_{i}" for i in range(1, 12)]
    + [f"game_provider_id_{i}" for i in (2, 5, 10)]
)

TARGET_SPINS_LEFT = "spins_left_in_session"
TARGET_NEXT_BET = "next_spin_amount"

ADD_MISSING_INDICATORS = [
    "prev_bet", "prev_win", "prev_rtp_bucket_id", "days_since_last_session",
]

HIDDEN_DIMS = [256, 128, 64]
DROPOUT = 0.2
USE_BATCHNORM = True

BATCH_SIZE = 1024
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 8
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.5
GRAD_CLIP_NORM = 5.0

LOSS_WEIGHT_SPINS_LEFT = 1.0
LOSS_WEIGHT_NEXT_BET = 1.0

RANDOM_SEED = 42
DEVICE = "cuda"
