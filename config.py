import os

DATA_DIR = "/Top/ACG/Year_2/trimester 3/capstone/data/ml/generic/fixed/"
TRAIN_FILE = "train_features_fix.parquet"
VALID_FILE = "valid_features_fix.parquet"
TEST_FILE = "test_features_fix.parquet"
OUTPUT_DIR = "/Top/ACG/Year_2/trimester 3/capstone/code/DL/outputs/"

SPLIT_COL = "split_name"
ACCOUNT_COL = "account"
SESSION_COL = "sessionid"
TIME_COL = "starttime"

# ---------------------------------------------------------------------
# Sequencing
# ---------------------------------------------------------------------
# Unit of sequencing: one session per sequence (spins are already
# contiguous within a session and spin_number_in_session gives a clean
# order). Sessions longer than MAX_SEQ_LEN are truncated to the most
# recent MAX_SEQ_LEN spins (keeps the tensor size bounded; long
# sessions are rare tail cases per the earlier duration-based filtering
# in the SQL pipeline).
MAX_SEQ_LEN = 64
MIN_SEQ_LEN = 2  # sessions with a single spin have no next_spin_amount target to learn from meaningfully

# ---------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------
# Per-step numeric features fed to the model at every timestep. Lag
# features (prev_bet, prev_win, prev_rtp_bucket_id, bet_change_direction_id)
# were removed upstream in the SQL: the LSTM sees the raw per-step
# history directly and can learn any such relationship itself.
NUMERIC_FEATURES = [
    "betbaseamount", "round_duration_sec", "win_over_bet_ratio",
    "spin_number_in_session", "inter_spin_gap_sec",
    "rtp_bucket_id",
    "days_since_last_session", "period_of_day",
    "is_weekend", "player_round_count", "player_avg_bet",
    "player_avg_win_over_bet", "player_pct_no_win", "player_avg_round_duration",
    "player_distinct_sessions",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
]

ONEHOT_FEATURES = (
    [f"currency_{i}" for i in (1, 2, 3)]
    + [f"device_{i}" for i in (1, 2, 3)]
    + [f"game_id_{i}" for i in range(1, 12)]
    + [f"game_provider_id_{i}" for i in (2, 5, 10)]
)

TARGET_SPINS_LEFT = "spins_left_in_session"
TARGET_NEXT_BET = "next_spin_amount"

# Missing-indicator treatment. days_since_last_session is NULL on an
# account's first-ever session (see is_first_session in the SQL); it's
# genuinely missing (no prior session exists), so it gets an explicit
# was_missing flag + median fill like the other columns.
ADD_MISSING_INDICATORS = [
    "days_since_last_session",
]

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
RNN_HIDDEN_SIZE = 128
RNN_NUM_LAYERS = 2
RNN_DROPOUT = 0.2
HEAD_HIDDEN = 32

# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
BATCH_SIZE = 256  # sequences per batch, not rows
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
