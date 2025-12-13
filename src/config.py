# src/config.py
"""Central configuration values for the trading experiment."""

DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_TIMEFRAME = "1m"
DEFAULT_LIMIT = 200
STATE_PATH = "data/state.json"
TRADE_LOG_PATH = "data/trades.csv"
RUNS_DIR = "data/runs"
DEFAULT_CHECKPOINT_EVERY = 2000
DEFAULT_FLUSH_TRADES_EVERY = 500
DEFAULT_KEEP_LAST_CHECKPOINTS = 5
DEFAULT_RANDOM_SEED = 1337

# Trading costs
FEE_RATE = 0.001            # exchange fee per transaction
TURNOVER_PENALTY = 0.001    # extra friction per executed trade as fraction of notional

INITIAL_CASH = 1000.0
MIN_TRAINING_CASH = 50.0    # auto-refill threshold to avoid stalled learning

# ---- Bandit / RL knobs -------------------------------------------------------

# Temporal-difference learning
USE_TD = True               # enable TD(0) target inside the linear bandit update
TD_GAMMA = 0.90             # discount for TD target

# Ridge prior strength: higher -> more conservative updates early on
RIDGE_FACTOR = 1.0

# Thompson sampling scale (exploration). We decay this with a half-life schedule.
POSTERIOR_SCALE = 0.35
POSTERIOR_SCALE_MIN = 0.0
POSTERIOR_DECAY_HALF_LIFE_STEPS = 25_000  # ~half the exploration after this many steps; 0 disables decay

# Reward scaling / risk budgets
REWARD_SCALE = 50.0
DRAWDOWN_BUDGET = 0.12
DRAWDOWN_PENALTY = 0.75
TURNOVER_BUDGET_MULTIPLIER = 1.2
TURNOVER_BUDGET_WINDOW = 500
TURNOVER_BUDGET_PENALTY = 0.25
RETURN_HISTORY_WINDOW = 200

# Numeric safety
FEATURE_CLIP = 10.0
WEIGHT_CLIP = 10.0
ERROR_CLIP = 5.0

# ---- Execution policy knobs --------------------------------------------------

# Trade timing safeguards
MIN_HOLD_STEPS = 3
MIN_TRADE_GAP_STEPS = 2

# Position sizing (dynamic)
POSITION_FRACTION_MIN = 0.10     # min fraction when taking a trade
POSITION_FRACTION_MAX = 0.75     # max fraction when model is very confident
CONFIDENCE_K = 3.0               # slope for sigmoid(confidence); higher = more decisive
PARTIAL_SELLS = True             # enable partial liquidation based on confidence

# Legacy (kept for compatibility; no longer used when dynamic sizing is on)
POSITION_FRACTION = 0.5

# Only trade if predicted advantage beats costs by this margin (model units)
EDGE_THRESHOLD = 0.005           # in scaled reward units (tanh space); 0 to disable gating

# Reporting
ACTION_HISTORY_WINDOW = 5_000
WALKFORWARD_FOLDS = 3
DASHBOARD_REFRESH = 1.0
