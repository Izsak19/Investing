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
FEE_RATE = 0.001  # 0.1% per transaction on Binance spot
INITIAL_CASH = 1000.0
# Minimum idle cash before we automatically refill the portfolio so the learner
# can keep exploring after burning down its balance during early training.
MIN_TRAINING_CASH = 50.0

# Hyperparameters for the simple bandit learner
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount factor for TD updates
USE_TD = True
EPSILON = 0.1  # exploration probability
EPSILON_START = 0.10
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 200_000

# Reward scaling and exploration tweaks
REWARD_SCALE = 50.0  # scales pct-return before tanh; tune 20–200
EPSILON_WHEN_FLAT = 0.05  # exploration floor when not in position
FLAT_EXPLORATION_WARMUP_STEPS = 20_000

# Numerical stability safeguards
# Caps for feature values and weights to prevent floating-point overflow when
# running long training sessions. Normalizing indicators by price (see
# Trainer.step) already keeps most inputs near 1, so these limits simply act as
# a backstop against outliers or unexpected data.
FEATURE_CLIP = 10.0
WEIGHT_CLIP = 10.0
ERROR_CLIP = 5.0

# Trading friction to discourage churn and account for slippage beyond exchange
# fees. Applied per executed trade as a percentage of the notional size.
TURNOVER_PENALTY = 0.001

# Anti-churn safeguards
MIN_HOLD_STEPS = 3
MIN_TRADE_GAP_STEPS = 2
POSITION_FRACTION = 0.5

# Dashboard refresh rate in seconds
DASHBOARD_REFRESH = 1.0
