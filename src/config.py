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
FEE_RATE = 0.001  # 0.1% per transaction on Binance spot
INITIAL_CASH = 1000.0
# Minimum idle cash before we automatically refill the portfolio so the learner
# can keep exploring after burning down its balance during early training.
MIN_TRAINING_CASH = 50.0

# Hyperparameters for the contextual bandit learner
ALPHA = 0.1  # retained for backwards compatibility with saved state
GAMMA = 0.9  # retained for backwards compatibility with saved state
USE_TD = True
# Strength of the L2 prior (lambda * I) on the design matrix; higher values make
# the posterior more conservative and slow down updates early in training.
RIDGE_FACTOR = 1.0
# Scale applied to the posterior covariance when drawing Thompson samples; lower
# values reduce exploration, higher values increase it.
POSTERIOR_SCALE = 0.35
POSTERIOR_SCALE_MIN = 0.0

# Reward scaling and risk budgets
REWARD_SCALE = 50.0  # scales pct-return before tanh; tune 20–200
DRAWDOWN_BUDGET = 0.12  # fraction of equity allowed before penalties apply
DRAWDOWN_PENALTY = 0.75  # capital-scaled penalty for exceeding drawdown budget
TURNOVER_BUDGET_MULTIPLIER = 1.2  # notional traded allowed (as a multiple of initial cash) in the window
TURNOVER_BUDGET_WINDOW = 500  # steps to accumulate turnover before applying budget penalty
TURNOVER_BUDGET_PENALTY = 0.25  # capital-scaled penalty for exceeding turnover budget
RETURN_HISTORY_WINDOW = 200  # rolling window for return- and volatility-aware metrics

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
ACTION_HISTORY_WINDOW = 5_000

# Anti-churn safeguards
MIN_HOLD_STEPS = 3
MIN_TRADE_GAP_STEPS = 2
POSITION_FRACTION = 0.5
WALKFORWARD_FOLDS = 3

# Dashboard refresh rate in seconds
DASHBOARD_REFRESH = 1.0
