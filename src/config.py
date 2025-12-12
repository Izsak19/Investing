"""Central configuration values for the trading experiment."""

DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_TIMEFRAME = "1m"
DEFAULT_LIMIT = 200
STATE_PATH = "data/state.json"
TRADE_LOG_PATH = "data/trades.csv"
FEE_RATE = 0.001  # 0.1% per transaction on Binance spot
INITIAL_CASH = 1000.0
# Minimum idle cash before we automatically refill the portfolio so the learner
# can keep exploring after burning down its balance during early training.
MIN_TRAINING_CASH = 50.0

# Hyperparameters for the simple bandit learner
ALPHA = 0.1  # learning rate
GAMMA = 0.0  # no discounting for one-step reward
EPSILON = 0.1  # exploration probability

# Numerical stability safeguards
# Caps for feature values and weights to prevent floating-point overflow when
# running long training sessions. Normalizing indicators by price (see
# Trainer.step) already keeps most inputs near 1, so these limits simply act as
# a backstop against outliers or unexpected data.
FEATURE_CLIP = 10.0
WEIGHT_CLIP = 10.0
ERROR_CLIP = 5.0

# Dashboard refresh rate in seconds
DASHBOARD_REFRESH = 1.0
