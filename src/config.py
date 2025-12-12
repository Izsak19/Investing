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
ALPHA = 0.2  # learning rate
GAMMA = 0.0  # no discounting for one-step reward
EPSILON = 0.1  # exploration probability

# Dashboard refresh rate in seconds
DASHBOARD_REFRESH = 1.0
