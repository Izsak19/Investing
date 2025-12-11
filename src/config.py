"""Central configuration values for the trading experiment."""

DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_TIMEFRAME = "5m"
DEFAULT_LIMIT = 200
STATE_PATH = "data/state.json"
TRADE_LOG_PATH = "data/trades.csv"

# Hyperparameters for the simple bandit learner
ALPHA = 0.2  # learning rate
GAMMA = 0.0  # no discounting for one-step reward
EPSILON = 0.1  # exploration probability

# Dashboard refresh rate in seconds
DASHBOARD_REFRESH = 1.0
