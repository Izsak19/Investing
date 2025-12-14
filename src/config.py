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
FORGETTING_FACTOR = 0.99    # exponential decay for RLS updates; 1.0 disables forgetting

# Thompson sampling scale (exploration). We decay this with a half-life schedule.
POSTERIOR_SCALE = 0.65
POSTERIOR_SCALE_MIN = 0.05
POSTERIOR_DECAY_HALF_LIFE_STEPS = 7_500  # ~half the exploration after this many steps; 0 disables decay

# Reward scaling / risk budgets
REWARD_SCALE = 125.0
REWARD_SCALE_MIN = 80.0
REWARD_SCALE_MAX = 200.0
ADAPTIVE_REWARD_SCALE = True
ADAPTIVE_REWARD_DECAY = 0.15       # smoothing factor for updates
ADAPTIVE_TARGET_RETURN_VOL = 0.0025  # target daily-ish volatility for tanh scaling
ADAPTIVE_SHARPE_SOFT = 0.25
ADAPTIVE_SHARPE_STRONG = 1.0

DRAWDOWN_BUDGET = 0.12
DRAWDOWN_BUDGET_MIN = 0.08
DRAWDOWN_BUDGET_MAX = 0.20
ADAPTIVE_RISK_RANGE = 0.45         # how much to loosen/tighten budgets based on Sharpe

DRAWDOWN_PENALTY = 0.0
TURNOVER_BUDGET_MULTIPLIER = 12.0
TURNOVER_BUDGET_MIN = 6.0
TURNOVER_BUDGET_MAX = 20.0
TURNOVER_BUDGET_WINDOW = 500
TURNOVER_BUDGET_PENALTY = 0.0
RETURN_HISTORY_WINDOW = 200

# Numeric safety
FEATURE_CLIP = 10.0
WEIGHT_CLIP = 10.0
ERROR_CLIP = 5.0

# ---- Execution policy knobs --------------------------------------------------

# Trade timing safeguards
MIN_HOLD_STEPS = 1
MIN_TRADE_GAP_STEPS = 1

# Adaptive rescue when the agent gets stuck in HOLD
ENABLE_STUCK_UNFREEZE = True
STUCK_HOLD_WINDOW = 800           # lookback actions used to decide if we're stuck
STUCK_HOLD_RATIO = 0.9            # trigger when holds dominate this share of recent actions
STUCK_POSTERIOR_BOOST = 0.35      # additive boost to exploration scale when stuck
STUCK_EDGE_THRESHOLD = 0.00005    # relaxed edge gate used while stuck

# Position sizing (dynamic)
POSITION_FRACTION_MIN = 0.05     # min fraction when taking a trade
POSITION_FRACTION_MAX = 0.75     # max fraction when model is very confident
CONFIDENCE_K = 3.5               # slope for sigmoid(confidence); higher = more decisive
PARTIAL_SELLS = True             # enable partial liquidation based on confidence

# Legacy (kept for compatibility; no longer used when dynamic sizing is on)
POSITION_FRACTION = 0.5

# Only trade if predicted advantage beats costs by this margin (model units)
EDGE_THRESHOLD = 0.0005          # in scaled reward units (tanh space); 0 to disable gating
WARMUP_TRADES_BEFORE_GATING = 10

# Reporting
ACTION_HISTORY_WINDOW = 5_000
WALKFORWARD_FOLDS = 3
DASHBOARD_REFRESH = 1.0

# Penalty profiles
PENALTY_PROFILES = {
    "train": {
        "drawdown_penalty": DRAWDOWN_PENALTY * 0.8,
        "turnover_budget_penalty": TURNOVER_BUDGET_PENALTY * 0.75,
    },
    "eval": {
        "drawdown_penalty": DRAWDOWN_PENALTY,
        "turnover_budget_penalty": TURNOVER_BUDGET_PENALTY,
    },
}

# Presets to quickly adjust exploration/penalties/trade gates
PROFILES = {
    "debug_more_trades": {
        "POSTERIOR_SCALE": 0.85,
        "POSTERIOR_DECAY_HALF_LIFE_STEPS": 5_000,
        "POSTERIOR_SCALE_MIN": 0.08,
        "EDGE_THRESHOLD": 0.0001,
        "MIN_HOLD_STEPS": 0,
        "MIN_TRADE_GAP_STEPS": 0,
        "REWARD_SCALE": 140.0,
        "DRAWDOWN_PENALTY": 0.5,
        "TURNOVER_BUDGET_PENALTY": 0.12,
        "POSITION_FRACTION_MIN": 0.04,
    },
    "conservative": {
        "POSTERIOR_SCALE": 0.5,
        "POSTERIOR_DECAY_HALF_LIFE_STEPS": 10_000,
        "POSTERIOR_SCALE_MIN": 0.04,
        "EDGE_THRESHOLD": 0.001,
        "MIN_HOLD_STEPS": 2,
        "MIN_TRADE_GAP_STEPS": 2,
        "REWARD_SCALE": 110.0,
        "DRAWDOWN_PENALTY": 0.7,
        "TURNOVER_BUDGET_PENALTY": 0.22,
        "POSITION_FRACTION_MIN": 0.06,
    },
}


def apply_profile(name: str | None) -> None:
    if not name:
        return
    profile = PROFILES.get(name)
    if not profile:
        return
    for key, value in profile.items():
        globals()[key] = value
    PENALTY_PROFILES["train"] = {
        "drawdown_penalty": globals().get("DRAWDOWN_PENALTY", DRAWDOWN_PENALTY) * 0.8,
        "turnover_budget_penalty": globals().get("TURNOVER_BUDGET_PENALTY", TURNOVER_BUDGET_PENALTY) * 0.75,
    }
    PENALTY_PROFILES["eval"] = {
        "drawdown_penalty": globals().get("DRAWDOWN_PENALTY", DRAWDOWN_PENALTY),
        "turnover_budget_penalty": globals().get("TURNOVER_BUDGET_PENALTY", TURNOVER_BUDGET_PENALTY),
    }
