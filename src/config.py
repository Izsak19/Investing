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
SLIPPAGE_RATE = 0.0005      # execution slippage as fraction of notional

# Cost-aware gating (1m data is extremely fee-sensitive)
COST_AWARE_GATING = True
EDGE_SAFETY_MARGIN = 0.0002  # extra edge required above estimated costs (tanh-space)

INITIAL_CASH = 1000.0
MIN_TRAINING_CASH = 50.0    # auto-refill threshold to avoid stalled learning

# --- Microstructure friction / gating ---------------------------------

# Extra safety margin added to the all-in cost estimate (fraction of notional).
GATE_SAFETY_MARGIN = 0.0002

# Multiplier to convert estimated cost fraction into an edge threshold.
# Increase if you still see churn; decrease if you see "stuck in hold".
COST_EDGE_MULT = 3.0

# Throttles (1m is extremely fee-sensitive).
# These values are deliberately conservative to reduce churn.
MIN_TRADE_GAP_STEPS = 60
MIN_HOLD_STEPS = 180

# Start gating early (otherwise you churn during warmup)
WARMUP_TRADES_BEFORE_GATING = 20

# ---- Bandit / RL knobs -------------------------------------------------------

# Temporal-difference learning
USE_TD = True               # enable TD(0) target inside the linear bandit update
TD_GAMMA = 0.90             # discount for TD target

# Ridge prior strength: higher -> more conservative updates early on
RIDGE_FACTOR = 4.0
FORGETTING_FACTOR = 0.99    # exponential decay for RLS updates; 1.0 disables forgetting

# Thompson sampling scale (exploration). We decay this with a half-life schedule.
# Reduced exploration pressure for late-stage stability
POSTERIOR_SCALE = 0.30
POSTERIOR_SCALE_MIN = 0.03
POSTERIOR_DECAY_HALF_LIFE_STEPS = 8_000  # ~half the exploration after this many steps; 0 disables decay

# Reward scaling / risk budgets
REWARD_SCALE = 110.0
REWARD_SCALE_MIN = 80.0
REWARD_SCALE_MAX = 150.0
ADAPTIVE_REWARD_SCALE = True
ADAPTIVE_REWARD_DECAY = 0.15       # smoothing factor for updates
ADAPTIVE_TARGET_RETURN_VOL = 0.0025  # target daily-ish volatility for tanh scaling
ADAPTIVE_SHARPE_SOFT = 0.25
ADAPTIVE_SHARPE_STRONG = 1.0
REWARD_RISK_BLEND = 0.45           # mix between raw return and volatility-normalised score
REWARD_STABILITY_WINDOW = 60       # how many recent steps to use for local volatility estimate
REWARD_STABILITY_MIN_OBS = 12

DRAWDOWN_BUDGET = 0.12
DRAWDOWN_BUDGET_MIN = 0.08
DRAWDOWN_BUDGET_MAX = 0.20
ADAPTIVE_RISK_RANGE = 0.45         # how much to loosen/tighten budgets based on Sharpe

# Risk penalties (were 0.0 previously). These provide teeth against deep drawdowns and churn.
DRAWDOWN_PENALTY = 0.6
# Tighter turnover budget to penalize high churn.
TURNOVER_BUDGET_MULTIPLIER = 6.0
TURNOVER_BUDGET_MIN = 6.0
TURNOVER_BUDGET_MAX = 20.0
TURNOVER_BUDGET_WINDOW = 500
TURNOVER_BUDGET_PENALTY = 0.2
RETURN_HISTORY_WINDOW = 200

# ---- Reward shaping / normalization -----------------------------------------
# These keep the learning signal informative (bounded), reduce variance via an
# advantage-style baseline, and optionally add early exploratory noise that has
# zero mean (so it doesn't bias the long-run optimum).
USE_ADVANTAGE_BASELINE = True
BASELINE_EMA_DECAY = 0.02          # EMA update rate for baseline (pct of initial_cash)
USE_REWARD_STD_NORMALIZATION = True
REWARD_SIGMA_FLOOR = 1e-4          # floor for std in pct units to avoid blow-ups
REWARD_VAR_INIT = ADAPTIVE_TARGET_RETURN_VOL ** 2
REWARD_TANH_INPUT_CLIP = 3.5       # clip tanh input to avoid saturation

# Probabilistic shaping noise: sample from Beta(alpha,beta), subtract its mean
# to make it zero-mean, then add it to the tanh input. This increases early
# exploration but decays away over time.
ENABLE_PROB_SHAPING = True
PROB_SHAPING_ALPHA0 = 1.0
PROB_SHAPING_BETA0 = 1.0
PROB_SHAPING_COUNT_DECAY = 0.001   # forgetting for alpha/beta counts
PROB_SHAPING_TANH_AMPLITUDE = 0.35 # max additive noise in tanh-input units
PROB_SHAPING_HALF_LIFE_STEPS = 20_000
PROB_SHAPING_MAX_COUNT = 50_000


# Numeric safety
FEATURE_CLIP = 10.0
WEIGHT_CLIP = 10.0
ERROR_CLIP = 5.0

# ---- Execution policy knobs --------------------------------------------------

# Adaptive rescue when the agent gets stuck in HOLD
ENABLE_STUCK_UNFREEZE = True
STUCK_HOLD_WINDOW = 800           # lookback actions used to decide if we're stuck
STUCK_HOLD_RATIO = 0.9            # trigger when holds dominate this share of recent actions
STUCK_POSTERIOR_BOOST = 0.35      # additive boost to exploration scale when stuck
STUCK_EDGE_THRESHOLD = 0.00005    # relaxed edge gate used while stuck

# Position sizing (dynamic)
POSITION_FRACTION_MIN = 0.03     # min fraction when taking a trade
POSITION_FRACTION_MAX = 0.20     # max fraction when model is very confident (lower => less churn)
CONFIDENCE_K = 3.5               # slope for sigmoid(confidence); higher = more decisive
PARTIAL_SELLS = True             # allow scaling out; improves exit behaviour on 1m

# Prevent repeated BUY legs while already long (fee/slippage amplifier on 1m).
# 1 = only the initial entry; increase to allow limited scaling-in.
MAX_BUY_LEGS_PER_POSITION = 1

# Legacy (kept for compatibility; no longer used when dynamic sizing is on)
POSITION_FRACTION = 0.5

# Only trade if predicted advantage beats costs by this margin (model units)
EDGE_THRESHOLD = 0.0005          # base threshold in scaled reward units (tanh space)
EDGE_THRESHOLD_MIN = 0.0001      # floor for adaptive edge gating
EDGE_THRESHOLD_MAX = 0.0020      # ceiling for adaptive edge gating

# Regime-adaptive edge gating (tighten in chop / relax in trend)
ENABLE_REGIME_EDGE_GATING = True
REGIME_EDGE_VOL_COL = "rv1m_pct_5m"   # expected in [0,1] (percentile)
REGIME_EDGE_LOW_PCT = 0.30
REGIME_EDGE_HIGH_PCT = 0.70
REGIME_EDGE_TIGHTEN_MULT = 1.5   # multiplier when vol percentile is low (chop)
REGIME_EDGE_RELAX_MULT = 0.7     # multiplier when vol percentile is high (trend)

# Action hysteresis (reduce flip-flopping between BUY/SELL)
ENABLE_ACTION_HYSTERESIS = True
HYSTERESIS_REQUIRED_STREAK = 3    # require this many consecutive proposals before a flip
HYSTERESIS_ALLOW_IF_EDGE_MULT = 2.0  # allow immediate flip if edge >= this * current gate threshold

# Multi-timeframe confirmation (lightweight 5m regime filter)
ENABLE_MTF_CONFIRMATION = True
MTF_VOL_PCT_MIN = 0.55          # require volatility percentile >= this for entries/exits
MTF_SPREAD_PCT_MAX = 0.85       # block trades when spread percentile is too high

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

    "cold_start_trades": {
        "POSTERIOR_SCALE": 0.6,                 # bigger Thompson noise early
        "POSTERIOR_DECAY_HALF_LIFE_STEPS": 5_000,
        "POSTERIOR_SCALE_MIN": 0.1,
        "EDGE_THRESHOLD": 0.0005,               # keep in tanh space
        "WARMUP_TRADES_BEFORE_GATING": 0,       # don't suppress edges at start
        "MIN_HOLD_STEPS": 0,
        "MIN_TRADE_GAP_STEPS": 0,
        # Optional for debugging: avoid silent resets while inspecting P&L
        # "MIN_TRAINING_CASH": 0.0,
    }
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
