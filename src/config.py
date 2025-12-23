# src/config.py
"""Central configuration values for the trading experiment.

This project is a *simulation / research harness*.
The defaults below are tuned to be more "professional" for 1m BTC data:
- lower exploration
- stronger anti-churn controls
- much smaller learning penalties for blocked proposals
- disable probabilistic reward shaping by default

The goal is to avoid fee-death-by-churn and distorted learning signals.
"""

# ----------------------------------------------------------------------------
# Basics
# ----------------------------------------------------------------------------

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

INITIAL_CASH = 1000.0
MIN_TRAINING_CASH = 50.0    # auto-refill threshold to avoid stalled learning

# ----------------------------------------------------------------------------
# Trading costs (execution)
# ----------------------------------------------------------------------------

FEE_RATE = 0.001            # exchange fee per transaction (fraction of notional)
SLIPPAGE_RATE = 0.0005      # execution slippage (fraction of notional)

# Prevent micro-trades that are dominated by fixed frictions on 1m.
# If an order would trade less than this notional, we skip it (unless closing dust).
MIN_TRADE_NOTIONAL = 25.0

# TURNOVER_PENALTY is a *learning regularizer* (NOT an execution cash cost).
# Keep small; it should not dominate reward.
TURNOVER_PENALTY = 0.00005

# ----------------------------------------------------------------------------
# Trade rate throttling (anti-churn, separate from cooldown)
# ----------------------------------------------------------------------------
#
# Cooldowns prevent back-to-back flips, but a noisy policy can still slowly churn
# (e.g., 1 trade every ~10 minutes). On 1m data, that is often still fee-death.
# This throttle adds a soft-but-firm ceiling on trade frequency, while still
# allowing risk-reducing exits.
TRADE_RATE_WINDOW_STEPS = 1000          # ~16.7h on 1m (large enough to smooth)
# Tighten this to actually bind: recent runs are ~165 trades / 1000 steps.
# Setting <165 forces a real reduction in churn on 1m while still allowing exits.
MAX_TRADES_PER_WINDOW = 90              # target <= ~90 trades / 1000 steps
# When trade rate is exceeded:
# - block BUYs entirely
# - allow SELLs only when the sell edge is strong (margin >= multiplier * cost_edge)
TRADE_RATE_SELL_BYPASS_EDGE_MULT = 1.5

# ----------------------------------------------------------------------------
# Cost-aware gating
# ----------------------------------------------------------------------------

COST_AWARE_GATING = True
# Extra safety margin added to the all-in cost estimate (fraction of notional).
GATE_SAFETY_MARGIN = 0.0001
# Extra edge required above estimated costs (tanh-space units; keep small).
EDGE_SAFETY_MARGIN = 0.00015
# Multiplier applied when translating estimated cost fraction into edge threshold.
COST_EDGE_MULT = 1.3

# Start gating early (otherwise you churn during warmup)
WARMUP_TRADES_BEFORE_GATING = 5

# ----------------------------------------------------------------------------
# Microstructure throttles / cooldown
# ----------------------------------------------------------------------------

# 1m is extremely fee-sensitive. These are intentionally conservative.
MIN_TRADE_GAP_STEPS = 10     # reduce churn: >=10 minutes between entries/exits on 1m
MIN_HOLD_STEPS = 30          # allow exits/adjustments sooner

# Adaptive/conditional cooldown (regime + turnover aware).
ENABLE_ADAPTIVE_COOLDOWN = True
COOLDOWN_VOL_COL = globals().get('REGIME_EDGE_VOL_COL', 'regime_edge_vol_pct')  # expected in [0,1]
COOLDOWN_VOL_LOW_PCT = 0.20
COOLDOWN_VOL_HIGH_PCT = 0.80
# Do not tighten beyond base cooldowns by regime; only relax slightly in high vol.
COOLDOWN_GAP_TIGHTEN_MULT = 1.0
COOLDOWN_GAP_RELAX_MULT = 0.65
COOLDOWN_HOLD_TIGHTEN_MULT = 1.0
COOLDOWN_HOLD_RELAX_MULT = 0.75
# Floors ensure there is never a full bypass loophole.
COOLDOWN_MIN_GAP_FLOOR = 5
COOLDOWN_MIN_HOLD_FLOOR = 10
# Turnover-aware scaling (lengthen only when stressed).
COOLDOWN_TURNOVER_SENSITIVITY = 0.35
# Proxy vol when percentile column missing.
ADAPTIVE_COOLDOWN_RET_EMA_DECAY = 0.05

# Exceptional trades during cooldown must be rare.
COOLDOWN_STRONG_EDGE_MULT = 2.5
COOLDOWN_BYPASS_SELL_ONLY = False
COOLDOWN_STRONG_MIN_GAP_STEPS = 5
# Allow risk-reducing exits during cooldown when the position is modestly underwater.
# This prevents large drawdowns caused by gated sells after a bad entry.
COOLDOWN_SELL_UNDERWATER_EXIT_PCT = 0.002

# Turnover hard block: if stressed, block new entries and allow exits only on strong edge.
TURNOVER_HARD_BLOCK_MULT = 2.0

# Close tiny residual positions.
DUST_POSITION_NOTIONAL = 10.0

# ----------------------------------------------------------------------------
# Penalize infeasible proposals (learning only)
# ----------------------------------------------------------------------------

# These must be *tiny*; otherwise they dominate the learning signal when blocks are frequent.
# Values are fractions of INITIAL_CASH.
BLOCKED_GATE_PENALTY = 0.00001
BLOCKED_TIMING_PENALTY = 0.000005
BLOCKED_BUDGET_PENALTY = 0.00002

# Data hygiene for percentile-like regime columns.
SANITIZE_PERCENTILE_COLUMNS = True

# ----------------------------------------------------------------------------
# Bandit / RL knobs
# ----------------------------------------------------------------------------

USE_TD = True
TD_GAMMA = 0.90

RIDGE_FACTOR = 6.0
FORGETTING_FACTOR = 0.995

# Thompson sampling (exploration) – lowered for stability on 1m.
POSTERIOR_SCALE = 0.18
POSTERIOR_SCALE_MIN = 0.05
POSTERIOR_DECAY_HALF_LIFE_STEPS = 6_000

# ----------------------------------------------------------------------------
# Reward scaling / risk budgets
# ----------------------------------------------------------------------------

# Lower defaults: keep tanh input in a useful range without saturating.
REWARD_SCALE = 70.0
REWARD_SCALE_MIN = 45.0
REWARD_SCALE_MAX = 110.0
ADAPTIVE_REWARD_SCALE = True
ADAPTIVE_REWARD_DECAY = 0.15
ADAPTIVE_TARGET_RETURN_VOL = 0.0020
ADAPTIVE_SHARPE_SOFT = 0.25
ADAPTIVE_SHARPE_STRONG = 1.0

REWARD_RISK_BLEND = 0.35
REWARD_STABILITY_WINDOW = 120
REWARD_STABILITY_MIN_OBS = 24

DRAWDOWN_BUDGET = 0.12
DRAWDOWN_BUDGET_MIN = 0.08
DRAWDOWN_BUDGET_MAX = 0.20
ADAPTIVE_RISK_RANGE = 0.45

DRAWDOWN_PENALTY = 0.6

# ----------------------------------------------------------------------------
# Safety kill-switch (stop a run when live expectancy is clearly negative)
# ----------------------------------------------------------------------------

# If enabled, the main loop will stop early when the strategy is structurally losing.
ENABLE_KILL_SWITCH = True
KILL_SWITCH_MIN_SELL_LEGS = 200
KILL_SWITCH_EXPECTANCY_PNL_PER_SELL_LEG = -0.05  # stop if expectancy stays worse than this
KILL_SWITCH_MAX_DRAWDOWN = 0.12                  # stop earlier when DD is already material
KILL_SWITCH_MAX_TRADES = 2500                    # absolute guardrail (especially for 5m)

# Turnover budget (rolling): keep tighter for 1m.
TURNOVER_BUDGET_MULTIPLIER = 6.0
TURNOVER_BUDGET_MIN = 5.0
TURNOVER_BUDGET_MAX = 20.0
TURNOVER_BUDGET_WINDOW = 500
TURNOVER_BUDGET_PENALTY = 0.18
RETURN_HISTORY_WINDOW = 200

# ----------------------------------------------------------------------------
# Reward shaping / normalization
# ----------------------------------------------------------------------------

USE_ADVANTAGE_BASELINE = True
BASELINE_EMA_DECAY = 0.02
REWARD_VAR_EMA_DECAY = 0.02
USE_REWARD_STD_NORMALIZATION = True
REWARD_SIGMA_FLOOR = 1e-4
REWARD_VAR_INIT = ADAPTIVE_TARGET_RETURN_VOL ** 2
REWARD_TANH_INPUT_CLIP = 3.5

# -----------------------------------------------------------------------------
# Directional shaping (helps entry learning when flat)
# -----------------------------------------------------------------------------
#
# Portfolio PnL provides a clean signal *when you are exposed*, but when flat it
# is often exactly zero. That can slow learning for entry timing, especially with
# conservative gates. We add a small, potential-based directional term that uses
# the next-bar return only when the agent is FLAT.
#
# Design goals:
# - keep it small vs. true PnL (never dominate)
# - do not pay the agent for churn (still cost-gated)
# - provide a gradient for BUY vs HOLD decisions during flat periods.
ENABLE_DIRECTIONAL_SHAPING = True
# Weight is in "fraction of initial cash per 1.0 return"; e.g. 0.12 means a +1%
# next-bar move is worth +0.12% of initial cash (before tanh/baseline/std).
DIRECTIONAL_SHAPING_WEIGHT = 0.12
# Penalty for HOLD while flat during large moves (discourages eternal HOLD).
FLAT_HOLD_MISS_PENALTY_WEIGHT = 0.04
# Clip the per-step shaping contribution (as fraction of initial cash).
DIRECTIONAL_SHAPING_CLIP = 0.004

# Disable probabilistic shaping by default (too destabilizing for 1m).
ENABLE_PROB_SHAPING = False
PROB_SHAPING_ALPHA0 = 1.0
PROB_SHAPING_BETA0 = 1.0
PROB_SHAPING_COUNT_DECAY = 0.001
PROB_SHAPING_TANH_AMPLITUDE = 0.35
PROB_SHAPING_HALF_LIFE_STEPS = 20_000
PROB_SHAPING_MAX_COUNT = 50_000

# ----------------------------------------------------------------------------
# Numeric safety
# ----------------------------------------------------------------------------

FEATURE_CLIP = 10.0
WEIGHT_CLIP = 10.0
MARGIN_SCALE_MULT = 1.0
ERROR_CLIP = 5.0

# ----------------------------------------------------------------------------
# Execution policy knobs
# ----------------------------------------------------------------------------

# Stuck rescue (keep, but conservative)
ENABLE_STUCK_UNFREEZE = True
STUCK_HOLD_WINDOW = 800
STUCK_HOLD_RATIO = 0.92
STUCK_POSTERIOR_BOOST = 0.20
STUCK_EDGE_THRESHOLD = 0.0

# ----------------------------------------------------------------------------
# Hard risk exits (optional; evaluated independently of gating)
# ----------------------------------------------------------------------------
#
# OFF by default to preserve baseline behaviour and keep comparisons clean.
ENABLE_HARD_RISK_EXITS = False

# Per-position thresholds (fractional returns vs entry).
STOP_LOSS_PCT = 0.008          # 0.8% hard stop
TRAILING_STOP_PCT = 0.006      # 0.6% trailing stop from peak
MAX_POSITION_HOLD_STEPS = 240  # 4h on 1m candles

# Position sizing (dynamic) – smaller sizing for 1m.
POSITION_FRACTION_MIN = 0.02
POSITION_FRACTION_MAX = 0.10
CONFIDENCE_K = 3.5
PARTIAL_SELLS = True

# Prevent repeated BUY legs while already long.
MAX_BUY_LEGS_PER_POSITION = 1

# Legacy (kept for compatibility)
POSITION_FRACTION = 0.5

# Base edge threshold (tanh-space). Slightly higher for 1m.
EDGE_THRESHOLD = 0.0015
EDGE_THRESHOLD_MIN = 0.0002
EDGE_THRESHOLD_MAX = 0.0030

# Regime-adaptive edge gating
ENABLE_REGIME_EDGE_GATING = True
REGIME_EDGE_VOL_COL = "rv1m_pct_5m"   # expected in [0,1]
REGIME_EDGE_LOW_PCT = 0.30
REGIME_EDGE_HIGH_PCT = 0.70
REGIME_EDGE_TIGHTEN_MULT = 1.6
REGIME_EDGE_RELAX_MULT = 0.6

# Action hysteresis

# Confidence-based HOLD gating (agent-side).

# This is separate from EDGE_THRESHOLD (used in cost gating).

# If set too high, it can force a permanent HOLD regime as exploration decays.
CONFIDENCE_HOLD_THRESHOLD = 0.0000
ENABLE_ACTION_HYSTERESIS = True
HYSTERESIS_REQUIRED_STREAK = 1
HYSTERESIS_ALLOW_IF_EDGE_MULT = 1.5

# Multi-timeframe confirmation
ENABLE_MTF_CONFIRMATION = False
MTF_VOL_PCT_MIN = 0.30
MTF_LOWVOL_EDGE_MULT = 1.5
MTF_SPREAD_PCT_MAX = 0.85
MTF_SPREAD_PCT_HARD = 0.995
MTF_HIGHSPREAD_EDGE_MULT = 2.0

# Reporting
ACTION_HISTORY_WINDOW = 5_000
WALKFORWARD_FOLDS = 3
DASHBOARD_REFRESH = 1.0

# ----------------------------------------------------------------------------
# Penalty profiles
# ----------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------
# Presets to quickly adjust exploration/penalties/trade gates
# ----------------------------------------------------------------------------

PROFILES = {
    "debug_more_trades": {
        "POSTERIOR_SCALE": 0.60,
        "POSTERIOR_DECAY_HALF_LIFE_STEPS": 5_000,
        "POSTERIOR_SCALE_MIN": 0.08,
        "EDGE_THRESHOLD": 0.0002,
        "MIN_HOLD_STEPS": 0,
        "MIN_TRADE_GAP_STEPS": 0,
        "REWARD_SCALE": 110.0,
        "DRAWDOWN_PENALTY": 0.5,
        "TURNOVER_BUDGET_PENALTY": 0.12,
        "POSITION_FRACTION_MIN": 0.03,
        "POSITION_FRACTION_MAX": 0.15,
        "ENABLE_PROB_SHAPING": False,
    },
    "conservative": {
        "POSTERIOR_SCALE": 0.10,
        "POSTERIOR_DECAY_HALF_LIFE_STEPS": 8_000,
        "POSTERIOR_SCALE_MIN": 0.02,
        "EDGE_THRESHOLD": 0.0015,
        "MIN_HOLD_STEPS": 120,
        "MIN_TRADE_GAP_STEPS": 45,
        "REWARD_SCALE": 60.0,
        "DRAWDOWN_PENALTY": 0.7,
        "TURNOVER_BUDGET_PENALTY": 0.22,
        "POSITION_FRACTION_MIN": 0.015,
        "POSITION_FRACTION_MAX": 0.08,
        "TURNOVER_BUDGET_MULTIPLIER": 5.0,
        "ENABLE_PROB_SHAPING": False,
    },
    "cold_start_trades": {
        "POSTERIOR_SCALE": 0.30,
        "POSTERIOR_DECAY_HALF_LIFE_STEPS": 5_000,
        "POSTERIOR_SCALE_MIN": 0.08,
        "EDGE_THRESHOLD": 0.0006,
        "WARMUP_TRADES_BEFORE_GATING": 0,
        "MIN_HOLD_STEPS": 0,
        "MIN_TRADE_GAP_STEPS": 0,
        "ENABLE_PROB_SHAPING": False,
    },
    # 5m-specific: conservative but not frozen.
    # Steps are 5-minute candles (so MIN_HOLD_STEPS=6 => 30 minutes).
    "tf_5m_conservative": {
        # Key fix vs the prior version: the old (EDGE_THRESHOLD=0.0075, COST_EDGE_MULT=3.8)
        # pre-masked almost every BUY/SELL into HOLD, starving learning.
        "WARMUP_TRADES_BEFORE_GATING": 20,
        # Confidence gating: disable for 5m profiles; cost-aware gating already controls churn.
        "CONFIDENCE_HOLD_THRESHOLD": 0.0,
        # Exploration: enough to collect informative samples early, then decay.
        "POSTERIOR_SCALE": 0.16,
        "POSTERIOR_DECAY_HALF_LIFE_STEPS": 8_000,
        "POSTERIOR_SCALE_MIN": 0.03,
        "FORGETTING_FACTOR": 0.996,
        # Cost-aware edge requirements (moderate).
        "EDGE_THRESHOLD": 0.0022,
        "COST_EDGE_MULT": 1.6,
        "EDGE_SAFETY_MARGIN": 0.00025,
        # Timing / churn controls (5m steps).
        "MIN_HOLD_STEPS": 6,
        "MIN_TRADE_GAP_STEPS": 4,
        "HYSTERESIS_REQUIRED_STREAK": 2,
        "HYSTERESIS_ALLOW_IF_EDGE_MULT": 1.6,
        # Trade-rate ceiling: allow learning, still cap churn.
        # Window=240 steps ~= 20 hours; 30 trades ~= 1.5 trades/hour max.
        "TRADE_RATE_WINDOW_STEPS": 240,
        "MAX_TRADES_PER_WINDOW": 30,
        "TRADE_RATE_SELL_BYPASS_EDGE_MULT": 2.0,
        # Turnover discipline (learning regularizer + budget).
        "TURNOVER_PENALTY": 0.00006,
        "TURNOVER_BUDGET_MULTIPLIER": 2.0,
        "TURNOVER_BUDGET_MIN": 1.0,
        "TURNOVER_BUDGET_PENALTY": 0.26,
        # Safety net against HOLD starvation.
        "ENABLE_STUCK_UNFREEZE": True,
        # Risk exits: tail-risk limiter to prevent long drifts while learning on sparse signals.
        "ENABLE_HARD_RISK_EXITS": True,
        "STOP_LOSS_PCT": 0.008,
        "TRAILING_STOP_PCT": 0.010,
        "MAX_POSITION_HOLD_STEPS": 72,
        # Make trades meaningful on 5m (avoid tiny, fee-dominated probe orders).
        "POSITION_FRACTION_MIN": 0.03,
        "POSITION_FRACTION_MAX": 0.12,
        # Learning stability
        "REWARD_SCALE": 65.0,
        "ADAPTIVE_REWARD_DECAY": 0.08,
        "DRAWDOWN_PENALTY": 0.55,
        "DIRECTIONAL_SHAPING_WEIGHT": 0.05,
        "FLAT_HOLD_MISS_PENALTY_WEIGHT": 0.03,
        "ENABLE_PROB_SHAPING": False,
    },

    # 5m learning-first preset: trade enough to learn signal, then let the gates do their job.
    # Use this when the agent is HOLD-starving or in early development.
    "tf_5m_learn": {
        "WARMUP_TRADES_BEFORE_GATING": 50,
        "CONFIDENCE_HOLD_THRESHOLD": 0.0,
        "POSTERIOR_SCALE": 0.22,
        "POSTERIOR_DECAY_HALF_LIFE_STEPS": 10_000,
        "POSTERIOR_SCALE_MIN": 0.05,
        "FORGETTING_FACTOR": 0.995,
        "EDGE_THRESHOLD": 0.0014,
        "COST_EDGE_MULT": 1.3,
        "EDGE_SAFETY_MARGIN": 0.00020,
        "MIN_HOLD_STEPS": 3,
        "MIN_TRADE_GAP_STEPS": 2,
        "HYSTERESIS_REQUIRED_STREAK": 1,
        "HYSTERESIS_ALLOW_IF_EDGE_MULT": 1.4,
        "TRADE_RATE_WINDOW_STEPS": 240,
        "MAX_TRADES_PER_WINDOW": 60,
        "TRADE_RATE_SELL_BYPASS_EDGE_MULT": 1.7,
        "TURNOVER_PENALTY": 0.00005,
        "TURNOVER_BUDGET_MULTIPLIER": 3.0,
        "TURNOVER_BUDGET_MIN": 1.0,
        "TURNOVER_BUDGET_PENALTY": 0.22,
        "ENABLE_STUCK_UNFREEZE": True,
        "ENABLE_HARD_RISK_EXITS": True,
        "STOP_LOSS_PCT": 0.010,
        "TRAILING_STOP_PCT": 0.012,
        "MAX_POSITION_HOLD_STEPS": 60,
        "POSITION_FRACTION_MIN": 0.03,
        "POSITION_FRACTION_MAX": 0.15,
        "REWARD_SCALE": 70.0,
        "ADAPTIVE_REWARD_DECAY": 0.10,
        "DRAWDOWN_PENALTY": 0.55,
        "DIRECTIONAL_SHAPING_WEIGHT": 0.06,
        "FLAT_HOLD_MISS_PENALTY_WEIGHT": 0.03,
        "ENABLE_PROB_SHAPING": False,
        "ENABLE_MTF_CONFIRMATION": False,
    },

    # 5m profitability-focused preset: trade less, cut losses early, and only
    # participate when edge clearly exceeds full friction. This is intended as
    # a safer default for live experiments; it prioritizes avoiding negative
    # expectancy churn over frequent trading.
    "tf_5m_profit_focus": {
        "WARMUP_TRADES_BEFORE_GATING": 0,
        # Exploration down: avoid constant probe trades when we don't have edge.
        "POSTERIOR_SCALE": 0.04,
        "POSTERIOR_DECAY_HALF_LIFE_STEPS": 8_000,
        "POSTERIOR_SCALE_MIN": 0.006,
        "FORGETTING_FACTOR": 0.997,

        # Require edge that clears all-in costs (fee + slippage + safety margin).
        # Tuned to reduce marginal entries that can't beat friction while still allowing
        # enough trades to learn / validate on 5m.
        # Raise edge requirement to reduce BUY proposals (trainer clamps using EDGE_THRESHOLD_MAX).
        "EDGE_THRESHOLD_MAX": 0.0045,
        "EDGE_THRESHOLD": 0.0038,
        # Tighten cost-aware edge requirement to reduce marginal buy impulses.
        "COST_EDGE_MULT": 2.4,
        "EDGE_SAFETY_MARGIN": 0.00035,
        "GATE_SAFETY_MARGIN": 0.00015,

        # Reduce flip-flopping / churn.
        # 5m steps: 6 = 30m minimum hold; 4 = 20m minimum trade gap.
        "MIN_HOLD_STEPS": 6,
        "MIN_TRADE_GAP_STEPS": 4,
        # Require more persistence before acting on a BUY/SELL signal.
        "HYSTERESIS_REQUIRED_STREAK": 3,
        "HYSTERESIS_ALLOW_IF_EDGE_MULT": 1.8,
        "TRADE_RATE_WINDOW_STEPS": 240,
        # Relax trade-rate limiter: fewer forced HOLDs due to trade-rate budget.
        "MAX_TRADES_PER_WINDOW": 30,
        "TRADE_RATE_SELL_BYPASS_EDGE_MULT": 2.8,

        # Cooldown: allow risk-reducing exits sooner; avoid delayed sells becoming larger losses.
        "COOLDOWN_STRONG_EDGE_MULT": 1.6,
        "COOLDOWN_STRONG_MIN_GAP_STEPS": 2,

        # Turnover discipline and adaptive budget floors suitable for 5m.
        "TURNOVER_PENALTY": 0.00010,
        "TURNOVER_BUDGET_MULTIPLIER": 1.3,
        "TURNOVER_BUDGET_MIN": 1.0,
        "TURNOVER_BUDGET_PENALTY": 0.30,

        # Prefer clean exits while we are still validating the edge.
        "PARTIAL_SELLS": False,

        # Loss containment overlay (mandatory for profitability attempts).
        "ENABLE_HARD_RISK_EXITS": True,
        # Option A: shrink losers
        "STOP_LOSS_PCT": 0.008,
        "TRAILING_STOP_PCT": 0.012,
        "MAX_POSITION_HOLD_STEPS": 60,

        # Reward shaping: keep learning stable on sparse, higher-quality trades.
        "REWARD_SCALE": 55.0,
        # Debugging: keep reward scale stable; avoid pinning at REWARD_SCALE_MAX in low-vol regimes.
        "ADAPTIVE_REWARD_SCALE": False,
        "ADAPTIVE_REWARD_DECAY": 0.05,
        "DRAWDOWN_PENALTY": 0.60,
        # Reduce flat-state "bribe" that can encourage marginal buys.
        # Reduce directional shaping to curb impulsive BUY proposals
        "DIRECTIONAL_SHAPING_WEIGHT": 0.02,
        "FLAT_HOLD_MISS_PENALTY_WEIGHT": 0.01,
        "ENABLE_PROB_SHAPING": False,
    },
}

# Track the last applied profile so runs can record configuration provenance.
ACTIVE_PROFILE: str | None = None

def apply_profile(name: str | None) -> None:
    if not name:
        return
    # Record provenance for metrics/debugging.
    globals()["ACTIVE_PROFILE"] = name
    profile = PROFILES.get(name)
    if not profile:
        return
    for key, value in profile.items():
        globals()[key] = value
    # keep penalty profile wiring consistent
    PENALTY_PROFILES["train"] = {
        "drawdown_penalty": globals().get("DRAWDOWN_PENALTY", DRAWDOWN_PENALTY) * 0.8,
        "turnover_budget_penalty": globals().get("TURNOVER_BUDGET_PENALTY", TURNOVER_BUDGET_PENALTY) * 0.75,
    }
    PENALTY_PROFILES["eval"] = {
        "drawdown_penalty": globals().get("DRAWDOWN_PENALTY", DRAWDOWN_PENALTY),
        "turnover_budget_penalty": globals().get("TURNOVER_BUDGET_PENALTY", TURNOVER_BUDGET_PENALTY),
    }
