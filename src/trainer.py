# src/trainer.py
from __future__ import annotations

import csv
import json
import math
from collections import Counter, deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.agent import AgentState, BanditAgent, ACTIONS
from src import config
from src.indicators import INDICATOR_COLUMNS
from src.metrics import compute_max_drawdown, compute_sharpe_ratio, total_return
from src.persistence import atomic_write_json
from src.timeframe import periods_per_year


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


@dataclass
class Portfolio:
    cash: float = 1000.0
    position: float = 0.0
    entry_price: float = 0.0
    entry_value: float = 0.0
    def value(self, price: float) -> float:
        return self.cash + self.position * price

@dataclass
class StepResult:
    action: str
    proposed_action: str
    trainer_reward: float
    scaled_reward: float
    trade_executed: bool
    trade_size: float
    notional_traded: float
    fee_paid: float
    turnover_penalty: float
    slippage_paid: float
    refilled: bool
    realized_pnl: float
    edge_margin: float
    hold_reason: str | None
    forced_exit: bool = False
    forced_exit_reason: str | None = None
    gate_blocked: bool = False
    timing_blocked: bool = False
    budget_blocked: bool = False
    stuck_relax: bool = False

@dataclass
class TrainerState:
    version: int = 5
    run_id: str = ""
    steps: int = 0
    prev_price: float | None = None
    prev_value: float | None = None
    refill_count: int = 0
    total_fee_paid: float = 0.0
    total_turnover_penalty_paid: float = 0.0
    total_slippage_paid: float = 0.0
    total_notional_traded: float = 0.0
    # decision_count is total_steps; executed_trade_count counts only actual fills
    executed_trade_count: int = 0
    buy_legs: int = 0
    sell_legs: int = 0
    winning_sell_legs: int = 0
    # Per-sell-leg net PnL diagnostics (used for avg win/loss and expectancy)
    win_pnl_sum: float = 0.0
    loss_pnl_sum: float = 0.0
    win_pnl_count: int = 0
    loss_pnl_count: int = 0
    # legacy fields kept for backwards compatibility (older dashboards/state)
    total_trades: int = 0
    successful_trades: int = 0
    sell_trades: int = 0
    winning_sells: int = 0
    portfolio_cash: float = 0.0
    portfolio_position: float = 0.0
    portfolio_entry_price: float = 0.0
    portfolio_entry_value: float = 0.0
    total_steps: int = 0
    positive_steps: int = 0
    last_trade_step: int = -1
    last_entry_step: int = -1
    # action hysteresis state (persisted for resume)
    last_direction: str | None = None        # last executed non-hold action
    flip_candidate: str | None = None       # proposed direction we're considering flipping to
    flip_streak: int = 0                   # consecutive proposals for flip_candidate
    gate_blocks: int = 0
    timing_blocks: int = 0
    budget_blocks: int = 0
    penalty_profile: str = "train"
    reward_scale: float = config.REWARD_SCALE
    drawdown_budget: float = config.DRAWDOWN_BUDGET
    turnover_budget_multiplier: float = config.TURNOVER_BUDGET_MULTIPLIER
    # reward normalization / shaping state (persisted for resume)
    reward_baseline_ema: float = 0.0
    reward_var_ema: float = config.REWARD_VAR_INIT
    prob_alpha: float = config.PROB_SHAPING_ALPHA0
    prob_beta: float = config.PROB_SHAPING_BETA0
    def to_json(self, path: Path) -> None:
        atomic_write_json(path, asdict(self))
    @classmethod
    def from_json(cls, path: Path) -> "TrainerState":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        defaults = asdict(cls())
        defaults.update(data)
        return cls(**defaults)

def build_features(row: pd.Series, portfolio: Portfolio) -> np.ndarray:
    price = float(row["close"])
    indicator_values = np.nan_to_num(
        row[INDICATOR_COLUMNS].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )

    portfolio_value = portfolio.value(price)
    cash_frac = portfolio.cash / max(portfolio_value, 1e-6)
    unrealized_ret = (
        (price / portfolio.entry_price) - 1.0 if portfolio.position > 0 and portfolio.entry_price > 0 else 0.0
    )
    position_value = portfolio.position * price
    pos_frac = position_value / max(portfolio_value, 1e-6)
    time_since_trade = float(row.get("time_since_trade", 0.0))

    features = np.concatenate(
        [
            indicator_values,
            [pos_frac, cash_frac, unrealized_ret, time_since_trade],
        ]
    )
    return np.clip(features, -config.FEATURE_CLIP, config.FEATURE_CLIP)

class Trainer:
    def __init__(
        self,
        agent: BanditAgent,
        initial_cash: float = config.INITIAL_CASH,
        min_cash: float = config.MIN_TRAINING_CASH,
        timeframe: str | None = None,
        penalty_profile: str = "train",
    ):
        self.agent = agent
        self.portfolio = Portfolio(cash=initial_cash)
        self.history: List[Tuple[int, str, float, float]] = []
        self._last_flushed_trade_idx = 0
        self.total_steps = 0
        self.positive_steps = 0
        self.initial_cash = initial_cash
        self.min_cash = min_cash
        self.refill_count = 0
        self.total_fee_paid = 0.0
        self.total_turnover_penalty_paid = 0.0
        self.total_slippage_paid = 0.0
        # execution / leg counters (separate from decisions)
        self.executed_trade_count = 0
        self.buy_legs = 0
        self.sell_legs = 0
        self.winning_sell_legs = 0
        # PnL diagnostics (per SELL leg, net of frictions)
        self._win_pnl_sum: float = 0.0
        self._loss_pnl_sum: float = 0.0
        self._win_pnl_count: int = 0
        self._loss_pnl_count: int = 0
        # legacy counters (kept for compatibility)
        self.steps = 0
        self.sell_trades = 0
        self.winning_sells = 0
        self._last_price: float | None = None
        self._last_value: float | None = None
        self.last_trade_step = -1
        self.last_entry_step = -1
        # action hysteresis state
        self._last_direction: str | None = None
        self._flip_candidate: str | None = None
        self._flip_streak: int = 0
        self.last_data_is_live: bool | None = None
        self._equity_curve: list[float] = []
        self._return_history: deque[float] = deque(maxlen=config.RETURN_HISTORY_WINDOW)
        self._recent_actions: deque[str] = deque(maxlen=config.ACTION_HISTORY_WINDOW)
        # Executed-action distribution (what actually happened after gates/cooldowns).
        self._action_counter: Counter[str] = Counter()
        # Proposed-action distribution (what the agent wanted to do before guards).
        self._proposed_action_counter: Counter[str] = Counter()
        # HOLD diagnostics: why an action was converted into HOLD.
        self._hold_reason_counter: Counter[str] = Counter()
        # Gate diagnostics: which gate type blocked (cost, mtf, etc.).
        self._gate_reason_counter: Counter[str] = Counter()
        # Pre-mask gate diagnostics: actions removed *before* proposed_action is recorded.
        self._premask_gate_reason_counter: Counter[str] = Counter()
        # Hard risk-exit diagnostics (forced exits; off by default).
        self.forced_exit_count: int = 0
        self._forced_exit_reason_counter: Counter[str] = Counter()
        self._turnover_window: deque[float] = deque(maxlen=config.TURNOVER_BUDGET_WINDOW)
        # Separate anti-churn window: count executed trade legs (not notional)
        self._trade_count_window: deque[int] = deque(maxlen=int(getattr(config, 'TRADE_RATE_WINDOW_STEPS', 0) or 0))
        # cumulative notional traded (for stable metrics; windowed sums can mislead)
        self.total_notional_traded: float = 0.0
        self.timeframe = timeframe or agent.state.timeframe or config.DEFAULT_TIMEFRAME
        self.periods_per_year = periods_per_year(self.timeframe)
        self.penalty_profile = penalty_profile
        self.gate_blocks = 0
        self.timing_blocks = 0
        self.budget_blocks = 0
        self._last_direction = None
        self._flip_candidate = None
        self._flip_streak = 0
        self._hold_reason_log: list[tuple[int, str, str]] = []
        # per-position buy-leg cap to avoid repeated scaling-in on noisy 1m
        self._buy_legs_current: int = 0
        # adaptive knobs (start from config values)
        self.reward_scale = config.REWARD_SCALE
        self.drawdown_budget = config.DRAWDOWN_BUDGET
        self.turnover_budget_multiplier = config.TURNOVER_BUDGET_MULTIPLIER
        self._reward_baseline = 0.0
        self._reward_var_ema = float(config.REWARD_VAR_INIT)
        self._prob_alpha = float(config.PROB_SHAPING_ALPHA0)
        self._prob_beta = float(config.PROB_SHAPING_BETA0)
        # warn-once registry for percentile sanitization (data hygiene)
        self._percentile_warned: set[str] = set()

    def reset_portfolio(self) -> None:
        """Reset portfolio and tracking buffers to their initial state."""
        self.portfolio = Portfolio(cash=self.initial_cash)
        self.history.clear()
        self._last_flushed_trade_idx = 0
        self._last_price = None
        self._last_value = None
        self._equity_curve.clear()
        self._return_history.clear()
        self._recent_actions.clear()
        self._action_counter.clear()
        self._proposed_action_counter.clear()
        self._hold_reason_counter.clear()
        self._gate_reason_counter.clear()
        self._premask_gate_reason_counter.clear()
        self.forced_exit_count = 0
        self._forced_exit_reason_counter.clear()
        self._turnover_window.clear()
        if hasattr(self, '_trade_count_window'):
            self._trade_count_window.clear()
        self.total_notional_traded = 0.0
        self.steps = 0
        self.total_steps = 0
        self.positive_steps = 0
        self.total_fee_paid = 0.0
        self.total_turnover_penalty_paid = 0.0
        self.total_slippage_paid = 0.0
        self.executed_trade_count = 0
        self.buy_legs = 0
        self.sell_legs = 0
        self.winning_sell_legs = 0
        self._win_pnl_sum = 0.0
        self._loss_pnl_sum = 0.0
        self._win_pnl_count = 0
        self._loss_pnl_count = 0
        self.sell_trades = 0
        self.winning_sells = 0
        self.gate_blocks = 0
        self.timing_blocks = 0
        self.budget_blocks = 0
        self._hold_reason_log.clear()
        self._buy_legs_current = 0
        self.reward_scale = config.REWARD_SCALE
        self.drawdown_budget = config.DRAWDOWN_BUDGET
        self.turnover_budget_multiplier = config.TURNOVER_BUDGET_MULTIPLIER
        self._reward_baseline = 0.0
        self._reward_var_ema = float(config.REWARD_VAR_INIT)
        self._prob_alpha = float(config.PROB_SHAPING_ALPHA0)
        self._prob_beta = float(config.PROB_SHAPING_BETA0)
        # warn-once registry for percentile sanitization (data hygiene)
        self._percentile_warned: set[str] = set()

    # --- helpers --------------------------------------------------------------

    def _log_action(self, action: str) -> None:
        if len(self._recent_actions) == self._recent_actions.maxlen:
            oldest = self._recent_actions[0]
            self._action_counter[oldest] -= 1
            if self._action_counter[oldest] <= 0:
                del self._action_counter[oldest]
        self._recent_actions.append(action)
        self._action_counter[action] += 1

    def _update_adaptive_risk_controls(self) -> None:
        if not config.ADAPTIVE_REWARD_SCALE:
            return
        returns = np.asarray(self._return_history, dtype=float)
        min_obs = max(25, config.RETURN_HISTORY_WINDOW // 4)
        if len(returns) < min_obs:
            return

        # scale reward to keep tanh input in a useful range and respond to performance
        vol = float(np.std(returns))
        sharpe = compute_sharpe_ratio(returns, periods_per_year=self.periods_per_year)
        target_vol = max(config.ADAPTIVE_TARGET_RETURN_VOL, 1e-6)
        vol_adj = _clamp(target_vol / max(vol, 1e-6), 0.5, 2.5)
        desired_reward_scale = config.REWARD_SCALE * vol_adj

        if sharpe < config.ADAPTIVE_SHARPE_SOFT:
            desired_reward_scale *= 0.9
        elif sharpe > config.ADAPTIVE_SHARPE_STRONG:
            desired_reward_scale *= 1.1

        mix = _clamp(config.ADAPTIVE_REWARD_DECAY, 0.0, 1.0)
        self.reward_scale = (1 - mix) * self.reward_scale + mix * desired_reward_scale
        self.reward_scale = _clamp(self.reward_scale, config.REWARD_SCALE_MIN, config.REWARD_SCALE_MAX)

        # adapt risk budgets with a soft performance score
        perf_score = math.tanh(sharpe)
        dd_target = config.DRAWDOWN_BUDGET * (1 + perf_score * config.ADAPTIVE_RISK_RANGE)
        to_target = config.TURNOVER_BUDGET_MULTIPLIER * (1 + 0.6 * perf_score * config.ADAPTIVE_RISK_RANGE)

        self.drawdown_budget = (1 - mix) * self.drawdown_budget + mix * dd_target
        self.turnover_budget_multiplier = (1 - mix) * self.turnover_budget_multiplier + mix * to_target

        self.drawdown_budget = _clamp(self.drawdown_budget, config.DRAWDOWN_BUDGET_MIN, config.DRAWDOWN_BUDGET_MAX)
        self.turnover_budget_multiplier = _clamp(
            self.turnover_budget_multiplier, config.TURNOVER_BUDGET_MIN, config.TURNOVER_BUDGET_MAX
        )

    def _stuck_adaptation(self, posterior_scale_override: float | None) -> tuple[float | None, float, bool]:
        """Return (posterior_scale_override, edge_threshold, stuck_relax).

        stuck_relax indicates we are in a HOLD-dominant regime and should loosen
        gating (and optionally boost exploration) to avoid freezing.
        """
        if not config.ENABLE_STUCK_UNFREEZE:
            return posterior_scale_override, config.EDGE_THRESHOLD, False

        window = config.STUCK_HOLD_WINDOW
        min_obs = max(25, window // 5)
        total = len(self._recent_actions)
        if total < min_obs:
            return posterior_scale_override, config.EDGE_THRESHOLD, False

        hold_ratio = self._action_counter.get("hold", 0) / max(total, 1)
        if hold_ratio < config.STUCK_HOLD_RATIO:
            return posterior_scale_override, config.EDGE_THRESHOLD, False

        base_scale = posterior_scale_override
        if base_scale is None:
            base_scale = self.agent.posterior_scale
        boosted_scale = base_scale + config.STUCK_POSTERIOR_BOOST
        relaxed_edge = min(config.EDGE_THRESHOLD, config.STUCK_EDGE_THRESHOLD)
        if hold_ratio >= 0.99:
            relaxed_edge = 0.0
        return boosted_scale, relaxed_edge, True

    def _regime_adjust_edge(self, row: pd.Series, base_edge: float, *, stuck_relax: bool) -> float:
        """Adjust edge threshold based on a volatility percentile column.

        Tighten edge in low-vol chop; relax edge in high-vol trend.
        Disabled when stuck_relax is active (we already loosen gates there).
        """
        edge = float(base_edge)
        if not config.ENABLE_REGIME_EDGE_GATING or stuck_relax:
            return edge
        try:
            v = float(row.get(config.REGIME_EDGE_VOL_COL, np.nan))
        except Exception:
            v = float('nan')
        if not np.isfinite(v):
            return edge
        v = self._sanitize_unit_percentile(v, col=str(config.REGIME_EDGE_VOL_COL))
        if v < config.REGIME_EDGE_LOW_PCT:
            edge *= float(config.REGIME_EDGE_TIGHTEN_MULT)
        elif v > config.REGIME_EDGE_HIGH_PCT:
            edge *= float(config.REGIME_EDGE_RELAX_MULT)
        return _clamp(edge, config.EDGE_THRESHOLD_MIN, config.EDGE_THRESHOLD_MAX)

    def _walk_forward_returns(self, folds: int) -> list[float]:
        if folds <= 1 or len(self._equity_curve) < folds + 1:
            return []
        fold_size = len(self._equity_curve) // folds
        if fold_size <= 0:
            return []
        out: list[float] = []
        start = 0
        for k in range(folds):
            end = (k + 1) * fold_size if k < folds - 1 else len(self._equity_curve)
            if end - start < 2:
                continue
            start_val = self._equity_curve[start]
            end_val = self._equity_curve[end - 1]
            out.append(total_return(start_val, end_val))
            start = end
        return out

    def _maybe_refill_portfolio(self, price: float) -> bool:
        # why: prevent learning stalls after burn-down
        if self.portfolio.value(price) < self.min_cash:
            self.portfolio = Portfolio(cash=self.initial_cash)
            self._equity_curve.clear()
            self._return_history.clear()
            self._turnover_window.clear()
            if hasattr(self, '_trade_count_window'):
                self._trade_count_window.clear()
            self.refill_count += 1
            return True
        return False

    # --- properties -----------------------------------------------------------

    @property
    def success_rate(self) -> float:
        return 0.0 if self.total_steps == 0 else (self.positive_steps / self.total_steps) * 100

    @property
    def step_win_rate(self) -> float:
        return self.success_rate

    @property
    def trade_win_rate(self) -> float:
        # Prefer per-sell-leg win rate (works with partial sells). Falls back to legacy.
        if getattr(self, 'sell_legs', 0) > 0:
            return float(getattr(self, 'winning_sell_legs', 0)) / max(1, int(getattr(self, 'sell_legs', 0)))
        return self.winning_sells / max(1, self.sell_trades)

    @property
    def avg_win_pnl(self) -> float:
        return 0.0 if self._win_pnl_count <= 0 else (self._win_pnl_sum / self._win_pnl_count)

    @property
    def avg_loss_pnl(self) -> float:
        # negative value (average losing SELL-leg PnL)
        return 0.0 if self._loss_pnl_count <= 0 else (self._loss_pnl_sum / self._loss_pnl_count)

    @property
    def win_loss_ratio(self) -> float:
        aw = self.avg_win_pnl
        al = self.avg_loss_pnl
        denom = abs(al) if al != 0 else 0.0
        return 0.0 if denom <= 0 else (aw / denom)

    @property
    def expectancy_pnl_per_sell_leg(self) -> float:
        total_legs = self._win_pnl_count + self._loss_pnl_count
        total = self._win_pnl_sum + self._loss_pnl_sum
        return 0.0 if total_legs <= 0 else (total / total_legs)

    @property
    def action_distribution(self) -> dict[str, float]:
        total = sum(self._action_counter.values())
        return {} if total <= 0 else {a: c / total for a, c in self._action_counter.items()}

    @property
    def sharpe_ratio(self) -> float:
        return compute_sharpe_ratio(list(self._return_history), periods_per_year=self.periods_per_year)

    @property
    def max_drawdown(self) -> float:
        return compute_max_drawdown(self._equity_curve)

    @property
    def total_return(self) -> float:
        # Use the last equity-curve point if available (it's computed using price_next),
        # otherwise fall back to current marked-to-market value.
        if self._equity_curve:
            final_value = float(self._equity_curve[-1])
            return total_return(self.initial_cash, final_value)
        price = self._last_price if self._last_price is not None else 0.0
        return total_return(self.initial_cash, self.portfolio.value(price))

    # --- core step ------------------------------------------------------------

    def _sanitize_unit_percentile(self, v: float, *, col: str) -> float:
        """Coerce percentile-like columns to unit interval [0,1].

        Some indicator pipelines emit percentiles in 0..100. If we treat those
        as 0..1, MTF gates can hard-veto almost every entry.
        """
        if not getattr(config, "SANITIZE_PERCENTILE_COLUMNS", True):
            return float(v)
        if not np.isfinite(v):
            return float(v)
        x = float(v)
        # Heuristic: 0..100 percentile.
        if 1.0 < x <= 100.0:
            if col not in self._percentile_warned:
                print(f"[warn] {col} looks like 0..100; rescaling to 0..1 (value={x})")
                self._percentile_warned.add(col)
            x = x / 100.0
        # If it's slightly outside due to numeric drift, clamp.
        if x < 0.0 or x > 1.0:
            if col not in self._percentile_warned:
                print(f"[warn] {col} outside [0,1]; clamping (value={x})")
                self._percentile_warned.add(col)
            x = _clamp(x, 0.0, 1.0)
        return float(x)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _dynamic_fraction(self, margin: float) -> float:
        conf = self._sigmoid(config.CONFIDENCE_K * max(0.0, margin))
        lo, hi = config.POSITION_FRACTION_MIN, config.POSITION_FRACTION_MAX
        return lo + conf * (hi - lo)

    def step(
        self,
        row: pd.Series,
        next_row: pd.Series,
        step_idx: int,
        *,
        train: bool = True,
        posterior_scale_override: float | None = None,
    ) -> StepResult:
        price_now = float(row["close"])
        price_next = float(next_row["close"])
        refilled = self._maybe_refill_portfolio(price_now)

        # Warmup: during the first N executed trades we relax gating/cooldowns so the
        # agent can explore and collect informative samples.
        warmup_active = self.agent.state.trades < config.WARMUP_TRADES_BEFORE_GATING

        # --- early stuck detection (MUST be before gating/action masking) -------
        posterior_scale_effective, edge_threshold, stuck_relax = self._stuck_adaptation(
            posterior_scale_override
        )
        # Regime-adaptive edge gating: tighten in chop / relax in trend.
        # Disabled when stuck_relax is active (we already loosen gates there).
        edge_threshold = self._regime_adjust_edge(row, edge_threshold, stuck_relax=stuck_relax)

        # ------------------------------------------------------------------
        # Hard risk exits (optional): evaluate BEFORE policy action.
        #
        # This is intentionally not "gating". It is a risk overlay that can
        # override the policy to cut tail risk. It is OFF by default.
        # ------------------------------------------------------------------
        forced_exit = False
        forced_exit_reason: str | None = None
        forced_action: str | None = None
        if getattr(config, "ENABLE_HARD_RISK_EXITS", False) and self.portfolio.position > 0:
            entry = float(self.portfolio.entry_price or 0.0)
            if entry > 0:
                unreal = (price_now / entry) - 1.0
                hold_steps = (self.steps - self.last_entry_step) if self.last_entry_step >= 0 else 0
                if hold_steps >= int(getattr(config, "MAX_POSITION_HOLD_STEPS", 10**9)):
                    forced_action = "sell"
                    forced_exit_reason = "time_stop"
                if unreal <= -float(getattr(config, "STOP_LOSS_PCT", 0.0)):
                    forced_action = "sell"
                    forced_exit_reason = "stop_loss"
                # trailing stop from peak while in position
                if not hasattr(self, "_trail_peak_price"):
                    self._trail_peak_price = price_now
                self._trail_peak_price = max(float(self._trail_peak_price), price_now)
                trail_from_peak = (price_now / max(float(self._trail_peak_price), 1e-9)) - 1.0
                if trail_from_peak <= -float(getattr(config, "TRAILING_STOP_PCT", 0.0)):
                    forced_action = "sell"
                    forced_exit_reason = "trailing_stop"
        else:
            # Reset peak tracker when flat
            if hasattr(self, "_trail_peak_price"):
                self._trail_peak_price = price_now

        features = build_features(row, self.portfolio)

        # ------------------------------------------------------------------
        # Pre-mask actions using *timing feasibility* so the agent does not
        # constantly propose trades that the cooldown will veto.
        #
        # This fixes the "proposed_action_distribution is trade-heavy but
        # executed_action_distribution is HOLD-heavy" mismatch and reduces
        # distorted learning signals on 1m data.
        # ------------------------------------------------------------------
        base_gap = int(getattr(config, "MIN_TRADE_GAP_STEPS", 0))
        base_hold = int(getattr(config, "MIN_HOLD_STEPS", 0))
        eff_gap = base_gap
        eff_hold = base_hold
        if getattr(config, "ENABLE_ADAPTIVE_COOLDOWN", False):
            # volatility percentile in [0,1] (if missing, fall back to neutral 0.5).
            vol_col = getattr(config, "COOLDOWN_VOL_COL", getattr(config, "REGIME_EDGE_VOL_COL", ""))
            use_proxy = True
            if vol_col:
                try:
                    raw_v = row.get(vol_col, None)
                    if raw_v is not None and np.isfinite(raw_v):
                        v = float(raw_v)
                        if 0.0 <= v <= 1.0:
                            vol_pct = v
                            use_proxy = False
                except Exception:
                    pass
            if use_proxy:
                ret = 0.0
                if self._last_price is not None and self._last_price > 0:
                    ret = abs(price_now / self._last_price - 1.0)
                decay = float(getattr(config, 'ADAPTIVE_COOLDOWN_RET_EMA_DECAY', 0.05))
                decay = _clamp(decay, 0.0, 1.0)
                if not hasattr(self, '_absret_ema'):
                    self._absret_ema = ret
                else:
                    self._absret_ema = (1 - decay) * float(self._absret_ema) + decay * ret
                base = max(float(getattr(self, '_absret_ema', 0.0)), 1e-12)
                ratio = ret / base
                vol_pct = 0.5 * (1.0 + math.tanh((ratio - 1.0) / 1.5))
            vol_pct = _clamp(float(vol_pct), 0.0, 1.0)
            low = float(getattr(config, "COOLDOWN_VOL_LOW_PCT", 0.30))
            high = float(getattr(config, "COOLDOWN_VOL_HIGH_PCT", 0.70))
            if vol_pct <= low:
                gap_mult = float(getattr(config, "COOLDOWN_GAP_TIGHTEN_MULT", 1.0))
                hold_mult = float(getattr(config, "COOLDOWN_HOLD_TIGHTEN_MULT", 1.0))
            elif vol_pct >= high:
                gap_mult = float(getattr(config, "COOLDOWN_GAP_RELAX_MULT", 1.0))
                hold_mult = float(getattr(config, "COOLDOWN_HOLD_RELAX_MULT", 1.0))
            else:
                t = (vol_pct - low) / max(high - low, 1e-9)
                gap_mult = (1 - t) * float(getattr(config, "COOLDOWN_GAP_TIGHTEN_MULT", 1.0)) + t * float(getattr(config, "COOLDOWN_GAP_RELAX_MULT", 1.0))
                hold_mult = (1 - t) * float(getattr(config, "COOLDOWN_HOLD_TIGHTEN_MULT", 1.0)) + t * float(getattr(config, "COOLDOWN_HOLD_RELAX_MULT", 1.0))
            gap_mult = _clamp(gap_mult, min(float(getattr(config, "COOLDOWN_GAP_RELAX_MULT", 1.0)), 1.0), 1.0)
            hold_mult = _clamp(hold_mult, min(float(getattr(config, "COOLDOWN_HOLD_RELAX_MULT", 1.0)), 1.0), 1.0)
            turnover_budget = self.initial_cash * self.turnover_budget_multiplier
            turnover_ratio = float(sum(self._turnover_window)) / max(turnover_budget, 1e-6)
            sens = float(getattr(config, "COOLDOWN_TURNOVER_SENSITIVITY", 0.0))
            scale = 1.0 + max(0.0, turnover_ratio - 1.0) * max(0.0, sens)
            eff_gap = int(round(base_gap * gap_mult * scale))
            eff_hold = int(round(base_hold * hold_mult * scale))
            eff_gap = max(eff_gap, int(getattr(config, "COOLDOWN_MIN_GAP_FLOOR", 0)))
            eff_hold = max(eff_hold, int(getattr(config, "COOLDOWN_MIN_HOLD_FLOOR", 0)))
        gap_ok = self.last_trade_step < 0 or (self.steps - self.last_trade_step) >= eff_gap
        hold_ok = self.last_entry_step < 0 or (self.steps - self.last_entry_step) >= eff_hold
        since_last_trade = (self.steps - self.last_trade_step) if self.last_trade_step >= 0 else 10**9
        # In stuck mode, allow limited relaxation of the trade-gap constraint (never below the floor).
        gap_floor = int(getattr(config, "COOLDOWN_MIN_GAP_FLOOR", 0))
        stuck_gap_ok = bool(stuck_relax) and (since_last_trade >= gap_floor)

        allowed_actions = ["hold"]
        # Allow BUY when flat; optionally allow *limited* scaling-in via MAX_BUY_LEGS_PER_POSITION.
        if self.portfolio.cash > 0:
            if self.portfolio.position <= 0:
                # If we're inside the trade-gap cooldown, do not even offer BUY to the agent.
                # This prevents thousands of doomed BUY proposals (cooldown_gap blocks).
                if warmup_active or gap_ok or stuck_gap_ok:
                    allowed_actions.append("buy")
            else:
                max_legs = int(getattr(config, "MAX_BUY_LEGS_PER_POSITION", 1))
                if max_legs > 1 and self._buy_legs_current < max_legs:
                    if warmup_active or gap_ok or stuck_gap_ok:
                        allowed_actions.append("buy")
        if self.portfolio.position > 0:
            # SELL: do not offer during trade-gap cooldown unless it qualifies for a strong-exit bypass.
            # This prevents the recurring "proposed SELL >> executed SELL" mismatch.
            if warmup_active or gap_ok or stuck_gap_ok:
                allowed_actions.append("sell")
            else:
                # Only allow if we can at least satisfy the strong-min-gap requirement;
                # the actual edge check happens after we compute margins from sampled scores.
                since_last_trade = (self.steps - self.last_trade_step) if self.last_trade_step >= 0 else 10**9
                strong_min_gap = int(getattr(config, 'COOLDOWN_STRONG_MIN_GAP_STEPS', 0))
                strong_min_gap = min(strong_min_gap, max(eff_gap, int(getattr(config, "COOLDOWN_MIN_GAP_FLOOR", 0))))
                if since_last_trade >= strong_min_gap:
                    allowed_actions.append("sell")

        # (moved earlier) stuck adaptation + regime edge adjustment are computed before action masking

        if forced_action is not None:
            # Policy override: log as a forced exit; the agent still learns from the executed action.
            action = forced_action
            sampled_scores = np.zeros(len(ACTIONS), dtype=float)
            means = np.zeros(len(ACTIONS), dtype=float)
            forced_exit = True
        else:
            action, sampled_scores, means = self.agent.act_with_scores(
                features,
                allowed=allowed_actions,
                step=self.steps,
                posterior_scale_override=posterior_scale_effective,
            )

        # --- cost-aware + cooldown-aware action masking (prevents spam) --------
        # Goal: keep *proposed* actions aligned with what can actually execute, so the
        # agent doesn't learn to "ask" for trades that will be vetoed by costs/cooldowns.
        hold_idx = ACTIONS.index("hold")
        margin_scale = max(config.WEIGHT_CLIP * float(getattr(config, "MARGIN_SCALE_MULT", 1.0)), 1e-9)
        buy_margin_raw = sampled_scores[ACTIONS.index("buy")] - sampled_scores[hold_idx]
        sell_margin_raw = sampled_scores[ACTIONS.index("sell")] - sampled_scores[hold_idx]
        buy_margin = math.tanh(buy_margin_raw / margin_scale)
        sell_margin = math.tanh(sell_margin_raw / margin_scale)

        if config.COST_AWARE_GATING:
            est_cost = (config.FEE_RATE + config.SLIPPAGE_RATE + config.GATE_SAFETY_MARGIN)
            if stuck_relax:
                # In stuck mode, require only raw estimated costs (+ safety), not the multiplied threshold.
                cost_edge = max(edge_threshold, est_cost + config.EDGE_SAFETY_MARGIN)
            else:
                cost_edge = max(edge_threshold, config.COST_EDGE_MULT * est_cost) + config.EDGE_SAFETY_MARGIN
        else:
            cost_edge = edge_threshold

        if not warmup_active:
            feasible = list(allowed_actions)

            # 1) Cost feasibility: do not propose trades that do not clear cost_edge.
            if "buy" in feasible and buy_margin < cost_edge:
                feasible.remove("buy")
                # Pre-mask veto: the action never becomes proposed_action, so without
                # this counter it would be invisible in gate_blocks/gate_reason_counts.
                self._premask_gate_reason_counter["cost_gate_buy"] += 1
            # SELL: do not pre-mask exits on cost edge. Exits are *risk reducing* and the agent
            # must always have the option to close exposure; otherwise the system can HOLD-freeze
            # in-position for thousands of steps when margins are small.
            # We still account for real execution frictions in PnL and learning reward.
            if "sell" in feasible and self.portfolio.position <= 0:
                # (defensive) Only apply sell cost gating when flat (should be rare).
                if sell_margin < cost_edge:
                    feasible.remove("sell")
                    self._premask_gate_reason_counter["cost_gate_sell"] += 1

            # 2) Cooldown feasibility for SELL: if we are inside the trade-gap cooldown,
            # only allow SELL proposals when they qualify for the strong-exit bypass.
            if "sell" in feasible and (not gap_ok) and (not stuck_relax):
                strong_min_gap = int(getattr(config, 'COOLDOWN_STRONG_MIN_GAP_STEPS', 0))
                strong_min_gap = min(strong_min_gap, max(eff_gap, int(getattr(config, "COOLDOWN_MIN_GAP_FLOOR", 0))))
                min_gap_ok = since_last_trade >= strong_min_gap
                bypass_allowed = True if config.COOLDOWN_BYPASS_SELL_ONLY else True
                strong_sell = (sell_margin >= (config.COOLDOWN_STRONG_EDGE_MULT * cost_edge)) and bypass_allowed and min_gap_ok
                if not strong_sell:
                    feasible.remove("sell")
                    # Pre-mask veto: sell removed due to cooldown feasibility.
                    self._premask_gate_reason_counter["cooldown_sell"] += 1

            # Ensure HOLD is always present.
            if "hold" not in feasible:
                feasible.append("hold")
            if action not in feasible:
                # re-pick best among feasible using the same sampled scores
                idxs = [ACTIONS.index(a) for a in feasible]
                best_val = float(np.max(sampled_scores[idxs]))
                cands = [a for a in feasible if float(sampled_scores[ACTIONS.index(a)]) == best_val]
                action = str(np.random.choice(cands))

        proposed_action = action
        self._proposed_action_counter[proposed_action] += 1
        # buy_margin/sell_margin already computed above from sampled_scores
        edge_margin = buy_margin if proposed_action == "buy" else sell_margin if proposed_action == "sell" else 0.0

        hold_reason: str | None = None
        gate_blocked = False
        timing_blocked = False
        budget_blocked = False

        # --- cost-aware gating -------------------------------------------------
        # In stuck mode, do not re-veto here (we already softened the threshold above).
        if config.COST_AWARE_GATING:
            est_cost = (config.FEE_RATE + config.SLIPPAGE_RATE + config.GATE_SAFETY_MARGIN)
            if stuck_relax:
                cost_edge = max(edge_threshold, est_cost + config.EDGE_SAFETY_MARGIN)
            else:
                cost_edge = max(edge_threshold, config.COST_EDGE_MULT * est_cost) + config.EDGE_SAFETY_MARGIN
        else:
            cost_edge = edge_threshold

        if (not warmup_active) and (not stuck_relax):
            if action == "buy" and buy_margin < cost_edge:
                action = "hold"
                hold_reason = "cost_gate"
                gate_blocked = True
                self.gate_blocks += 1
                self._gate_reason_counter["cost_gate"] += 1
            elif action == "sell":
                # Do not cost-gate exits. SELL reduces exposure and is required for recovery from
                # bad entries; cost-gating SELL creates permanent HOLD regimes in live 5m runs.
                # Execution frictions are still applied in portfolio PnL and learning reward.
                pass

        # --- adaptive cooldown (regime + turnover aware) ----------------------
        # NOTE: eff_gap/eff_hold and gap_ok/hold_ok are computed earlier (before act())
        # so the agent only sees timing-feasible actions. We keep using the computed
        # gap_ok/hold_ok here for the final enforcement step below.

        if (not warmup_active) and (not stuck_relax):
            # Cooldown gating: enforce minimum spacing between trades.
            # We allow a *very conservative* bypass only for strong, risk-reducing exits,
            # and even then we keep a small minimum gap to prevent churn spirals.
            # With adaptive cooldown active, a separate "bypass" is usually unnecessary and risky.
            # However, it can prevent pathological HOLD-freeze when the effective gap gets small
            # (high-vol regime) but a conservative strong-min-gap accidentally makes bypass impossible.
            since_last_trade = (self.steps - self.last_trade_step) if self.last_trade_step >= 0 else 10**9
            action_margin = buy_margin if action == "buy" else sell_margin if action == "sell" else 0.0
            strong_enough = action_margin >= (config.COOLDOWN_STRONG_EDGE_MULT * cost_edge)
            bypass_allowed = (action == "sell") if config.COOLDOWN_BYPASS_SELL_ONLY else (action in ("buy", "sell"))
            # Strong-min-gap is capped to the effective gap so bypass can never be made impossible
            # by config drift (a common source of "always HOLD").
            strong_min_gap = int(getattr(config, 'COOLDOWN_STRONG_MIN_GAP_STEPS', 0))
            strong_min_gap = min(strong_min_gap, max(eff_gap, int(getattr(config, "COOLDOWN_MIN_GAP_FLOOR", 0))))
            min_gap_ok = since_last_trade >= strong_min_gap
            strong_cooldown_ok = strong_enough and bypass_allowed and min_gap_ok
            if action == "buy" and (not gap_ok) and (not strong_cooldown_ok):
                action = "hold"
                hold_reason = hold_reason or "cooldown_gap"
                timing_blocked = True
                self.timing_blocks += 1
            # Exits should not be trapped by a minimum-hold timer; only enforce trade-gap spacing.
            if action == "sell" and (not gap_ok) and (not strong_cooldown_ok):
                action = "hold"
                hold_reason = hold_reason or "cooldown_gap"
                timing_blocked = True
                self.timing_blocks += 1

        # --- action hysteresis ------------------------------------------------
        # Reduce flip-flopping between BUY/SELL. We only allow a direction flip
        # after it has been proposed consistently for a few steps, unless the
        # edge is very strong (edge >= HYSTERESIS_ALLOW_IF_EDGE_MULT * cost_edge).
        if (
            config.ENABLE_ACTION_HYSTERESIS
            and (not warmup_active)
            and (not stuck_relax)
            and proposed_action in ("buy", "sell")
            and self._last_direction in ("buy", "sell")
            and proposed_action != self._last_direction
        ):
            # Very-strong-edge override: allow immediate flip
            strong_ok = edge_margin >= (config.HYSTERESIS_ALLOW_IF_EDGE_MULT * cost_edge)
            if not strong_ok:
                if self._flip_candidate != proposed_action:
                    self._flip_candidate = proposed_action
                    self._flip_streak = 1
                else:
                    self._flip_streak += 1
                if self._flip_streak < int(config.HYSTERESIS_REQUIRED_STREAK):
                    action = "hold"
                    hold_reason = hold_reason or "hysteresis"
                    timing_blocked = True
                    self.timing_blocks += 1
                else:
                    # flip allowed; reset streak
                    self._flip_candidate = None
                    self._flip_streak = 0
            else:
                # strong edge: allow immediate flip and reset
                self._flip_candidate = None
                self._flip_streak = 0
        elif proposed_action in ("buy", "sell") and proposed_action == self._last_direction:
            # reaffirmation resets flip tracking
            self._flip_candidate = None
            self._flip_streak = 0

        # --- multi-timeframe confirmation ------------------------------------
        # Lightweight regime filter using 5m rolling percentiles computed in indicators.
        # Avoid trading in low-vol chop and during high-spread microstructure stress.
        if (
            config.ENABLE_MTF_CONFIRMATION
            and (not warmup_active)
            and (not stuck_relax)
            and action in ("buy", "sell")
        ):
            try:
                vol_pct = float(row.get("rv1m_pct_5m", np.nan))
                spr_pct = float(row.get("spread_pct_5m", np.nan))
            except Exception:
                vol_pct, spr_pct = float('nan'), float('nan')
            if np.isfinite(vol_pct) and np.isfinite(spr_pct):
                vol_pct = self._sanitize_unit_percentile(vol_pct, col="rv1m_pct_5m")
                spr_pct = self._sanitize_unit_percentile(spr_pct, col="spread_pct_5m")
                if action == "buy":
                    # BUY: avoid low-vol chop and high-spread microstructure, but do not hard-freeze the system.
                    # Extreme spread is a hard block; moderate spread tightens the edge requirement.
                    hard = float(getattr(config, "MTF_SPREAD_PCT_HARD", 1.0))
                    soft = float(getattr(config, "MTF_SPREAD_PCT_MAX", 1.0))
                    if spr_pct > hard:
                        action = "hold"
                        hold_reason = hold_reason or "mtf_buy_spread_hard"
                        gate_blocked = True
                        self.gate_blocks += 1
                        self._gate_reason_counter["mtf_buy_spread_hard"] += 1
                    else:
                        # Start from the low-vol multiplier, then optionally tighten further for high spread.
                        req_mult = 1.0
                        lowvol = vol_pct < float(getattr(config, "MTF_VOL_PCT_MIN", 0.0))
                        if lowvol:
                            req_mult = max(req_mult, float(getattr(config, "MTF_LOWVOL_EDGE_MULT", 1.0)))
                        if spr_pct > soft:
                            req_mult = max(req_mult, float(getattr(config, "MTF_HIGHSPREAD_EDGE_MULT", 1.0)))
                        if req_mult > 1.0 and buy_margin < (req_mult * cost_edge):
                            action = "hold"
                            # Distinguish which condition tightened the requirement for better tuning.
                            if spr_pct > soft and lowvol:
                                reason = "mtf_buy_lowvol_highspread"
                            elif spr_pct > soft:
                                reason = "mtf_buy_highspread"
                            else:
                                reason = "mtf_buy_lowvol"
                            hold_reason = hold_reason or reason
                            gate_blocked = True
                            self.gate_blocks += 1
                            self._gate_reason_counter[reason] += 1
                else:
                    # SELL: do NOT trap exits in calm regimes. Only block if spreads are extreme
                    # and the sell edge is weak (i.e., not clearly worth crossing the spread).
                    if (spr_pct > float(config.MTF_SPREAD_PCT_MAX)) and (sell_margin < (2.0 * cost_edge)):
                        action = "hold"
                        hold_reason = hold_reason or "mtf_sell"
                        gate_blocked = True
                        self.gate_blocks += 1
                        self._gate_reason_counter["mtf_sell"] += 1

        # --- turnover hard blocking -----------------------------------------
        # If we've already exceeded a hard turnover budget in the recent window,
        # stop opening new exposure. Allow exits only when the sell edge is strong
        # enough to justify the additional friction.
        turnover_budget = self.initial_cash * self.turnover_budget_multiplier
        turnover_now = float(sum(self._turnover_window))
        turnover_stressed = turnover_now > (turnover_budget * float(getattr(config, 'TURNOVER_HARD_BLOCK_MULT', 1.0)))
        # Turnover budget is meant to stop *new exposure* (BUY) when the strategy is
        # churning fees. It must NOT block exits; otherwise you get permanent HOLD regimes.
        if (not warmup_active) and turnover_stressed and action == "buy":
            action = "hold"
            hold_reason = hold_reason or "turnover_budget"
            budget_blocked = True
            self.budget_blocks += 1

        # --- trade-rate throttling (anti-churn) -------------------------------
        # Count executed trade *legs* over a rolling step window and stop opening
        # new exposure when the policy is churning. This complements cooldowns:
        # cooldowns prevent immediate flip-flops; this prevents slow fee bleed.
        tr_window = int(getattr(config, 'TRADE_RATE_WINDOW_STEPS', 0) or 0)
        tr_max = int(getattr(config, 'MAX_TRADES_PER_WINDOW', 0) or 0)
        # Trade-rate throttling prevents churn on entries. It must NOT block exits; otherwise
        # long runs can freeze into HOLD forever after hitting the trade limit once.
        if (not warmup_active) and (not stuck_relax) and tr_window > 0 and tr_max > 0 and action == "buy":
            recent_trades = int(sum(getattr(self, '_trade_count_window', [])))
            if recent_trades >= tr_max:
                action = "hold"
                hold_reason = hold_reason or "trade_rate"
                budget_blocked = True
                self.budget_blocks += 1

        trade_executed = False
        fee_paid = 0.0
        turnover_penalty = 0.0
        slippage_paid = 0.0
        realized_pnl = 0.0
        notional_traded = 0.0
        trade_size = 0.0

        value_before = self.portfolio.value(price_now)
        position_before = self.portfolio.position
        cash_before = self.portfolio.cash
        # Will be updated after execution; used for unbiased reward calculation.
        value_after = value_before

        if action == "hold" and stuck_relax and proposed_action in allowed_actions:
            # When the agent is clearly stuck in HOLD, allow a *limited* execution of the
            # originally proposed action (buy/sell) to keep exploration alive.
            # IMPORTANT: do NOT bypass cost gating; only bypass the *edge confidence* gate
            # (and still respect timing locks) so this doesn't become a churn/cost bypass.
            if config.COST_AWARE_GATING:
                # IMPORTANT: TURNOVER_PENALTY is a *learning regularizer*, not an execution cost.
                # Do not include it in cost-aware gating, otherwise the strategy gets over-conservative
                # and HOLD-freezes (especially on 1m).
                est_cost = (
                    config.FEE_RATE
                    + config.SLIPPAGE_RATE
                    + config.GATE_SAFETY_MARGIN
                )
                # In stuck mode we require the edge to beat *at least* raw estimated costs
                # (without the usual multiplier), but still respect any edge_threshold floor.
                stuck_cost_edge = max(edge_threshold, est_cost + config.EDGE_SAFETY_MARGIN)
            else:
                stuck_cost_edge = edge_threshold
            if proposed_action == "buy" and cash_before > 0 and (gap_ok or stuck_gap_ok) and buy_margin >= stuck_cost_edge:
                action = "buy"
                hold_reason = None
                gate_blocked = False
            elif (
                proposed_action == "sell"
                and position_before > 0
                and (gap_ok or stuck_gap_ok)
                and sell_margin >= stuck_cost_edge
            ):
                action = "sell"
                hold_reason = None
                gate_blocked = False

        # --- execution ---------------------------------------------------------
        # Professional separation of frictions:
        # - fee_paid + slippage_paid are *execution costs* (affect portfolio value)
        # - turnover_penalty is a *learning regularizer* (does NOT affect portfolio value; applied in trainer_reward)
        if action == "buy" and cash_before > 0:
            pos_frac = self._dynamic_fraction(buy_margin)
            trade_cash = cash_before * pos_frac
            # Avoid micro-buys that are almost entirely fees/slippage on 1m data.
            min_notional = float(getattr(config, 'MIN_TRADE_NOTIONAL', 0.0) or 0.0)
            if min_notional > 0.0 and trade_cash < min_notional:
                action = "hold"
                hold_reason = hold_reason or "min_notional"
                budget_blocked = True
                trade_cash = 0.0

            # penalties applied ONCE by shrinking shares (keep cash outlay = trade_cash)
            fee_paid = trade_cash * config.FEE_RATE
            slippage_paid = trade_cash * config.SLIPPAGE_RATE
            # NOTE: turnover_penalty is a learning regularizer (applied later in trainer_reward),
            # not an execution cash cost.
            investable = trade_cash - fee_paid - slippage_paid
            cash_outlay = trade_cash  # <-- no "+ turnover_penalty" here

            if investable > 0 and cash_outlay <= cash_before:
                trade_executed = True
                notional_traded = trade_cash  # for turnover stats
                trade_size = investable / price_now

                prior_pos = self.portfolio.position
                prior_cost = prior_pos * self.portfolio.entry_price
                new_pos = prior_pos + trade_size

                # basis tracks actual cash spent this leg (trade_cash)
                total_cost = prior_cost + cash_outlay
                self.portfolio.entry_price = total_cost / max(new_pos, 1e-9)

                self.portfolio.position = new_pos
                self.portfolio.cash = cash_before - cash_outlay

                if prior_pos == 0:
                    self.portfolio.entry_value = value_before
                    self._buy_legs_current = 1
                else:
                    self._buy_legs_current += 1

                self.last_trade_step = self.steps
                self.last_entry_step = self.steps

                # DEBUG (optional): verify the immediate MTM impact of this BUY
                # Why: catches any hidden re-charges or double counting early.
                trade_impact = self.portfolio.value(price_now) - value_before
                expected_impact = -(fee_paid + slippage_paid)
                if abs(trade_impact - expected_impact) > 1e-6:
                    print(
                        f"[warn] buy impact mismatch: got {trade_impact:.6f}, expected {expected_impact:.6f}"
                    )
            else:
                action = "hold"
                hold_reason = hold_reason or "budget"
                budget_blocked = True
                fee_paid = 0.0
                turnover_penalty = 0.0
                slippage_paid = 0.0

        elif action == "sell" and position_before > 0:
            sell_frac = self._dynamic_fraction(sell_margin) if config.PARTIAL_SELLS else 1.0
            # If the sell signal is very strong, prefer a clean exit to avoid
            # hundreds of tiny partial sells that keep the position "alive" and block re-entries.
            if sell_margin >= (config.COOLDOWN_STRONG_EDGE_MULT * cost_edge):
                sell_frac = 1.0
            trade_size = position_before * min(1.0, max(0.0, sell_frac))
            # Dust handling: if we'd leave a tiny residual position, just close it.
            remaining_pos = position_before - trade_size
            if remaining_pos > 0 and (remaining_pos * price_now) < float(config.DUST_POSITION_NOTIONAL):
                trade_size = position_before
            if trade_size > 0:
                trade_executed = True
                # stash entry price BEFORE mutating portfolio (needed for per-leg win-rate)
                entry_price_for_leg = self.portfolio.entry_price
                gross_proceeds = trade_size * price_now
                # Avoid micro-sells that are dominated by friction unless we're closing a dust position.
                min_notional = float(getattr(config, 'MIN_TRADE_NOTIONAL', 0.0) or 0.0)
                if min_notional > 0.0 and gross_proceeds < min_notional:
                    full_notional = position_before * price_now
                    # If the whole position is tiny (dust-ish), just close it; otherwise skip the micro-sell.
                    if full_notional <= max(min_notional, float(getattr(config, 'DUST_POSITION_NOTIONAL', 0.0) or 0.0)):
                        trade_size = position_before
                        gross_proceeds = trade_size * price_now
                    else:
                        trade_executed = False
                        action = "hold"
                        hold_reason = hold_reason or "min_notional"
                        budget_blocked = True
                        fee_paid = 0.0
                        slippage_paid = 0.0
                        notional_traded = 0.0
                        trade_size = 0.0
                if action == "sell":
                    fee_paid = gross_proceeds * config.FEE_RATE
                slippage_paid = gross_proceeds * config.SLIPPAGE_RATE
                # NOTE: turnover_penalty is a learning regularizer (applied later in trainer_reward),
                # not an execution cash cost.
                notional_traded = gross_proceeds
                net = gross_proceeds - fee_paid - slippage_paid
                self.portfolio.cash += net
                self.portfolio.position = position_before - trade_size
                fully_closed = self.portfolio.position <= 1e-12
                if fully_closed:
                    realized_pnl = self.portfolio.cash - self.portfolio.entry_value
                    self.portfolio.entry_price = 0.0
                    self.portfolio.entry_value = 0.0
                    self.last_entry_step = -1
                    self._buy_legs_current = 0
                else:
                    # keep entry_value; basis unchanged for remaining units
                    pass
                self.last_trade_step = self.steps
            else:
                action = "hold"
                hold_reason = hold_reason or "budget"
                budget_blocked = True

        if budget_blocked:
            self.budget_blocks += 1

        if hold_reason:
            self._hold_reason_log.append((self.steps, proposed_action, hold_reason))
            self._hold_reason_counter[hold_reason] += 1

        # --- reward & penalties ------------------------------------------------
        # IMPORTANT ALIGNMENT NOTE:
        # We track portfolio performance in two places:
        #   - total_return / equity_curve (includes execution frictions immediately)
        #   - sharpe_ratio from _return_history
        # Historically, _return_history used (value_next - value_after), which *excluded* the
        # immediate fee/slippage hit at the execution step. That can produce a confusing
        # situation where Sharpe looks great while total_return is deeply negative (fee-death-by-churn).
        #
        # To align learning + reporting, we define the step reward as the change from the
        # *pre-trade* mark-to-market value at price_now to the next mark-to-market value.
        # This includes execution frictions in the same step they occur and makes Sharpe and
        # total_return consistent diagnostics.
        value_after = self.portfolio.value(price_now)
        value_next = self.portfolio.value(price_next)
        self._log_action(action)
        # pre-trade -> next-step value change (includes fee/slippage impact when trade happens)
        reward = value_next - value_before

        # --- directional shaping (flat-state curriculum signal) ---------------
        # When flat, portfolio PnL is often exactly 0, which can slow learning of entry timing.
        # Add a *small*, clipped term based on the next-bar return when FLAT only.
        # This never affects portfolio value; it only shapes the learning signal.
        if getattr(config, 'ENABLE_DIRECTIONAL_SHAPING', False) and position_before <= 0 and cash_before > 0:
            try:
                next_ret = (price_next / max(price_now, 1e-9)) - 1.0
            except Exception:
                next_ret = 0.0
            w = float(getattr(config, 'DIRECTIONAL_SHAPING_WEIGHT', 0.0))
            miss_w = float(getattr(config, 'FLAT_HOLD_MISS_PENALTY_WEIGHT', 0.0))
            clip_frac = float(getattr(config, 'DIRECTIONAL_SHAPING_CLIP', 0.0))
            clip_val = max(0.0, clip_frac) * self.initial_cash
            shaped = 0.0
            if action == 'buy':
                shaped = w * self.initial_cash * float(next_ret)
            elif action == 'hold':
                shaped = -miss_w * self.initial_cash * abs(float(next_ret))
            if clip_val > 0.0:
                shaped = _clamp(shaped, -clip_val, clip_val)
            reward += shaped

        step_return = reward / max(value_before, 1e-6)
        self._return_history.append(step_return)
        self._turnover_window.append(notional_traded)
        # Trade-rate window: track executed trade legs to throttle slow churn on 1m.
        if hasattr(self, '_trade_count_window') and getattr(self._trade_count_window, 'maxlen', 0):
            self._trade_count_window.append(1 if trade_executed else 0)
        self.total_notional_traded += float(notional_traded)
        self._update_adaptive_risk_controls()

        prospective_curve = self._equity_curve + [value_next]
        drawdown = compute_max_drawdown(prospective_curve)
        drawdown_over = max(0.0, drawdown - self.drawdown_budget)
        penalties = config.PENALTY_PROFILES.get(
            self.penalty_profile, config.PENALTY_PROFILES.get("train", {})
        )
        dd_penalty_rate = penalties.get("drawdown_penalty", config.DRAWDOWN_PENALTY)
        to_penalty_rate = penalties.get("turnover_budget_penalty", config.TURNOVER_BUDGET_PENALTY)
        dd_penalty_value = drawdown_over * self.initial_cash * dd_penalty_rate

        turnover_budget = self.initial_cash * self.turnover_budget_multiplier
        turnover_over = max(0.0, sum(self._turnover_window) - turnover_budget)
        to_penalty_value = (turnover_over / max(turnover_budget, 1e-6)) * self.initial_cash * to_penalty_rate

        # Turnover penalty: apply a *per-trade* learning regularizer based on this step's notional.
        # This is separate from the turnover budget penalty above (which activates only when the
        # rolling window exceeds the budget).
        turnover_penalty = float(notional_traded) * float(getattr(config, "TURNOVER_PENALTY", 0.0))

        # Penalize infeasible proposals: agent proposed BUY/SELL but guards converted to HOLD.
        blocked_penalty = 0.0
        if proposed_action in ("buy", "sell") and action == "hold":
            if gate_blocked:
                blocked_penalty += float(getattr(config, "BLOCKED_GATE_PENALTY", 0.0)) * self.initial_cash
            if timing_blocked:
                blocked_penalty += float(getattr(config, "BLOCKED_TIMING_PENALTY", 0.0)) * self.initial_cash
            if budget_blocked:
                blocked_penalty += float(getattr(config, "BLOCKED_BUDGET_PENALTY", 0.0)) * self.initial_cash

        risk_penalty_value = dd_penalty_value + to_penalty_value + turnover_penalty + blocked_penalty

        trainer_reward = reward - risk_penalty_value

        # ---- Reward shaping (bounded + variance-reduced) ---------------------
        # 1) convert to pct-of-initial-cash (stable scale)
        pct_raw = trainer_reward / max(self.initial_cash, 1e-6)

        # Optional: blend raw reward with a volatility-normalised component.
        # This makes REWARD_RISK_BLEND / REWARD_STABILITY_* config knobs real,
        # and stabilises learning across regime shifts without changing the sign
        # of the learning signal in normal conditions.
        pct = pct_raw
        blend = float(getattr(config, "REWARD_RISK_BLEND", 0.0))
        if blend > 0.0:
            window = int(getattr(config, "REWARD_STABILITY_WINDOW", 0))
            min_obs = int(getattr(config, "REWARD_STABILITY_MIN_OBS", 0))
            if window > 1 and len(self._return_history) >= max(min_obs, 2):
                recent = list(self._return_history)[-window:]
                vol = float(np.std(np.asarray(recent, dtype=float)))
                target = max(float(getattr(config, "ADAPTIVE_TARGET_RETURN_VOL", 1e-6)), 1e-6)
                denom = max(vol, target, 1e-6)
                pct_volnorm = pct_raw * (target / denom)
                b = max(0.0, min(1.0, blend))
                pct = (1.0 - b) * pct_raw + b * pct_volnorm

        # 2) advantage baseline (EMA) to reduce variance; does not change optimum
        if config.USE_ADVANTAGE_BASELINE:
            decay = _clamp(config.BASELINE_EMA_DECAY, 0.0, 1.0)
            self._reward_baseline = (1 - decay) * self._reward_baseline + decay * pct
        advantage = pct - (self._reward_baseline if config.USE_ADVANTAGE_BASELINE else 0.0)

        # 3) normalize by an EMA of advantage variance (stabilizes posterior updates)
        if config.USE_REWARD_STD_NORMALIZATION:
            # Use a dedicated decay for the variance estimator; defaults to BASELINE_EMA_DECAY
            # so existing behaviour remains unchanged unless you tune it explicitly.
            decay = _clamp(float(getattr(config, "REWARD_VAR_EMA_DECAY", config.BASELINE_EMA_DECAY)), 0.0, 1.0)
            self._reward_var_ema = (1 - decay) * self._reward_var_ema + decay * (advantage * advantage)
            sigma = math.sqrt(max(self._reward_var_ema, config.REWARD_SIGMA_FLOOR ** 2))
            advantage = advantage / sigma

        # 4) bound the learning signal to [-1,1] via tanh, with input clipping
        tanh_in = advantage * self.reward_scale
        tanh_in = _clamp(tanh_in, -config.REWARD_TANH_INPUT_CLIP, config.REWARD_TANH_INPUT_CLIP)

        # 5) optional probabilistic shaping noise (zero-mean, decays with steps)
        if config.ENABLE_PROB_SHAPING and self.reward_scale > 0:
            # decay alpha/beta counts slowly to avoid locking-in early noise patterns
            cdec = _clamp(config.PROB_SHAPING_COUNT_DECAY, 0.0, 1.0)
            self._prob_alpha = (1 - cdec) * self._prob_alpha + cdec * config.PROB_SHAPING_ALPHA0
            self._prob_beta = (1 - cdec) * self._prob_beta + cdec * config.PROB_SHAPING_BETA0
            # update counts with the sign of the *true* trainer reward (not action-dependent)
            if trainer_reward > 0:
                self._prob_alpha = min(self._prob_alpha + 1.0, config.PROB_SHAPING_MAX_COUNT)
            elif trainer_reward < 0:
                self._prob_beta = min(self._prob_beta + 1.0, config.PROB_SHAPING_MAX_COUNT)
            # sample exploration noise and re-center to zero mean
            a = max(self._prob_alpha, 1e-6)
            b = max(self._prob_beta, 1e-6)
            p = float(np.random.beta(a, b))
            centered = p - (a / (a + b))
            # step-based decay so shaping fades out
            hl = max(1.0, float(config.PROB_SHAPING_HALF_LIFE_STEPS))
            amp = config.PROB_SHAPING_TANH_AMPLITUDE * (0.5 ** (self.steps / hl))
            tanh_in = tanh_in + _clamp(centered, -0.5, 0.5) * 2.0 * amp

        scaled_reward = math.tanh(tanh_in)

        # --- learning ---------------------------------------------------------
        next_features = build_features(next_row, self.portfolio)
        next_allowed = ["hold"]
        if self.portfolio.position > 0:
            next_allowed.append("sell")
        if self.portfolio.position <= 0 and self.portfolio.cash > 0:
            next_allowed.append("buy")

        if train:
            self.agent.update(
                action,
                scaled_reward,
                features,
                actual_reward=reward,
                trade_executed=trade_executed,
                next_features=next_features,
                allowed_next=next_allowed,
            )
            self.history.append((step_idx, action, price_now, reward))
        elif trade_executed:
            self.agent.state.trades += 1

        self.total_steps += 1
        if trainer_reward > 0:
            self.positive_steps += 1
        self.steps += 1
        self.total_fee_paid += fee_paid
        self.total_turnover_penalty_paid += turnover_penalty
        self.total_slippage_paid += slippage_paid
        self._equity_curve.append(value_next)

        # execution counters and per-leg win-rate accounting
        if trade_executed and action in ("buy", "sell"):
            self._last_direction = action
            self._flip_candidate = None
            self._flip_streak = 0

        # execution counters and per-leg win-rate accounting
        if trade_executed:
            self.executed_trade_count += 1
            if action == "buy":
                self.buy_legs += 1
            elif action == "sell":
                self.sell_legs += 1
                if forced_exit:
                    self.forced_exit_count += 1
                    if forced_exit_reason:
                        self._forced_exit_reason_counter[forced_exit_reason] += 1
                # per-leg PnL: use entry price BEFORE the sell (especially important on full closes)
                # We stash it during execution in entry_price_for_leg.
                entry_price_for_leg = locals().get("entry_price_for_leg", self.portfolio.entry_price)
                leg_qty = notional_traded / max(price_now, 1e-9)
                # entry_price_for_leg includes buy-side frictions via basis; subtract sell frictions for net leg PnL
                leg_pnl_net = (price_now - entry_price_for_leg) * leg_qty - fee_paid - slippage_paid
                if leg_pnl_net > 0:
                    self.winning_sell_legs += 1
                    self._win_pnl_sum += float(leg_pnl_net)
                    self._win_pnl_count += 1
                elif leg_pnl_net < 0:
                    self._loss_pnl_sum += float(leg_pnl_net)
                    self._loss_pnl_count += 1

        # legacy full-close sell win rate (kept for backward compatibility)
        if action == "sell" and trade_executed:
            self.sell_trades += 1
            if realized_pnl > 0:
                self.winning_sells += 1

        self._last_price = price_now
        self._last_value = self.portfolio.value(price_now)

        return StepResult(
            action=action,
            trainer_reward=trainer_reward,
            scaled_reward=scaled_reward,
            trade_executed=trade_executed,
            trade_size=trade_size,
            notional_traded=notional_traded,
            fee_paid=fee_paid,
            turnover_penalty=turnover_penalty,
            slippage_paid=slippage_paid,
            refilled=refilled,
            realized_pnl=realized_pnl,
            proposed_action=proposed_action,
            edge_margin=edge_margin,
            hold_reason=hold_reason,
            gate_blocked=gate_blocked,
            timing_blocked=timing_blocked,
            budget_blocked=budget_blocked,
            stuck_relax=stuck_relax,
        )

    # NOTE: run(), _flush_trades_and_metrics(), _persist_trades(), _persist_metrics(), _save_trainer_state()
    # remain identical to your current version.

    # --- persistence ---------------------------------------------------------

    def _persist_trades(self, run_dir: Path, data_is_live: bool | None = None) -> None:
        path = run_dir / "trades.csv"
        new_trades = self.history[self._last_flushed_trade_idx :]
        if not new_trades:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        header = ["step", "action", "price", "reward", "data_is_live"]
        write_header = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            for step_idx, action, price, reward in new_trades:
                writer.writerow([step_idx, action, price, reward, data_is_live])

    def _persist_metrics(
        self,
        run_dir: Path,
        *,
        data_is_live: bool | None = None,
        baseline_final_value: float | None = None,
        val_final_value: float | None = None,
        max_drawdown: float | None = None,
        executed_trades: int | None = None,
        ma_baseline_final_value: float | None = None,
    ) -> None:
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_steps": self.total_steps,
            "success_rate": self.success_rate,
            "trade_win_rate": self.trade_win_rate,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": max_drawdown if max_drawdown is not None else self.max_drawdown,
            "total_return": self.total_return,
            "total_fee_paid": self.total_fee_paid,
            "turnover_penalty_paid": self.total_turnover_penalty_paid,
            "slippage_paid": getattr(self, 'total_slippage_paid', 0.0),
            "gate_blocks": self.gate_blocks,
            "timing_blocks": self.timing_blocks,
            "budget_blocks": self.budget_blocks,
            "data_is_live": data_is_live,
            "baseline_final_value": baseline_final_value,
            "val_final_value": val_final_value,
            "ma_baseline_final_value": ma_baseline_final_value,
            # decision_count is total_steps; executed_trade_count counts only actual fills
            "decision_count": self.total_steps,
            "executed_trade_count": int(getattr(self, 'executed_trade_count', 0)),
            "buy_legs": int(getattr(self, 'buy_legs', 0)),
            "sell_legs": int(getattr(self, 'sell_legs', 0)),
            "winning_sell_legs": int(getattr(self, 'winning_sell_legs', 0)),
            "avg_win_pnl": float(getattr(self, 'avg_win_pnl', 0.0)),
            "avg_loss_pnl": float(getattr(self, 'avg_loss_pnl', 0.0)),
            "win_loss_ratio": float(getattr(self, 'win_loss_ratio', 0.0)),
            "expectancy_pnl_per_sell_leg": float(getattr(self, 'expectancy_pnl_per_sell_leg', 0.0)),
            "avg_notional_per_trade": (float(getattr(self, 'total_notional_traded', 0.0)) / max(1, int(getattr(self, 'executed_trade_count', 0)))) if int(getattr(self, 'executed_trade_count', 0)) > 0 else 0.0,
            "turnover_per_1000_steps": (sum(self._turnover_window) / max(1, self.total_steps)) * 1000.0,
            # legacy field name kept (now uses executed_trade_count by default)
            "executed_trades": executed_trades if executed_trades is not None else int(getattr(self, 'executed_trade_count', 0)),
            "reward_scale": self.reward_scale,
            "drawdown_budget": self.drawdown_budget,
            "turnover_budget_multiplier": self.turnover_budget_multiplier,
            "action_distribution": self.action_distribution,
            "proposed_action_distribution": (
                {} if sum(self._proposed_action_counter.values()) <= 0
                else {k: v / sum(self._proposed_action_counter.values()) for k, v in self._proposed_action_counter.items()}
            ),
            "hold_reason_counts": dict(self._hold_reason_counter),
            "gate_reason_counts": dict(self._gate_reason_counter),
            "premask_gate_reason_counts": dict(self._premask_gate_reason_counter),
        }
        path = run_dir / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: list[dict] = []
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except json.JSONDecodeError:
                existing = []
        existing.append(metrics)
        atomic_write_json(path, existing)

    def _save_trainer_state(
        self,
        run_dir: Path,
        run_id: str,
        *,
        checkpoint: bool = False,
        keep_last: int = 5,
    ) -> Path:
        state = TrainerState(
            run_id=run_id,
            steps=self.steps,
            prev_price=self._last_price,
            prev_value=self._last_value,
            refill_count=self.refill_count,
            total_fee_paid=self.total_fee_paid,
            total_turnover_penalty_paid=self.total_turnover_penalty_paid,
            total_slippage_paid=getattr(self, 'total_slippage_paid', 0.0),
            total_notional_traded=float(getattr(self, 'total_notional_traded', 0.0)),
            executed_trade_count=int(getattr(self, 'executed_trade_count', 0)),
            buy_legs=int(getattr(self, 'buy_legs', 0)),
            sell_legs=int(getattr(self, 'sell_legs', 0)),
            winning_sell_legs=int(getattr(self, 'winning_sell_legs', 0)),
            win_pnl_sum=float(getattr(self, '_win_pnl_sum', 0.0)),
            loss_pnl_sum=float(getattr(self, '_loss_pnl_sum', 0.0)),
            win_pnl_count=int(getattr(self, '_win_pnl_count', 0)),
            loss_pnl_count=int(getattr(self, '_loss_pnl_count', 0)),
            # legacy fields
            total_trades=len(self.history),
            successful_trades=self.winning_sells,
            sell_trades=self.sell_trades,
            winning_sells=self.winning_sells,
            portfolio_cash=self.portfolio.cash,
            portfolio_position=self.portfolio.position,
            portfolio_entry_price=self.portfolio.entry_price,
            portfolio_entry_value=self.portfolio.entry_value,
            total_steps=self.total_steps,
            positive_steps=self.positive_steps,
            last_trade_step=self.last_trade_step,
            last_entry_step=self.last_entry_step,
            last_direction=self._last_direction,
            flip_candidate=self._flip_candidate,
            flip_streak=int(self._flip_streak),
            gate_blocks=self.gate_blocks,
            timing_blocks=self.timing_blocks,
            budget_blocks=self.budget_blocks,
            penalty_profile=self.penalty_profile,
            reward_scale=self.reward_scale,
            drawdown_budget=self.drawdown_budget,
            turnover_budget_multiplier=self.turnover_budget_multiplier,
            # metrics helpers
            # (not used for execution; purely for reporting)
            # total_notional_traded is cumulative across the run
            
            reward_baseline_ema=self._reward_baseline,
            reward_var_ema=self._reward_var_ema,
            prob_alpha=self._prob_alpha,
            prob_beta=self._prob_beta,
        )

        target_dir = Path(run_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        latest = target_dir / "trainer_state_latest.json"
        state.to_json(latest)

        saved = latest
        if checkpoint:
            ck = target_dir / f"trainer_state_step_{self.steps}.json"
            state.to_json(ck)
            ckpts = sorted(
                [p for p in target_dir.glob("trainer_state_step_*.json")],
                key=lambda p: int(p.stem.split("_")[-1]),
            )
            keep_last = max(0, keep_last)
            for old in ckpts[:-keep_last]:
                try:
                    old.unlink()
                except OSError:
                    pass
            saved = ck
        return saved

    def _flush_trades_and_metrics(
        self,
        run_dir: Path,
        *,
        force: bool = False,
        data_is_live: bool | None = None,
        baseline_final_value: float | None = None,
        val_final_value: float | None = None,
        max_drawdown: float | None = None,
        executed_trades: int | None = None,
        ma_baseline_final_value: float | None = None,
    ) -> None:
        new_trades = self.history[self._last_flushed_trade_idx :]
        if new_trades or force:
            self._persist_trades(run_dir, data_is_live=data_is_live)
            self._persist_metrics(
                run_dir,
                data_is_live=data_is_live,
                baseline_final_value=baseline_final_value,
                val_final_value=val_final_value,
                max_drawdown=max_drawdown,
                executed_trades=executed_trades,
                ma_baseline_final_value=ma_baseline_final_value,
            )
            self._last_flushed_trade_idx = len(self.history)

    # --- execution ------------------------------------------------------------

    def run(
        self,
        frame: pd.DataFrame,
        *,
        max_steps: int | None = None,
        run_id: str,
        run_dir: Path,
        checkpoint_every: int,
        flush_trades_every: int,
        keep_last: int,
        data_is_live: bool = False,
    ) -> None:
        if frame.empty:
            return

        limit = max_steps if max_steps is not None else len(frame) - 1
        limit = min(limit, len(frame) - 1)

        self.last_data_is_live = data_is_live
        first_price = float(frame.iloc[0]["close"])
        pv_prev_after = self.portfolio.value(first_price)

        for idx in range(limit):
            row = frame.iloc[idx]
            next_row = frame.iloc[idx + 1]
            price = float(row["close"])
            before_trade_value = self.portfolio.value(price)
            result = self.step(row, next_row, idx, train=True)
            after_trade_value = self.portfolio.value(price)
            trade_impact = after_trade_value - before_trade_value
            mtm_delta = after_trade_value - pv_prev_after
            pv_prev_after = after_trade_value

            if flush_trades_every > 0 and self.steps % flush_trades_every == 0:
                self._flush_trades_and_metrics(run_dir, data_is_live=data_is_live)
            if checkpoint_every > 0 and self.steps % checkpoint_every == 0:
                self.agent.save(run_dir=run_dir, checkpoint=True, keep_last=keep_last)
                self._save_trainer_state(run_dir, run_id, checkpoint=True, keep_last=keep_last)

            # track last trade timing
            if result.trade_executed:
                self.last_trade_step = idx
                if result.action == "buy":
                    self.last_entry_step = idx

            # NOTE: return history is updated inside step() using post-trade value_after/value_next.
            # Keeping a second update here would double-count and distort adaptive controls.

    # --- restore helpers ------------------------------------------------------


def resume_from(run_dir: Path, agent: BanditAgent, trainer: Trainer) -> None:
    """Restore agent and trainer state from a previous run directory."""

    agent_state_path = Path(run_dir) / "agent_state_latest.json"
    if agent_state_path.exists():
        agent.state = AgentState.from_json(agent_state_path)
        agent._prepare_state()

    trainer_state_path = Path(run_dir) / "trainer_state_latest.json"
    if trainer_state_path.exists():
        state = TrainerState.from_json(trainer_state_path)
        trainer.steps = state.steps
        trainer.total_steps = state.total_steps
        trainer.positive_steps = state.positive_steps
        trainer.refill_count = state.refill_count
        trainer.total_fee_paid = state.total_fee_paid
        trainer.total_turnover_penalty_paid = state.total_turnover_penalty_paid
        trainer.total_slippage_paid = float(getattr(state, 'total_slippage_paid', 0.0))
        trainer.total_notional_traded = float(getattr(state, 'total_notional_traded', 0.0))
        trainer.executed_trade_count = int(getattr(state, 'executed_trade_count', 0))
        trainer.buy_legs = int(getattr(state, 'buy_legs', 0))
        trainer.sell_legs = int(getattr(state, 'sell_legs', 0))
        trainer.winning_sell_legs = int(getattr(state, 'winning_sell_legs', 0))
        trainer._win_pnl_sum = float(getattr(state, 'win_pnl_sum', 0.0))
        trainer._loss_pnl_sum = float(getattr(state, 'loss_pnl_sum', 0.0))
        trainer._win_pnl_count = int(getattr(state, 'win_pnl_count', 0))
        trainer._loss_pnl_count = int(getattr(state, 'loss_pnl_count', 0))
        trainer.sell_trades = state.sell_trades
        trainer.winning_sells = state.winning_sells
        trainer.last_trade_step = state.last_trade_step
        trainer.last_entry_step = state.last_entry_step
        trainer._last_direction = getattr(state, 'last_direction', None)
        trainer._flip_candidate = getattr(state, 'flip_candidate', None)
        trainer._flip_streak = int(getattr(state, 'flip_streak', 0) or 0)
        trainer.gate_blocks = state.gate_blocks
        trainer.timing_blocks = state.timing_blocks
        trainer.budget_blocks = state.budget_blocks
        trainer.penalty_profile = state.penalty_profile or trainer.penalty_profile
        trainer.reward_scale = state.reward_scale
        trainer.drawdown_budget = state.drawdown_budget
        trainer.turnover_budget_multiplier = state.turnover_budget_multiplier
        trainer.portfolio.cash = state.portfolio_cash
        trainer.portfolio.position = state.portfolio_position
        trainer.portfolio.entry_price = state.portfolio_entry_price
        trainer.portfolio.entry_value = state.portfolio_entry_value
        trainer._last_price = state.prev_price
        trainer._last_value = state.prev_value
        # reward shaping state
        trainer._reward_baseline = float(getattr(state, 'reward_baseline_ema', 0.0))
        trainer._reward_var_ema = float(getattr(state, 'reward_var_ema', config.REWARD_VAR_INIT))
        trainer._prob_alpha = float(getattr(state, 'prob_alpha', config.PROB_SHAPING_ALPHA0))
        trainer._prob_beta = float(getattr(state, 'prob_beta', config.PROB_SHAPING_BETA0))


