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


class EdgeGateModel:
    def __init__(self, size: int, *, ridge: float, decay: float) -> None:
        self.size = size
        self.ridge = max(float(ridge), 1e-12)
        self.decay = _clamp(float(decay), 0.0, 1.0)
        self.xtx = np.zeros((size, size), dtype=float)
        self.xty = np.zeros(size, dtype=float)
        self.weights = np.zeros(size, dtype=float)

    def predict(self, x: np.ndarray) -> float:
        return float(np.dot(self.weights, x))

    def update(self, x: np.ndarray, y: float) -> None:
        self.xtx = (self.decay * self.xtx) + np.outer(x, x)
        self.xty = (self.decay * self.xty) + (x * y)
        ridge_eye = np.eye(self.size, dtype=float) * self.ridge
        try:
            self.weights = np.linalg.solve(self.xtx + ridge_eye, self.xty)
        except np.linalg.LinAlgError:
            self.weights = np.zeros(self.size, dtype=float)


class EdgeGateClassifier:
    def __init__(self, size: int, *, lr: float, l2: float, decay: float) -> None:
        self.size = size
        self.lr = max(float(lr), 1e-6)
        self.l2 = max(float(l2), 0.0)
        self.decay = _clamp(float(decay), 0.0, 1.0)
        self.weights = np.zeros(size, dtype=float)

    def predict(self, x: np.ndarray) -> float:
        return float(np.dot(self.weights, x))

    def update(self, x: np.ndarray, y: float) -> None:
        y = float(y)
        if not np.isfinite(y):
            return
        self.weights *= self.decay
        z = float(np.dot(self.weights, x))
        pred = 1.0 / (1.0 + math.exp(-_clamp(z, -20.0, 20.0)))
        grad = (pred - y) * x + (self.l2 * self.weights)
        self.weights -= self.lr * grad

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
    version: int = 6
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
    last_stop_loss_step: int = -1
    trend_exit_dir: str | None = None
    trend_exit_streak: int = 0
    macro_lock_dir_candidate: str | None = None
    macro_lock_dir_streak: int = 0
    macro_lock_effective_dir: str = "neutral"
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
    unrealized_ret = 0.0
    if portfolio.entry_price > 0 and portfolio.position != 0:
        raw_ret = (price / portfolio.entry_price) - 1.0
        unrealized_ret = raw_ret if portfolio.position > 0 else -raw_ret
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
        self.history: List[
            Tuple[int, str, float, float, float, float, str, str, float, float, float, float]
        ] = []
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
        self.last_stop_loss_step = -1
        self._trend_exit_dir: str | None = None
        self._trend_exit_streak: int = 0
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
        # Stuck-unfreeze diagnostics: actions taken/proposed while stuck_relax is active.
        self._stuck_action_counter: Counter[str] = Counter()
        self._stuck_proposed_action_counter: Counter[str] = Counter()
        self._turnover_window: deque[float] = deque(maxlen=config.TURNOVER_BUDGET_WINDOW)
        # Separate anti-churn window: count executed trade legs (not notional)
        self._trade_count_window: deque[int] = deque(maxlen=int(getattr(config, 'TRADE_RATE_WINDOW_STEPS', 0) or 0))
        # cumulative notional traded (for stable metrics; windowed sums can mislead)
        self.total_notional_traded: float = 0.0
        # regime flip tracking for forced exits
        self._regime_flip_dir: str | None = None
        self._regime_flip_streak: int = 0
        # macro bias persistence
        self._macro_dir_candidate: str | None = None
        self._macro_dir_streak: int = 0
        self._macro_effective_dir: str = "neutral"
        # macro lock persistence
        self._macro_lock_dir_candidate: str | None = None
        self._macro_lock_dir_streak: int = 0
        self._macro_lock_effective_dir: str = "neutral"
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
        # per-position peak tracking for trailing take-profit
        self._pos_peak_price: float | None = None
        self._pos_trough_price: float | None = None
        # core + scalp EMA tracker
        self._scalp_ema: float | None = None
        # edge-gate predictor
        self._edge_gate: EdgeGateModel | EdgeGateClassifier | None = None
        self._edge_gate_steps: int = 0
        self._edge_gate_mode: str | None = None

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
        self._stuck_action_counter.clear()
        self._stuck_proposed_action_counter.clear()
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
        self.last_stop_loss_step = -1
        self._regime_flip_dir = None
        self._regime_flip_streak = 0
        self._trend_exit_dir = None
        self._trend_exit_streak = 0
        self._macro_dir_candidate = None
        self._macro_dir_streak = 0
        self._macro_effective_dir = "neutral"
        self._macro_lock_dir_candidate = None
        self._macro_lock_dir_streak = 0
        self._macro_lock_effective_dir = "neutral"
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
        self._pos_peak_price = None
        self._pos_trough_price = None
        self._scalp_ema = None
        self._edge_gate = None
        self._edge_gate_steps = 0
        self._edge_gate_mode = None

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
        if self.portfolio.position == 0:
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

    def _flat_unfreeze(self, posterior_scale_override: float | None) -> tuple[float | None, bool]:
        if not getattr(config, "ENABLE_FLAT_UNFREEZE", False):
            return posterior_scale_override, False
        if self.portfolio.position != 0:
            return posterior_scale_override, False
        min_steps = int(getattr(config, "FLAT_UNFREEZE_STEPS", 0) or 0)
        if min_steps <= 0:
            return posterior_scale_override, False
        steps_since_trade = self.steps if self.last_trade_step < 0 else (self.steps - self.last_trade_step)
        if steps_since_trade < min_steps:
            return posterior_scale_override, False
        base_scale = posterior_scale_override
        if base_scale is None:
            base_scale = self.agent.posterior_scale
        boosted_scale = base_scale + float(getattr(config, "FLAT_POSTERIOR_BOOST", 0.0))
        return boosted_scale, True

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

    def _compute_regime(self, row: pd.Series, *, warmup_active: bool) -> tuple[str, bool]:
        if not getattr(config, "ENABLE_REGIME_SWITCH", False) or warmup_active:
            return "neutral", False
        try:
            trend = float(row.get(getattr(config, "REGIME_TREND_COL", "trend_48"), np.nan))
        except Exception:
            trend = float("nan")
        try:
            vol = float(row.get(getattr(config, "REGIME_VOL_COL", "rv1m_pct_5m"), np.nan))
        except Exception:
            vol = float("nan")
        if not (np.isfinite(trend) and np.isfinite(vol)):
            return "neutral", True
        vol = self._sanitize_unit_percentile(vol, col=str(getattr(config, "REGIME_VOL_COL", "rv1m_pct_5m")))
        if vol < float(getattr(config, "REGIME_VOL_MIN", 0.0)):
            return "neutral", True
        if trend >= float(getattr(config, "REGIME_TREND_LONG_MIN", 0.0)):
            return "long", True
        if trend <= float(getattr(config, "REGIME_TREND_SHORT_MAX", 0.0)):
            return "short", True
        return "neutral", True

    def _compute_macro_bias(self, row: pd.Series) -> str:
        if not getattr(config, "ENABLE_MACRO_BIAS", False):
            return "neutral"
        try:
            trend = float(row.get(getattr(config, "MACRO_TREND_COL", "trend_240"), np.nan))
        except Exception:
            trend = float("nan")
        try:
            vol = float(row.get(getattr(config, "MACRO_VOL_COL", "rv1m_pct_5m"), np.nan))
        except Exception:
            vol = float("nan")
        if not (np.isfinite(trend) and np.isfinite(vol)):
            return "neutral"
        vol = self._sanitize_unit_percentile(vol, col=str(getattr(config, "MACRO_VOL_COL", "rv1m_pct_5m")))
        if vol < float(getattr(config, "MACRO_VOL_MIN", 0.0)):
            return "neutral"
        if trend >= float(getattr(config, "MACRO_TREND_LONG_MIN", 0.0)):
            return "long"
        if trend <= float(getattr(config, "MACRO_TREND_SHORT_MAX", 0.0)):
            return "short"
        return "neutral"

    def _compute_core_dir(self, row: pd.Series) -> str:
        if not getattr(config, "ENABLE_CORE_SCALP", False):
            return "neutral"
        try:
            trend = float(row.get(getattr(config, "CORE_TREND_COL", "trend_1000"), np.nan))
        except Exception:
            trend = float("nan")
        if not np.isfinite(trend):
            return "neutral"
        if trend >= float(getattr(config, "CORE_TREND_LONG_MIN", 0.0)):
            return "long"
        if trend <= float(getattr(config, "CORE_TREND_SHORT_MAX", 0.0)):
            return "short"
        return "neutral"

    def _update_scalp_ema(self, price: float) -> float:
        window = int(getattr(config, "SCALP_EMA_WINDOW", 20) or 20)
        if window <= 1:
            self._scalp_ema = price
            return price
        alpha = 2.0 / (window + 1.0)
        if self._scalp_ema is None:
            self._scalp_ema = price
        else:
            self._scalp_ema = (alpha * price) + ((1.0 - alpha) * float(self._scalp_ema))
        return float(self._scalp_ema)

    def _compute_macro_lock(self, row: pd.Series) -> str:
        if not getattr(config, "ENABLE_MACRO_LOCK", False):
            return "neutral"
        try:
            trend = float(row.get(getattr(config, "MACRO_LOCK_TREND_COL", "trend_1000"), np.nan))
        except Exception:
            trend = float("nan")
        try:
            vol = float(row.get(getattr(config, "MACRO_LOCK_VOL_COL", "rv1m_pct_5m"), np.nan))
        except Exception:
            vol = float("nan")
        if not (np.isfinite(trend) and np.isfinite(vol)):
            return "neutral"
        vol = self._sanitize_unit_percentile(vol, col=str(getattr(config, "MACRO_LOCK_VOL_COL", "rv1m_pct_5m")))
        if vol < float(getattr(config, "MACRO_LOCK_VOL_MIN", 0.0)):
            return "neutral"
        if trend >= float(getattr(config, "MACRO_LOCK_LONG_MIN", 0.0)):
            return "long"
        if trend <= float(getattr(config, "MACRO_LOCK_SHORT_MAX", 0.0)):
            return "short"
        return "neutral"

    def _dynamic_fraction(self, margin: float) -> float:
        conf = self._sigmoid(config.CONFIDENCE_K * max(0.0, margin))
        lo, hi = config.POSITION_FRACTION_MIN, config.POSITION_FRACTION_MAX
        return lo + conf * (hi - lo)

    def _edge_gate_features(self, row: pd.Series) -> np.ndarray:
        raw = np.nan_to_num(
            row[INDICATOR_COLUMNS].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        n = int(raw.size)
        rms = float(np.linalg.norm(raw) / max(np.sqrt(n), 1.0))
        scale = max(rms, 1e-6)
        scaled = raw / scale
        scaled = np.clip(scaled, -config.FEATURE_CLIP, config.FEATURE_CLIP)
        return np.concatenate([scaled, np.array([1.0])])

    def step(
        self,
        row: pd.Series,
        next_row: pd.Series,
        step_idx: int,
        *,
        train: bool = True,
        posterior_scale_override: float | None = None,
        future_price: float | None = None,
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
        flat_relax = False
        if not stuck_relax:
            posterior_scale_effective, flat_relax = self._flat_unfreeze(posterior_scale_effective)
        # Regime-adaptive edge gating: tighten in chop / relax in trend.
        # Disabled when stuck_relax is active (we already loosen gates there).
        edge_threshold = self._regime_adjust_edge(row, edge_threshold, stuck_relax=stuck_relax)
        regime, regime_gate_active = self._compute_regime(row, warmup_active=warmup_active)
        macro_dir = self._compute_macro_bias(row)
        macro_dir_effective = macro_dir
        if getattr(config, "ENABLE_MACRO_BIAS", False):
            if macro_dir in ("long", "short"):
                if macro_dir == self._macro_dir_candidate:
                    self._macro_dir_streak += 1
                else:
                    self._macro_dir_candidate = macro_dir
                    self._macro_dir_streak = 1
                if self._macro_dir_streak >= int(getattr(config, "MACRO_BIAS_DIR_STREAK", 1)):
                    self._macro_effective_dir = macro_dir
            macro_dir_effective = self._macro_effective_dir
        macro_lock_dir = self._compute_macro_lock(row)
        macro_lock_dir_effective = macro_lock_dir
        if getattr(config, "ENABLE_MACRO_LOCK", False):
            if macro_lock_dir == self._macro_lock_dir_candidate:
                self._macro_lock_dir_streak += 1
            else:
                self._macro_lock_dir_candidate = macro_lock_dir
                self._macro_lock_dir_streak = 1
            if self._macro_lock_dir_streak >= int(getattr(config, "MACRO_LOCK_DIR_STREAK", 1)):
                self._macro_lock_effective_dir = macro_lock_dir
            macro_lock_dir_effective = self._macro_lock_effective_dir

        # ------------------------------------------------------------------
        # Hard risk exits (optional): evaluate BEFORE policy action.
        #
        # This is intentionally not "gating". It is a risk overlay that can
        # override the policy to cut tail risk. It is OFF by default.
        # ------------------------------------------------------------------
        forced_exit = False
        forced_exit_reason: str | None = None
        forced_action: str | None = None
        scalp_override = False
        scalp_action: str | None = None
        edge_gate_pred: float | None = None
        if getattr(config, "ENABLE_HARD_RISK_EXITS", False) and self.portfolio.position != 0:
            entry = float(self.portfolio.entry_price or 0.0)
            if entry > 0:
                pos_dir = 1.0 if self.portfolio.position > 0 else -1.0
                pos_ret = ((price_now / entry) - 1.0) * pos_dir
                hold_steps = (self.steps - self.last_entry_step) if self.last_entry_step >= 0 else 0
                exit_action = "sell" if pos_dir > 0 else "buy"
                if hold_steps >= int(getattr(config, "MAX_POSITION_HOLD_STEPS", 10**9)):
                    forced_action = exit_action
                    forced_exit_reason = "time_stop"
                if pos_ret <= -float(getattr(config, "HARD_STOP_LOSS_PCT", 0.0)):
                    forced_action = exit_action
                    forced_exit_reason = "stop_loss"
                # trailing stop from peak (long) or trough (short) while in position
                if pos_dir > 0:
                    if not hasattr(self, "_trail_peak_price"):
                        self._trail_peak_price = price_now
                    self._trail_peak_price = max(float(self._trail_peak_price), price_now)
                    trail_from_peak = (price_now / max(float(self._trail_peak_price), 1e-9)) - 1.0
                    if trail_from_peak <= -float(getattr(config, "TRAILING_STOP_PCT", 0.0)):
                        forced_action = "sell"
                        forced_exit_reason = "trailing_stop"
                else:
                    if not hasattr(self, "_trail_trough_price"):
                        self._trail_trough_price = price_now
                    self._trail_trough_price = min(float(self._trail_trough_price), price_now)
                    trail_from_trough = (price_now / max(float(self._trail_trough_price), 1e-9)) - 1.0
                    if trail_from_trough >= float(getattr(config, "TRAILING_STOP_PCT", 0.0)):
                        forced_action = "buy"
                        forced_exit_reason = "trailing_stop"
        else:
            # Reset trackers when flat
            if hasattr(self, "_trail_peak_price"):
                self._trail_peak_price = price_now
            if hasattr(self, "_trail_trough_price"):
                self._trail_trough_price = price_now

        if (
            forced_action is None
            and getattr(config, "ENABLE_MACRO_BIAS", False)
            and self.portfolio.position != 0
        ):
            hold_steps = (self.steps - self.last_entry_step) if self.last_entry_step >= 0 else 0
            min_hold = int(getattr(config, "MACRO_BIAS_HOLD_MIN_STEPS", 0))
            pos_dir = 1.0 if self.portfolio.position > 0 else -1.0
            desired = "long" if pos_dir > 0 else "short"
            if macro_dir_effective == "neutral" and getattr(config, "MACRO_BIAS_EXIT_ON_NEUTRAL", False):
                if hold_steps >= min_hold:
                    forced_action = "sell" if pos_dir > 0 else "buy"
                    forced_exit_reason = "macro_neutral"
                    forced_exit = True
            elif (
                macro_dir_effective in ("long", "short")
                and macro_dir_effective != desired
                and getattr(config, "MACRO_BIAS_EXIT_ON_FLIP", False)
            ):
                if hold_steps >= min_hold:
                    forced_action = "sell" if pos_dir > 0 else "buy"
                    forced_exit_reason = "macro_flip"
                    forced_exit = True

        if (
            forced_action is None
            and getattr(config, "ENABLE_MACRO_LOCK", False)
            and self.portfolio.position != 0
        ):
            hold_steps = (self.steps - self.last_entry_step) if self.last_entry_step >= 0 else 0
            min_hold = int(getattr(config, "MACRO_LOCK_HOLD_MIN_STEPS", 0))
            pos_dir = 1.0 if self.portfolio.position > 0 else -1.0
            desired = "long" if pos_dir > 0 else "short"
            if macro_lock_dir_effective == "neutral" and getattr(config, "MACRO_LOCK_EXIT_ON_NEUTRAL", False):
                if hold_steps >= min_hold:
                    forced_action = "sell" if pos_dir > 0 else "buy"
                    forced_exit_reason = "macro_lock_neutral"
                    forced_exit = True
            elif (
                macro_lock_dir_effective in ("long", "short")
                and macro_lock_dir_effective != desired
                and getattr(config, "MACRO_LOCK_EXIT_ON_FLIP", False)
            ):
                if hold_steps >= min_hold:
                    forced_action = "sell" if pos_dir > 0 else "buy"
                    forced_exit_reason = "macro_lock_flip"
                    forced_exit = True

        if (
            forced_action is None
            and getattr(config, "ENABLE_TREND_EXIT", False)
            and self.portfolio.position != 0
        ):
            pos_dir = 1.0 if self.portfolio.position > 0 else -1.0
            current_dir = "long" if pos_dir > 0 else "short"
            if self._trend_exit_dir != current_dir:
                self._trend_exit_dir = current_dir
                self._trend_exit_streak = 0
            try:
                trend_val = float(row.get(getattr(config, "TREND_EXIT_COL", "trend_48"), np.nan))
            except Exception:
                trend_val = float("nan")
            if np.isfinite(trend_val):
                if pos_dir > 0:
                    against = trend_val <= float(getattr(config, "TREND_EXIT_LONG_MAX", 0.0))
                else:
                    against = trend_val >= float(getattr(config, "TREND_EXIT_SHORT_MIN", 0.0))
                if against:
                    self._trend_exit_streak += 1
                else:
                    self._trend_exit_streak = 0
                if self._trend_exit_streak >= int(getattr(config, "TREND_EXIT_STREAK", 1)):
                    forced_action = "sell" if pos_dir > 0 else "buy"
                    forced_exit_reason = "trend_exit"
                    forced_exit = True
            else:
                self._trend_exit_streak = 0
        else:
            self._trend_exit_dir = None
            self._trend_exit_streak = 0

        if (
            forced_action is None
            and getattr(config, "REGIME_EXIT_ON_FLIP", False)
            and regime_gate_active
            and self.portfolio.position != 0
        ):
            pos_dir = 1.0 if self.portfolio.position > 0 else -1.0
            desired_regime = "long" if pos_dir > 0 else "short"
            if regime in ("long", "short") and regime != desired_regime:
                if self._regime_flip_dir != regime:
                    self._regime_flip_dir = regime
                    self._regime_flip_streak = 1
                else:
                    self._regime_flip_streak += 1
                if self._regime_flip_streak >= int(getattr(config, "REGIME_EXIT_STREAK", 1)):
                    forced_action = "sell" if pos_dir > 0 else "buy"
                    forced_exit_reason = "regime_flip"
                    forced_exit = True
            else:
                self._regime_flip_dir = None
                self._regime_flip_streak = 0
        else:
            self._regime_flip_dir = None
            self._regime_flip_streak = 0

        core_dir = self._compute_core_dir(row)
        if getattr(config, "ENABLE_CORE_SCALP", False):
            ema = self._update_scalp_ema(price_now)
            band = float(getattr(config, "SCALP_BAND_PCT", 0.0))
            band_high = ema * (1.0 + band)
            band_low = ema * (1.0 - band)
            pv = self.portfolio.value(price_now)
            core_frac = float(getattr(config, "CORE_POSITION_FRACTION", 0.0))
            extra_frac = float(getattr(config, "SCALP_EXTRA_FRACTION", 0.0))
            max_legs = int(getattr(config, "SCALP_MAX_LEGS", 0) or 0)
            core_notional = pv * core_frac
            max_notional = pv * (core_frac + (extra_frac * max_legs))
            cur_notional = abs(self.portfolio.position) * price_now
            min_gap = int(getattr(config, "SCALP_MIN_GAP_STEPS", 0) or 0)
            since_trade = self.steps if self.last_trade_step < 0 else (self.steps - self.last_trade_step)
            gap_ok = since_trade >= min_gap
            if core_dir == "neutral" and getattr(config, "CORE_EXIT_ON_NEUTRAL", False) and self.portfolio.position != 0:
                if gap_ok and forced_action is None:
                    scalp_action = "sell" if self.portfolio.position > 0 else "buy"
            elif core_dir == "short" and gap_ok and forced_action is None:
                if self.portfolio.position >= 0 and cur_notional < core_notional:
                    scalp_action = "sell"
                elif price_now > band_high and cur_notional < max_notional:
                    scalp_action = "sell"
                elif price_now < band_low and cur_notional > core_notional:
                    scalp_action = "buy"
            elif core_dir == "long" and gap_ok and forced_action is None:
                if self.portfolio.position <= 0 and cur_notional < core_notional:
                    scalp_action = "buy"
                elif price_now < band_low and cur_notional < max_notional:
                    scalp_action = "buy"
                elif price_now > band_high and cur_notional > core_notional:
                    scalp_action = "sell"
            if scalp_action is not None:
                scalp_override = True

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

        flat_relax_filters = bool(flat_relax) and bool(getattr(config, "FLAT_RELAX_ENTRY_FILTERS", False))
        entry_vol_ok = True
        if (
            getattr(config, "ENABLE_ENTRY_VOL_FILTER", False)
            and (not warmup_active)
            and (not stuck_relax)
            and (not flat_relax_filters)
        ):
            try:
                v = float(row.get(getattr(config, "ENTRY_VOL_COL", "rv1m_pct_5m"), np.nan))
            except Exception:
                v = float("nan")
            if np.isfinite(v):
                v = self._sanitize_unit_percentile(v, col=str(getattr(config, "ENTRY_VOL_COL", "rv1m_pct_5m")))
                entry_vol_ok = v >= float(getattr(config, "ENTRY_VOL_PCT_MIN", 0.0))
        entry_basis_ok = True
        if (
            getattr(config, "ENABLE_BASIS_ENTRY_FILTER", False)
            and (not warmup_active)
            and (not stuck_relax)
            and (not flat_relax_filters)
        ):
            try:
                basis = float(row.get("basis_pct", np.nan))
            except Exception:
                basis = float("nan")
            if np.isfinite(basis):
                entry_basis_ok = basis >= float(getattr(config, "BASIS_ENTRY_MIN", 0.0))
        entry_trend_ok = True
        if (
            getattr(config, "ENABLE_TREND_ENTRY_FILTER", False)
            and (not warmup_active)
            and (not stuck_relax)
            and (not flat_relax_filters)
        ):
            try:
                trend_col = getattr(config, "TREND_ENTRY_COL", "trend_48")
                trend = float(row.get(trend_col, np.nan))
            except Exception:
                trend = float("nan")
            if np.isfinite(trend):
                entry_trend_ok = trend >= float(getattr(config, "TREND_ENTRY_MIN", 0.0))
        entry_short_trend_ok = True
        if (
            getattr(config, "ENABLE_SHORT_TREND_FILTER", False)
            and (not warmup_active)
            and (not stuck_relax)
            and (not flat_relax_filters)
        ):
            try:
                trend_col = getattr(config, "SHORT_TREND_ENTRY_COL", "trend_48")
                trend = float(row.get(trend_col, np.nan))
            except Exception:
                trend = float("nan")
            if np.isfinite(trend):
                entry_short_trend_ok = trend <= float(getattr(config, "SHORT_TREND_ENTRY_MAX", 0.0))
        entry_flow_ok = True
        if (
            getattr(config, "ENABLE_FLOW_ENTRY_FILTER", False)
            and (not warmup_active)
            and (not stuck_relax)
            and (not flat_relax_filters)
        ):
            try:
                flow_col = getattr(config, "FLOW_ENTRY_COL", "aggr_imb")
                flow_val = float(row.get(flow_col, np.nan))
            except Exception:
                flow_val = float("nan")
            if np.isfinite(flow_val):
                entry_flow_ok = flow_val >= float(getattr(config, "FLOW_ENTRY_LONG_MIN", 0.0))
        entry_flow_short_ok = True
        if (
            getattr(config, "ENABLE_FLOW_SHORT_FILTER", False)
            and (not warmup_active)
            and (not stuck_relax)
            and (not flat_relax_filters)
        ):
            try:
                flow_col = getattr(config, "FLOW_ENTRY_COL", "aggr_imb")
                flow_val = float(row.get(flow_col, np.nan))
            except Exception:
                flow_val = float("nan")
            if np.isfinite(flow_val):
                entry_flow_short_ok = flow_val <= float(getattr(config, "FLOW_ENTRY_SHORT_MAX", 0.0))


        strong_trend_ok = True
        strong_vol_ok = True
        strong_gate_active = bool(getattr(config, "ENABLE_STRONG_TREND_GATE", False)) and (not warmup_active)
        if strong_gate_active:
            try:
                trend = float(row.get(getattr(config, "STRONG_TREND_COL", "trend_48"), np.nan))
            except Exception:
                trend = float("nan")
            if np.isfinite(trend):
                strong_trend_ok = abs(trend) >= float(getattr(config, "STRONG_TREND_MIN_ABS", 0.0))
            else:
                strong_trend_ok = False
            vol_col = getattr(config, "STRONG_TREND_VOL_COL", "")
            if vol_col:
                try:
                    v = float(row.get(vol_col, np.nan))
                except Exception:
                    v = float("nan")
                if np.isfinite(v):
                    v = self._sanitize_unit_percentile(v, col=str(vol_col))
                    strong_vol_ok = v >= float(getattr(config, "STRONG_TREND_VOL_MIN", 0.0))
                else:
                    strong_vol_ok = False

        allowed_actions = ["hold"]
        allow_longs = bool(getattr(config, "ENABLE_LONGS", True))
        allow_shorts = bool(getattr(config, "ENABLE_SHORTS", False))
        stoploss_cooldown_steps = int(getattr(config, "STOP_LOSS_COOLDOWN_STEPS", 0) or 0)
        stoploss_cooldown_active = (
            stoploss_cooldown_steps > 0
            and self.last_stop_loss_step >= 0
            and (self.steps - self.last_stop_loss_step) < stoploss_cooldown_steps
        )
        macro_bias_active = bool(getattr(config, "ENABLE_MACRO_BIAS", False))
        macro_lock_enabled = bool(getattr(config, "ENABLE_MACRO_LOCK", False))
        macro_force_entry = False
        macro_force_dir: str | None = None
        macro_hold_to_flip = False
        macro_dir_for_limits: str | None = None
        macro_force_enabled = False
        macro_force_steps = 0
        macro_neutral_reason: str | None = None
        if macro_bias_active:
            macro_dir_for_limits = macro_dir_effective
            macro_hold_to_flip = bool(getattr(config, "MACRO_BIAS_HOLD_TO_FLIP", False))
            macro_force_enabled = bool(getattr(config, "MACRO_BIAS_FORCE_ENTRY", False))
            macro_force_steps = int(getattr(config, "MACRO_BIAS_FORCE_ENTRY_STEPS", 0))
            macro_neutral_reason = "macro_neutral"
        if macro_lock_enabled:
            macro_dir_for_limits = macro_lock_dir_effective
            macro_hold_to_flip = bool(getattr(config, "MACRO_LOCK_HOLD_TO_FLIP", False))
            macro_force_enabled = bool(getattr(config, "MACRO_LOCK_FORCE_ENTRY", False))
            macro_force_steps = int(getattr(config, "MACRO_LOCK_FORCE_ENTRY_STEPS", 0))
            macro_neutral_reason = "macro_lock_neutral"
        if macro_dir_for_limits is not None:
            if macro_dir_for_limits == "long":
                allow_shorts = False
            elif macro_dir_for_limits == "short":
                allow_longs = False
            else:
                allow_longs = False
                allow_shorts = False
                if macro_neutral_reason:
                    self._premask_gate_reason_counter[macro_neutral_reason] += 1
            if (
                macro_dir_for_limits in ("long", "short")
                and self.portfolio.position == 0
                and macro_force_enabled
            ):
                since_trade = self.steps if self.last_trade_step < 0 else (self.steps - self.last_trade_step)
                if since_trade >= macro_force_steps:
                    macro_force_entry = True
                    macro_force_dir = macro_dir_for_limits
        if macro_force_entry:
            if macro_force_dir == "long" and (not entry_trend_ok):
                macro_force_entry = False
            elif macro_force_dir == "short" and (not entry_short_trend_ok):
                macro_force_entry = False
        strong_gate_block = strong_gate_active and not (strong_trend_ok and strong_vol_ok)
        regime_gate_block = regime_gate_active and (regime == "neutral")
        if strong_gate_block:
            if not strong_trend_ok:
                self._premask_gate_reason_counter["strong_trend_gate"] += 1
            if not strong_vol_ok:
                self._premask_gate_reason_counter["strong_trend_vol_gate"] += 1
        if regime_gate_block:
            self._premask_gate_reason_counter["regime_neutral"] += 1
        if self.portfolio.cash > 0:
            if self.portfolio.position == 0:
                # Long entry
                if stoploss_cooldown_active:
                    self._premask_gate_reason_counter["stoploss_cooldown_entry"] += 1
                elif (not strong_gate_block) and (not regime_gate_block) and allow_longs and (warmup_active or gap_ok or stuck_gap_ok):
                    if regime_gate_active and regime != "long":
                        self._premask_gate_reason_counter["regime_block_long"] += 1
                    if not (stuck_relax and not getattr(config, "STUCK_ALLOW_BUY", False)):
                        if (not entry_vol_ok):
                            self._premask_gate_reason_counter["entry_vol_buy"] += 1
                        elif (not entry_basis_ok):
                            self._premask_gate_reason_counter["basis_entry_buy"] += 1
                        elif (not entry_trend_ok):
                            self._premask_gate_reason_counter["trend_entry_buy"] += 1
                        elif (not entry_flow_ok):
                            self._premask_gate_reason_counter["flow_entry_buy"] += 1
                        else:
                            allowed_actions.append("buy")
                # Short entry (optional)
                if stoploss_cooldown_active:
                    self._premask_gate_reason_counter["stoploss_cooldown_entry"] += 1
                elif (not strong_gate_block) and (not regime_gate_block) and allow_shorts and (warmup_active or gap_ok or stuck_gap_ok):
                    if regime_gate_active and regime != "short":
                        self._premask_gate_reason_counter["regime_block_short"] += 1
                    if not entry_short_trend_ok:
                        self._premask_gate_reason_counter["trend_entry_sell"] += 1
                    elif not entry_flow_short_ok:
                        self._premask_gate_reason_counter["flow_entry_sell"] += 1
                    else:
                        allowed_actions.append("sell")
            elif self.portfolio.position > 0:
                # Allow BUY scaling-in (optional).
                max_legs = int(getattr(config, "MAX_BUY_LEGS_PER_POSITION", 1))
                if (not strong_gate_block) and allow_longs and max_legs > 1 and self._buy_legs_current < max_legs:
                    if warmup_active or gap_ok or stuck_gap_ok:
                        if not (stuck_relax and not getattr(config, "STUCK_ALLOW_BUY", False)):
                            if (not entry_vol_ok):
                                self._premask_gate_reason_counter["entry_vol_buy"] += 1
                            elif (not entry_basis_ok):
                                self._premask_gate_reason_counter["basis_entry_buy"] += 1
                            elif (not entry_trend_ok):
                                self._premask_gate_reason_counter["trend_entry_buy"] += 1
                            elif (not entry_flow_ok):
                                self._premask_gate_reason_counter["flow_entry_buy"] += 1
                            else:
                                allowed_actions.append("buy")
                # SELL: do not offer during trade-gap cooldown unless it qualifies for a strong-exit bypass.
                if warmup_active or gap_ok or stuck_gap_ok:
                    allowed_actions.append("sell")
                else:
                    since_last_trade = (self.steps - self.last_trade_step) if self.last_trade_step >= 0 else 10**9
                    strong_min_gap = int(getattr(config, 'COOLDOWN_STRONG_MIN_GAP_STEPS', 0))
                    strong_min_gap = min(strong_min_gap, max(eff_gap, int(getattr(config, "COOLDOWN_MIN_GAP_FLOOR", 0))))
                    if since_last_trade >= strong_min_gap:
                        allowed_actions.append("sell")
            else:
                # Short position: allow BUY to cover (exit) under the same cooldown rules as sells.
                if warmup_active or gap_ok or stuck_gap_ok:
                    allowed_actions.append("buy")
                else:
                    since_last_trade = (self.steps - self.last_trade_step) if self.last_trade_step >= 0 else 10**9
                    strong_min_gap = int(getattr(config, 'COOLDOWN_STRONG_MIN_GAP_STEPS', 0))
                    strong_min_gap = min(strong_min_gap, max(eff_gap, int(getattr(config, "COOLDOWN_MIN_GAP_FLOOR", 0))))
                    if since_last_trade >= strong_min_gap:
                        allowed_actions.append("buy")

        if (
            macro_hold_to_flip
            and self.portfolio.position != 0
            and macro_dir_for_limits in ("long", "short")
        ):
            pos_dir = "long" if self.portfolio.position > 0 else "short"
            if macro_dir_for_limits == pos_dir:
                # Hold the macro-aligned position until the regime flips or a forced exit fires.
                exit_action = "sell" if pos_dir == "long" else "buy"
                if forced_action is None:
                    allowed_actions = [a for a in allowed_actions if a != exit_action]

        edge_gate_active = bool(getattr(config, "ENABLE_EDGE_GATE", False))
        if edge_gate_active and self._edge_gate is None:
            gate_mode = str(getattr(config, "EDGE_GATE_MODEL", "ridge")).lower()
            if gate_mode == "logistic":
                self._edge_gate = EdgeGateClassifier(
                    len(INDICATOR_COLUMNS) + 1,
                    lr=float(getattr(config, "EDGE_GATE_LR", 0.01)),
                    l2=float(getattr(config, "EDGE_GATE_L2", 0.0)),
                    decay=float(getattr(config, "EDGE_GATE_DECAY", 1.0)),
                )
                self._edge_gate_mode = "logistic"
            else:
                self._edge_gate = EdgeGateModel(
                    len(INDICATOR_COLUMNS) + 1,
                    ridge=float(getattr(config, "EDGE_GATE_RIDGE", 1.0)),
                    decay=float(getattr(config, "EDGE_GATE_DECAY", 1.0)),
                )
                self._edge_gate_mode = "ridge"
        edge_gate_ready = edge_gate_active and (self._edge_gate is not None)
        edge_gate_ready = edge_gate_ready and (self._edge_gate_steps >= int(getattr(config, "EDGE_GATE_WARMUP_STEPS", 0)))
        if edge_gate_ready and (not macro_force_entry) and (not scalp_override) and (forced_action is None):
            edge_features = self._edge_gate_features(row)
            pred_edge = self._edge_gate.predict(edge_features)
            edge_gate_pred = pred_edge
            if self._edge_gate_mode == "logistic":
                prob = 1.0 / (1.0 + math.exp(-_clamp(float(pred_edge), -20.0, 20.0)))
                edge_gate_pred = (prob * 2.0) - 1.0
            sign_only = bool(getattr(config, "EDGE_GATE_SIGN_ONLY", False))
            if sign_only:
                edge_gate_pred = 1.0 if edge_gate_pred >= 0.0 else -1.0
            est_cost = (config.FEE_RATE + config.SLIPPAGE_RATE + config.GATE_SAFETY_MARGIN)
            edge_threshold = 0.0 if sign_only else max(
                float(getattr(config, "EDGE_GATE_MIN_EDGE", 0.0)),
                float(getattr(config, "EDGE_GATE_COST_MULT", 1.0)) * est_cost,
            )
            if not bool(getattr(config, "ENABLE_EDGE_POLICY", False)):
                if "buy" in allowed_actions and self.portfolio.position >= 0 and edge_gate_pred < edge_threshold:
                    allowed_actions = [a for a in allowed_actions if a != "buy"]
                    self._premask_gate_reason_counter["edge_gate_buy"] += 1
                if "sell" in allowed_actions and self.portfolio.position == 0 and allow_shorts and edge_gate_pred > -edge_threshold:
                    allowed_actions = [a for a in allowed_actions if a != "sell"]
                    self._premask_gate_reason_counter["edge_gate_sell"] += 1

        edge_policy_active = bool(getattr(config, "ENABLE_EDGE_POLICY", False)) and edge_gate_ready and (edge_gate_pred is not None)

        # ------------------------------------------------------------------
        # Turnover feasibility pre-mask: if we're already beyond the hard
        # turnover budget in the recent window, do not even offer BUY.
        #
        # Why: otherwise the agent keeps proposing BUY and later gets converted
        # to HOLD with hold_reason="turnover_budget", inflating budget_blocks and
        # weakening the learning signal.
        # ------------------------------------------------------------------
        if not warmup_active and not macro_force_entry and not scalp_override and not edge_policy_active:
            turnover_budget = self.initial_cash * self.turnover_budget_multiplier
            turnover_now = float(sum(self._turnover_window))
            hard_mult = float(getattr(config, 'TURNOVER_HARD_BLOCK_MULT', 1.0))
            turnover_stressed = turnover_now > (turnover_budget * hard_mult)
            bypass_mult = float(getattr(config, "TURNOVER_BUY_BYPASS_EDGE_MULT", 0.0))
            if turnover_stressed and bypass_mult <= 0.0:
                # Block new exposure (long or short) when turnover is stressed.
                if self.portfolio.position >= 0 and "buy" in allowed_actions:
                    allowed_actions = [a for a in allowed_actions if a != "buy"]
                    self._premask_gate_reason_counter["turnover_budget_buy"] += 1
                if self.portfolio.position == 0 and "sell" in allowed_actions and allow_shorts:
                    allowed_actions = [a for a in allowed_actions if a != "sell"]
                    self._premask_gate_reason_counter["turnover_budget_sell"] += 1

        # (moved earlier) stuck adaptation + regime edge adjustment are computed before action masking

        if forced_action is not None:
            # Policy override: log as a forced exit; the agent still learns from the executed action.
            action = forced_action
            sampled_scores = np.zeros(len(ACTIONS), dtype=float)
            means = np.zeros(len(ACTIONS), dtype=float)
            forced_exit = True
        elif scalp_override and scalp_action is not None:
            action = scalp_action
            sampled_scores = np.zeros(len(ACTIONS), dtype=float)
            means = np.zeros(len(ACTIONS), dtype=float)
        elif macro_force_entry:
            force_dir = macro_force_dir or macro_dir_effective
            action = "buy" if force_dir == "long" else "sell"
            sampled_scores = np.zeros(len(ACTIONS), dtype=float)
            means = np.zeros(len(ACTIONS), dtype=float)
        elif (
            bool(getattr(config, "ENABLE_EDGE_POLICY", False))
            and edge_gate_pred is not None
            and edge_gate_ready
        ):
            sign_only = bool(getattr(config, "EDGE_POLICY_SIGN_ONLY", False))
            edge_min = 0.0 if sign_only else float(getattr(config, "EDGE_POLICY_MIN_EDGE", 0.0))
            if edge_gate_pred >= edge_min:
                action = "buy"
            elif edge_gate_pred <= -edge_min:
                action = "sell"
            else:
                action = "hold"
            if action not in allowed_actions:
                action = "hold"
            sampled_scores = np.zeros(len(ACTIONS), dtype=float)
            means = np.zeros(len(ACTIONS), dtype=float)
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
        if edge_policy_active and edge_gate_pred is not None:
            if edge_gate_pred > 0.0:
                buy_margin = max(buy_margin, float(edge_gate_pred))
            elif edge_gate_pred < 0.0:
                sell_margin = max(sell_margin, float(-edge_gate_pred))

        if config.COST_AWARE_GATING:
            est_cost = (config.FEE_RATE + config.SLIPPAGE_RATE + config.GATE_SAFETY_MARGIN)
            # Keep the standard cost gate even when stuck to avoid low-edge churn.
            cost_edge = max(edge_threshold, config.COST_EDGE_MULT * est_cost) + config.EDGE_SAFETY_MARGIN
        else:
            cost_edge = edge_threshold
        atr_edge = 0.0
        if getattr(config, "ENABLE_ATR_ENTRY_GATE", False):
            try:
                atr_col = getattr(config, "ATR_COL", "atr_14")
                atr = float(row.get(atr_col, 0.0))
            except Exception:
                atr = 0.0
            if atr > 0.0 and price_now > 0.0:
                atr_edge = float(getattr(config, "ATR_ENTRY_MULT", 0.0)) * (atr / price_now)

        if not warmup_active and not macro_force_entry and not scalp_override and not edge_policy_active:
            feasible = list(allowed_actions)

            # 1) Cost feasibility: do not propose trades that do not clear cost_edge.
            if "buy" in feasible and self.portfolio.position >= 0:
                req_edge = cost_edge
                atr_block = False
                if atr_edge > req_edge:
                    req_edge = atr_edge
                    atr_block = True
                if buy_margin < req_edge:
                    feasible.remove("buy")
                    # Pre-mask veto: the action never becomes proposed_action, so without
                    # this counter it would be invisible in gate_blocks/gate_reason_counts.
                    if atr_block:
                        self._premask_gate_reason_counter["atr_entry_gate"] += 1
                    else:
                        self._premask_gate_reason_counter["cost_gate_buy"] += 1
            # SELL: do not pre-mask exits on cost edge. Exits are *risk reducing* and the agent
            # must always have the option to close exposure; otherwise the system can HOLD-freeze
            # in-position for thousands of steps when margins are small.
            # We still account for real execution frictions in PnL and learning reward.
            if "sell" in feasible and self.portfolio.position == 0:
                # (defensive) Only apply sell cost gating when flat (short entry).
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
            # Cooldown feasibility for BUY when covering a short.
            if "buy" in feasible and self.portfolio.position < 0 and (not gap_ok) and (not stuck_relax):
                strong_min_gap = int(getattr(config, 'COOLDOWN_STRONG_MIN_GAP_STEPS', 0))
                strong_min_gap = min(strong_min_gap, max(eff_gap, int(getattr(config, "COOLDOWN_MIN_GAP_FLOOR", 0))))
                min_gap_ok = since_last_trade >= strong_min_gap
                strong_buy = (buy_margin >= (config.COOLDOWN_STRONG_EDGE_MULT * cost_edge)) and min_gap_ok
                if not strong_buy:
                    feasible.remove("buy")
                    self._premask_gate_reason_counter["cooldown_buy"] += 1

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
        value_before = self.portfolio.value(price_now)
        position_before = self.portfolio.position
        cash_before = self.portfolio.cash

        # --- cost-aware gating -------------------------------------------------
        # In stuck mode, do not re-veto here (we already softened the threshold above).
        if config.COST_AWARE_GATING:
            est_cost = (config.FEE_RATE + config.SLIPPAGE_RATE + config.GATE_SAFETY_MARGIN)
            cost_edge = max(edge_threshold, config.COST_EDGE_MULT * est_cost) + config.EDGE_SAFETY_MARGIN
        else:
            cost_edge = edge_threshold

        if (not warmup_active) and (not stuck_relax) and (not forced_exit) and (not macro_force_entry) and (not scalp_override) and (not edge_policy_active):
            if action == "buy" and position_before >= 0 and buy_margin < cost_edge:
                action = "hold"
                hold_reason = "cost_gate"
                gate_blocked = True
                self.gate_blocks += 1
                self._gate_reason_counter["cost_gate"] += 1
            elif action == "sell" and position_before <= 0 and sell_margin < cost_edge:
                action = "hold"
                hold_reason = "cost_gate"
                gate_blocked = True
                self.gate_blocks += 1
                self._gate_reason_counter["cost_gate"] += 1
            else:
                # Do not cost-gate exits. Exits reduce exposure and are required for recovery.
                # Execution frictions are still applied in portfolio PnL and learning reward.
                pass

        # --- adaptive cooldown (regime + turnover aware) ----------------------
        # NOTE: eff_gap/eff_hold and gap_ok/hold_ok are computed earlier (before act())
        # so the agent only sees timing-feasible actions. We keep using the computed
        # gap_ok/hold_ok here for the final enforcement step below.

        unrealized_ret = 0.0
        if position_before != 0 and self.portfolio.entry_price > 0:
            raw_ret = (price_now / max(self.portfolio.entry_price, 1e-9)) - 1.0
            unrealized_ret = raw_ret if position_before > 0 else -raw_ret

        if (not warmup_active) and (not stuck_relax) and (not macro_force_entry) and (not scalp_override) and (not edge_policy_active):
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
            exit_action = "sell" if position_before > 0 else "buy" if position_before < 0 else None
            if action == exit_action and not strong_cooldown_ok:
                underwater_exit = unrealized_ret <= -float(getattr(config, "COOLDOWN_SELL_UNDERWATER_EXIT_PCT", 0.0))
                if underwater_exit and since_last_trade >= gap_floor:
                    strong_cooldown_ok = True
            if action in ("buy", "sell") and (not gap_ok) and (not strong_cooldown_ok):
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
            and (not macro_force_entry)
            and (not scalp_override)
            and (not edge_policy_active)
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
        opening_long = action == "buy" and position_before >= 0
        opening_short = action == "sell" and position_before <= 0
        if (not warmup_active) and turnover_stressed and (opening_long or opening_short):
            bypass_mult = float(getattr(config, "TURNOVER_BUY_BYPASS_EDGE_MULT", 0.0))
            action_margin = buy_margin if action == "buy" else sell_margin
            if bypass_mult <= 0.0 or action_margin < (bypass_mult * cost_edge):
                action = "hold"
                hold_reason = hold_reason or "turnover_budget"
                budget_blocked = True

        # --- trade-rate throttling (anti-churn) -------------------------------
        # Count executed trade *legs* over a rolling step window and stop opening
        # new exposure when the policy is churning. This complements cooldowns:
        # cooldowns prevent immediate flip-flops; this prevents slow fee bleed.
        tr_window = int(getattr(config, 'TRADE_RATE_WINDOW_STEPS', 0) or 0)
        tr_max = int(getattr(config, 'MAX_TRADES_PER_WINDOW', 0) or 0)
        # Trade-rate throttling prevents churn on entries. It must NOT block exits; otherwise
        # long runs can freeze into HOLD forever after hitting the trade limit once.
        if (not warmup_active) and (not stuck_relax) and tr_window > 0 and tr_max > 0 and (opening_long or opening_short):
            recent_trades = int(sum(getattr(self, '_trade_count_window', [])))
            if recent_trades >= tr_max:
                action = "hold"
                hold_reason = hold_reason or "trade_rate"
                budget_blocked = True

        # --- position-level risk exits (hard stop + take-profit / trailing TP) ---
        entry_price = float(self.portfolio.entry_price or 0.0)
        if position_before != 0 and entry_price > 0:
            pos_dir = 1.0 if position_before > 0 else -1.0
            if pos_dir > 0:
                self._pos_trough_price = None
                self._pos_peak_price = (
                    price_now if self._pos_peak_price is None else max(float(self._pos_peak_price), price_now)
                )
            else:
                self._pos_peak_price = None
                self._pos_trough_price = (
                    price_now if getattr(self, "_pos_trough_price", None) is None else min(float(self._pos_trough_price), price_now)
                )
            pos_ret = ((price_now / entry_price) - 1.0) * pos_dir
            stop_pct = float(getattr(config, "STOP_LOSS_PCT", 0.0))
            tp_pct = float(getattr(config, "TAKE_PROFIT_PCT", 0.0))
            if getattr(config, "USE_ATR_EXITS", False):
                atr_col = getattr(config, "ATR_COL", "atr_14")
                atr = float(row.get(atr_col, 0.0) or 0.0)
                if atr > 0.0 and entry_price > 0.0:
                    stop_pct = float(getattr(config, "ATR_STOP_MULT", 0.0)) * (atr / entry_price)
                    tp_pct = float(getattr(config, "ATR_TP_MULT", 0.0)) * (atr / entry_price)
            stop_hit = pos_ret <= -stop_pct
            tp_hit = pos_ret >= tp_pct
            trail_hit = False
            if bool(getattr(config, "USE_TRAILING_TP", False)):
                if pos_dir > 0 and self._pos_peak_price is not None:
                    trail_hit = price_now <= float(self._pos_peak_price) * (1 - float(getattr(config, "TRAILING_TP_PCT", 0.0)))
                elif pos_dir < 0 and getattr(self, "_pos_trough_price", None) is not None:
                    trail_hit = price_now >= float(self._pos_trough_price) * (1 + float(getattr(config, "TRAILING_TP_PCT", 0.0)))
            if (stop_hit or tp_hit or trail_hit) and (not forced_exit):
                forced_exit = True
                forced_exit_reason = (
                    "stop_loss" if stop_hit else "take_profit" if tp_hit else "trailing_tp"
                )
                action = "sell" if pos_dir > 0 else "buy"
                hold_reason = None
                gate_blocked = False
                timing_blocked = False
                budget_blocked = False
        else:
            self._pos_peak_price = None
            self._pos_trough_price = None

        # --- discretionary sell break-even filter --------------------------------
        # Block churny SELLs that cannot cover estimated round-trip friction unless:
        # - we're in a forced/stuck risk exit, or
        # - the position is sufficiently underwater (loss-cut), or
        # - the edge is strong enough to override.
        exit_action = "sell" if position_before > 0 else "buy" if position_before < 0 else None
        if (
            action == exit_action
            and position_before != 0
            and (not forced_exit)
            and (not stuck_relax)
        ):
            entry = float(self.portfolio.entry_price or 0.0)
            raw_ret = (price_now / max(entry, 1e-9)) - 1.0 if entry > 0 else 0.0
            unreal = raw_ret if position_before > 0 else -raw_ret
            underwater_exit = unreal <= -float(getattr(config, "COOLDOWN_SELL_UNDERWATER_EXIT_PCT", 0.0))
            hold_steps = (self.steps - self.last_entry_step) if self.last_entry_step >= 0 else 0
            max_hold = int(getattr(config, "SELL_BREAKEVEN_MAX_HOLD_STEPS", 0) or 0)
            time_bypass = (max_hold > 0) and (hold_steps >= max_hold)
            est_oneway = config.FEE_RATE + config.SLIPPAGE_RATE
            est_roundtrip = (2.0 * est_oneway) + config.GATE_SAFETY_MARGIN
            req_profit = max(
                float(getattr(config, "MIN_PROFIT_TO_SELL_PCT", 0.0)),
                float(getattr(config, "MIN_PROFIT_TO_SELL_MULT_OF_COST", 1.0)) * est_roundtrip,
            )
            margin = sell_margin if exit_action == "sell" else buy_margin
            strong_override = margin >= (float(getattr(config, "SELL_BREAKEVEN_STRONG_EDGE_MULT", 0.0)) * cost_edge)
            if (not underwater_exit) and (not strong_override) and (not time_bypass) and unreal < req_profit:
                action = "hold"
                hold_reason = hold_reason or "sell_breakeven"
                gate_blocked = True
                self.gate_blocks += 1
                self._gate_reason_counter["sell_breakeven"] += 1

        trade_executed = False
        fee_paid = 0.0
        turnover_penalty = 0.0
        slippage_paid = 0.0
        realized_pnl = 0.0
        notional_traded = 0.0
        trade_size = 0.0

        # Will be updated after execution; used for unbiased reward calculation.
        value_after = value_before

        if action == "hold" and stuck_relax and proposed_action in allowed_actions:
            # When stuck in HOLD, allow a *limited* exit (SELL for longs, BUY for shorts)
            # to reduce exposure. Do NOT force new entries under stuck-relax.
            if config.COST_AWARE_GATING:
                # IMPORTANT: TURNOVER_PENALTY is a *learning regularizer*, not an execution cost.
                # Do not include it in cost-aware gating, otherwise the strategy gets over-conservative
                # and HOLD-freezes (especially on 1m).
                est_cost = (
                    config.FEE_RATE
                    + config.SLIPPAGE_RATE
                    + config.GATE_SAFETY_MARGIN
                )
                # Keep the standard cost edge; stuck mode should not bypass friction filters.
                stuck_cost_edge = max(edge_threshold, config.COST_EDGE_MULT * est_cost) + config.EDGE_SAFETY_MARGIN
            else:
                stuck_cost_edge = edge_threshold
            if (
                proposed_action == "sell"
                and position_before > 0
                and (gap_ok or stuck_gap_ok)
                and sell_margin >= stuck_cost_edge
            ):
                action = "sell"
                hold_reason = None
                gate_blocked = False
            elif (
                proposed_action == "buy"
                and position_before < 0
                and (gap_ok or stuck_gap_ok)
                and buy_margin >= stuck_cost_edge
            ):
                action = "buy"
                hold_reason = None
                gate_blocked = False

        if stuck_relax:
            self._stuck_action_counter[action] += 1
            self._stuck_proposed_action_counter[proposed_action] += 1

        # --- execution ---------------------------------------------------------
        # Professional separation of frictions:
        # - fee_paid + slippage_paid are *execution costs* (affect portfolio value)
        # - turnover_penalty is a *learning regularizer* (does NOT affect portfolio value; applied in trainer_reward)
        if action == "buy" and cash_before > 0:
            if position_before < 0:
                # Cover short position.
                buy_frac = self._dynamic_fraction(buy_margin) if config.PARTIAL_SELLS else 1.0
                if forced_exit and getattr(config, "FORCE_FULL_EXIT_ON_RISK", False):
                    buy_frac = 1.0
                if buy_margin >= (config.COOLDOWN_STRONG_EDGE_MULT * cost_edge):
                    buy_frac = 1.0
                trade_size = abs(position_before) * min(1.0, max(0.0, buy_frac))
                remaining_pos = position_before + trade_size
                if remaining_pos < 0 and (abs(remaining_pos) * price_now) < float(config.DUST_POSITION_NOTIONAL):
                    trade_size = abs(position_before)
                if trade_size > 0:
                    trade_executed = True
                    # stash entry price BEFORE mutating portfolio (needed for per-leg win-rate)
                    entry_price_for_leg = self.portfolio.entry_price
                    gross_cost = trade_size * price_now
                    min_notional = float(getattr(config, 'MIN_TRADE_NOTIONAL', 0.0) or 0.0)
                    if min_notional > 0.0 and gross_cost < min_notional:
                        full_notional = abs(position_before) * price_now
                        if full_notional <= max(min_notional, float(getattr(config, 'DUST_POSITION_NOTIONAL', 0.0) or 0.0)):
                            trade_size = abs(position_before)
                            gross_cost = trade_size * price_now
                        else:
                            trade_executed = False
                            action = "hold"
                            hold_reason = hold_reason or "min_notional"
                            budget_blocked = True
                            fee_paid = 0.0
                            slippage_paid = 0.0
                            notional_traded = 0.0
                            trade_size = 0.0
                    if trade_executed:
                        fee_paid = gross_cost * config.FEE_RATE
                        slippage_paid = gross_cost * config.SLIPPAGE_RATE
                        total_cost = gross_cost + fee_paid + slippage_paid
                        if total_cost <= cash_before:
                            notional_traded = gross_cost
                            self.portfolio.cash = cash_before - total_cost
                            self.portfolio.position = position_before + trade_size
                            fully_closed = self.portfolio.position >= -1e-12
                            if fully_closed:
                                realized_pnl = self.portfolio.cash - self.portfolio.entry_value
                                self.portfolio.entry_price = 0.0
                                self.portfolio.entry_value = 0.0
                                self._pos_trough_price = None
                                self.last_entry_step = -1
                                self._buy_legs_current = 0
                            self.last_trade_step = self.steps
                        else:
                            action = "hold"
                            hold_reason = hold_reason or "budget"
                            budget_blocked = True
                            fee_paid = 0.0
                            slippage_paid = 0.0
                            notional_traded = 0.0
                            trade_size = 0.0
                else:
                    action = "hold"
                    hold_reason = hold_reason or "budget"
                    budget_blocked = True
            else:
                pos_frac = self._dynamic_fraction(buy_margin)
                if getattr(config, "ENABLE_ATR_POSITION_SIZING", False):
                    try:
                        atr_col = getattr(config, "ATR_COL", "atr_14")
                        atr = float(row.get(atr_col, 0.0))
                    except Exception:
                        atr = 0.0
                    stop_pct = float(getattr(config, "STOP_LOSS_PCT", 0.0))
                    if getattr(config, "USE_ATR_EXITS", False) and atr > 0.0 and price_now > 0.0:
                        stop_pct = float(getattr(config, "ATR_STOP_MULT", 0.0)) * (atr / price_now)
                    target_risk = float(getattr(config, "ATR_TARGET_RISK_PCT", 0.0))
                    if stop_pct > 0.0 and target_risk > 0.0:
                        scale = min(1.0, target_risk / stop_pct)
                        pos_frac = max(0.0, pos_frac * scale)
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

        elif action == "sell":
            if position_before > 0:
                sell_frac = self._dynamic_fraction(sell_margin) if config.PARTIAL_SELLS else 1.0
                if forced_exit and getattr(config, "FORCE_FULL_EXIT_ON_RISK", False):
                    sell_frac = 1.0
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
                        self._pos_peak_price = None
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
            elif position_before == 0 and bool(getattr(config, "ENABLE_SHORTS", False)) and cash_before > 0:
                # Open short position.
                pos_frac = self._dynamic_fraction(sell_margin)
                if getattr(config, "ENABLE_ATR_POSITION_SIZING", False):
                    try:
                        atr_col = getattr(config, "ATR_COL", "atr_14")
                        atr = float(row.get(atr_col, 0.0))
                    except Exception:
                        atr = 0.0
                    stop_pct = float(getattr(config, "STOP_LOSS_PCT", 0.0))
                    if getattr(config, "USE_ATR_EXITS", False) and atr > 0.0 and price_now > 0.0:
                        stop_pct = float(getattr(config, "ATR_STOP_MULT", 0.0)) * (atr / price_now)
                    target_risk = float(getattr(config, "ATR_TARGET_RISK_PCT", 0.0))
                    if stop_pct > 0.0 and target_risk > 0.0:
                        scale = min(1.0, target_risk / stop_pct)
                        pos_frac = max(0.0, pos_frac * scale)
                trade_cash = cash_before * pos_frac
                min_notional = float(getattr(config, 'MIN_TRADE_NOTIONAL', 0.0) or 0.0)
                if min_notional > 0.0 and trade_cash < min_notional:
                    action = "hold"
                    hold_reason = hold_reason or "min_notional"
                    budget_blocked = True
                    trade_cash = 0.0

                fee_paid = trade_cash * config.FEE_RATE
                slippage_paid = trade_cash * config.SLIPPAGE_RATE
                net_proceeds = trade_cash - fee_paid - slippage_paid
                if net_proceeds > 0 and trade_cash > 0:
                    trade_executed = True
                    notional_traded = trade_cash
                    trade_size = trade_cash / price_now

                    prior_pos = self.portfolio.position
                    prior_proceeds = abs(prior_pos) * self.portfolio.entry_price
                    new_pos = prior_pos - trade_size

                    total_proceeds = prior_proceeds + net_proceeds
                    self.portfolio.entry_price = total_proceeds / max(abs(new_pos), 1e-9)

                    self.portfolio.position = new_pos
                    self.portfolio.cash = cash_before + net_proceeds

                    if prior_pos == 0:
                        self.portfolio.entry_value = value_before
                        self._buy_legs_current = 0

                    self.last_trade_step = self.steps
                    self.last_entry_step = self.steps
                else:
                    action = "hold"
                    hold_reason = hold_reason or "budget"
                    budget_blocked = True
                    fee_paid = 0.0
                    turnover_penalty = 0.0
                    slippage_paid = 0.0
            else:
                action = "hold"
                hold_reason = hold_reason or "position"
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

        # --- edge-gate supervised update -----------------------------------
        if getattr(config, "ENABLE_EDGE_GATE", False):
            if self._edge_gate is None:
                gate_mode = str(getattr(config, "EDGE_GATE_MODEL", "ridge")).lower()
                if gate_mode == "logistic":
                    self._edge_gate = EdgeGateClassifier(
                        len(INDICATOR_COLUMNS) + 1,
                        lr=float(getattr(config, "EDGE_GATE_LR", 0.01)),
                        l2=float(getattr(config, "EDGE_GATE_L2", 0.0)),
                        decay=float(getattr(config, "EDGE_GATE_DECAY", 1.0)),
                    )
                    self._edge_gate_mode = "logistic"
                else:
                    self._edge_gate = EdgeGateModel(
                        len(INDICATOR_COLUMNS) + 1,
                        ridge=float(getattr(config, "EDGE_GATE_RIDGE", 1.0)),
                        decay=float(getattr(config, "EDGE_GATE_DECAY", 1.0)),
                    )
                    self._edge_gate_mode = "ridge"
            target_price = float(future_price) if future_price is not None else price_next
            target_return = (target_price / max(price_now, 1e-9)) - 1.0
            x = self._edge_gate_features(row)
            if self._edge_gate_mode == "logistic":
                min_abs = float(getattr(config, "EDGE_GATE_TARGET_MIN_ABS", 0.0))
                if abs(target_return) >= min_abs:
                    y = 1.0 if target_return > 0.0 else 0.0
                    self._edge_gate.update(x, float(y))
                    self._edge_gate_steps += 1
            else:
                clip = float(getattr(config, "EDGE_GATE_TARGET_CLIP", 0.0))
                if clip > 0.0:
                    target_return = _clamp(target_return, -clip, clip)
                self._edge_gate.update(x, float(target_return))
                self._edge_gate_steps += 1

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
        elif self.portfolio.position < 0:
            next_allowed.append("buy")
        else:
            if self.portfolio.cash > 0:
                next_allowed.append("buy")
                if getattr(config, "ENABLE_SHORTS", False):
                    next_allowed.append("sell")

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
            self.history.append(
                (
                    step_idx,
                    action,
                    price_now,
                    reward,
                    float(edge_margin),
                    float(realized_pnl),
                    str(proposed_action),
                    str(hold_reason or ""),
                    float(row.get("rv1m_pct_5m", 0.0)),
                    float(row.get("spread_pct_5m", 0.0)),
                    float(row.get("ret_1m", 0.0)),
                    float(row.get("aggr_imb", 0.0)),
                )
            )
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
            exit_leg = (action == "sell" and position_before > 0) or (action == "buy" and position_before < 0)
            if exit_leg:
                self.sell_legs += 1  # count exits consistently for win-rate
                if forced_exit:
                    self.forced_exit_count += 1
                    if forced_exit_reason:
                        self._forced_exit_reason_counter[forced_exit_reason] += 1
                    if forced_exit_reason == "stop_loss":
                        self.last_stop_loss_step = self.steps
                # Per-leg PnL accounting (supports partial exits):
                # - use entry price BEFORE the exit (captured during execution)
                # - win/loss stats are based on per-leg net PnL, not just full closes
                entry_price_for_leg = locals().get("entry_price_for_leg", self.portfolio.entry_price)
                leg_qty = float(trade_size)
                if position_before > 0:
                    gross_leg_pnl = leg_qty * (price_now - entry_price_for_leg)
                else:
                    gross_leg_pnl = leg_qty * (entry_price_for_leg - price_now)
                leg_pnl_net = gross_leg_pnl - fee_paid - slippage_paid
                if leg_pnl_net > 0:
                    self.winning_sell_legs += 1
                    self._win_pnl_sum += float(leg_pnl_net)
                    self._win_pnl_count += 1
                else:
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
            forced_exit=forced_exit,
            forced_exit_reason=forced_exit_reason,
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
        header = [
            "step",
            "action",
            "price",
            "reward",
            "edge_margin",
            "realized_pnl",
            "proposed_action",
            "hold_reason",
            "rv1m_pct_5m",
            "spread_pct_5m",
            "ret_1m",
            "aggr_imb",
            "data_is_live",
        ]
        write_header = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            for (
                step_idx,
                action,
                price,
                reward,
                edge_margin,
                realized_pnl,
                proposed_action,
                hold_reason,
                rv1m_pct_5m,
                spread_pct_5m,
                ret_1m,
                aggr_imb,
            ) in new_trades:
                writer.writerow(
                    [
                        step_idx,
                        action,
                        price,
                        reward,
                        edge_margin,
                        realized_pnl,
                        proposed_action,
                        hold_reason,
                        rv1m_pct_5m,
                        spread_pct_5m,
                        ret_1m,
                        aggr_imb,
                        data_is_live,
                    ]
                )

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
            "forced_exit_count": int(getattr(self, 'forced_exit_count', 0)),
            "forced_exit_reason_counts": dict(getattr(self, '_forced_exit_reason_counter', {})),
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
            "stuck_action_counts": dict(self._stuck_action_counter),
            "stuck_proposed_action_counts": dict(self._stuck_proposed_action_counter),
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
            last_stop_loss_step=int(getattr(self, 'last_stop_loss_step', -1)),
            trend_exit_dir=getattr(self, '_trend_exit_dir', None),
            trend_exit_streak=int(getattr(self, '_trend_exit_streak', 0)),
            macro_lock_dir_candidate=getattr(self, '_macro_lock_dir_candidate', None),
            macro_lock_dir_streak=int(getattr(self, '_macro_lock_dir_streak', 0)),
            macro_lock_effective_dir=getattr(self, '_macro_lock_effective_dir', "neutral"),
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
            edge_horizon = max(1, int(getattr(config, "EDGE_GATE_TARGET_HORIZON", 1) or 1))
            future_idx = min(idx + edge_horizon, len(frame) - 1)
            future_price = float(frame.iloc[future_idx]["close"])
            price = float(row["close"])
            before_trade_value = self.portfolio.value(price)
            result = self.step(row, next_row, idx, train=True, future_price=future_price)
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
        trainer.last_stop_loss_step = int(getattr(state, 'last_stop_loss_step', -1))
        trainer._trend_exit_dir = getattr(state, 'trend_exit_dir', None)
        trainer._trend_exit_streak = int(getattr(state, 'trend_exit_streak', 0) or 0)
        trainer._macro_lock_dir_candidate = getattr(state, 'macro_lock_dir_candidate', None)
        trainer._macro_lock_dir_streak = int(getattr(state, 'macro_lock_dir_streak', 0) or 0)
        trainer._macro_lock_effective_dir = getattr(state, 'macro_lock_effective_dir', "neutral")
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
