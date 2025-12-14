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
    fee_paid: float
    turnover_penalty: float
    refilled: bool
    realized_pnl: float
    edge_margin: float
    hold_reason: str | None
    gate_blocked: bool = False
    timing_blocked: bool = False
    budget_blocked: bool = False
    stuck_relax: bool = False

@dataclass
class TrainerState:
    version: int = 4
    run_id: str = ""
    steps: int = 0
    prev_price: float | None = None
    prev_value: float | None = None
    refill_count: int = 0
    total_fee_paid: float = 0.0
    total_turnover_penalty_paid: float = 0.0
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
    gate_blocks: int = 0
    timing_blocks: int = 0
    budget_blocks: int = 0
    penalty_profile: str = "train"
    reward_scale: float = config.REWARD_SCALE
    drawdown_budget: float = config.DRAWDOWN_BUDGET
    turnover_budget_multiplier: float = config.TURNOVER_BUDGET_MULTIPLIER
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
    raw = row[INDICATOR_COLUMNS].to_numpy(dtype=float)
    scale_frac = float(row.get("feature_scale_frac", 0.0))
    feature_scale = float(row.get("feature_scale", 0.0))
    if not math.isfinite(scale_frac):
        scale_frac = 0.0
    if not math.isfinite(feature_scale) or feature_scale <= 0:
        feature_scale = 0.0
    scale_guess = price * max(scale_frac, 0.01)
    price_scale = max(feature_scale, scale_guess, 1e-6)
    atr_scale = max(price_scale, feature_scale)
    vals: list[float] = []
    for col, v in zip(INDICATOR_COLUMNS, raw):
        if col in {"ma","ema","wma","boll_mid","boll_upper","boll_lower","vwap","sar","supertrend"}:
            vals.append((v - price) / price_scale)
        elif col == "atr":
            vals.append(v / max(atr_scale, 1e-6))
        elif col == "trix":
            vals.append(v / 100.0)
    pos_flag = 1.0 if portfolio.position > 0 else 0.0
    portfolio_value = portfolio.value(price)
    cash_frac = portfolio.cash / max(portfolio_value, 1e-6)
    unrealized_ret = (price / portfolio.entry_price) - 1.0 if portfolio.position > 0 and portfolio.entry_price > 0 else 0.0
    position_value = portfolio.position * price
    pos_frac = position_value / max(portfolio_value, 1e-6)
    vals.extend([pos_flag, cash_frac, unrealized_ret, 1.0, pos_frac])
    return np.clip(np.asarray(vals, dtype=float), -config.FEATURE_CLIP, config.FEATURE_CLIP)

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
        self.steps = 0
        self.sell_trades = 0
        self.winning_sells = 0
        self._last_price: float | None = None
        self._last_value: float | None = None
        self.last_trade_step = -1
        self.last_entry_step = -1
        self.last_data_is_live: bool | None = None
        self._equity_curve: list[float] = []
        self._return_history: deque[float] = deque(maxlen=config.RETURN_HISTORY_WINDOW)
        self._recent_actions: deque[str] = deque(maxlen=config.ACTION_HISTORY_WINDOW)
        self._action_counter: Counter[str] = Counter()
        self._turnover_window: deque[float] = deque(maxlen=config.TURNOVER_BUDGET_WINDOW)
        self.timeframe = timeframe or agent.state.timeframe or config.DEFAULT_TIMEFRAME
        self.periods_per_year = periods_per_year(self.timeframe)
        self.penalty_profile = penalty_profile
        self.gate_blocks = 0
        self.timing_blocks = 0
        self.budget_blocks = 0
        self._hold_reason_log: list[tuple[int, str, str]] = []
        # adaptive knobs (start from config values)
        self.reward_scale = config.REWARD_SCALE
        self.drawdown_budget = config.DRAWDOWN_BUDGET
        self.turnover_budget_multiplier = config.TURNOVER_BUDGET_MULTIPLIER

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
        self._turnover_window.clear()
        self.steps = 0
        self.total_steps = 0
        self.positive_steps = 0
        self.total_fee_paid = 0.0
        self.total_turnover_penalty_paid = 0.0
        self.sell_trades = 0
        self.winning_sells = 0
        self.gate_blocks = 0
        self.timing_blocks = 0
        self.budget_blocks = 0
        self._hold_reason_log.clear()
        self.reward_scale = config.REWARD_SCALE
        self.drawdown_budget = config.DRAWDOWN_BUDGET
        self.turnover_budget_multiplier = config.TURNOVER_BUDGET_MULTIPLIER

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
        return self.winning_sells / max(1, self.sell_trades)

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
        price = self._last_price if self._last_price is not None else 0.0
        return total_return(self.initial_cash, self.portfolio.value(price))

    # --- core step ------------------------------------------------------------

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

        features = build_features(row, self.portfolio)
        allowed_actions = ["hold"]
        if self.portfolio.cash > 0:
            allowed_actions.append("buy")
        if self.portfolio.position > 0:
            allowed_actions.append("sell")

        posterior_scale_effective, edge_threshold, stuck_relax = self._stuck_adaptation(
            posterior_scale_override
        )

        action, sampled_scores, means = self.agent.act_with_scores(
            features,
            allowed=allowed_actions,
            step=self.steps,
            posterior_scale_override=posterior_scale_effective,
        )

        proposed_action = action
        hold_idx = ACTIONS.index("hold")
        margin_scale = max(len(INDICATOR_COLUMNS) * config.WEIGHT_CLIP, 1e-9)
        buy_margin_raw = sampled_scores[ACTIONS.index("buy")] - sampled_scores[hold_idx]
        sell_margin_raw = sampled_scores[ACTIONS.index("sell")] - sampled_scores[hold_idx]
        buy_margin = math.tanh(buy_margin_raw / margin_scale)
        sell_margin = math.tanh(sell_margin_raw / margin_scale)
        edge_margin = buy_margin if proposed_action == "buy" else sell_margin if proposed_action == "sell" else 0.0

        hold_reason: str | None = None
        gate_blocked = False
        timing_blocked = False
        budget_blocked = False
        warmup_active = self.agent.state.trades < config.WARMUP_TRADES_BEFORE_GATING

        if not warmup_active:
            if action == "buy" and buy_margin < edge_threshold:
                action = "hold"
                hold_reason = "gate"
                gate_blocked = True
                self.gate_blocks += 1
            elif action == "sell" and sell_margin < edge_threshold:
                action = "hold"
                hold_reason = "gate"
                gate_blocked = True
                self.gate_blocks += 1

        gap_ok = self.last_trade_step < 0 or (self.steps - self.last_trade_step) >= config.MIN_TRADE_GAP_STEPS
        hold_ok = self.last_entry_step < 0 or (self.steps - self.last_entry_step) >= config.MIN_HOLD_STEPS

        if not warmup_active:
            if action == "buy" and not gap_ok:
                action = "hold"
                hold_reason = hold_reason or "timing_gap"
                timing_blocked = True
                self.timing_blocks += 1
            if action == "sell" and (not gap_ok or not hold_ok):
                action = "hold"
                hold_reason = hold_reason or "timing_hold"
                timing_blocked = True
                self.timing_blocks += 1

        trade_executed = False
        fee_paid = 0.0
        turnover_penalty = 0.0
        realized_pnl = 0.0
        notional_traded = 0.0

        value_before = self.portfolio.value(price_now)
        position_before = self.portfolio.position
        cash_before = self.portfolio.cash

        if action == "hold" and stuck_relax and proposed_action in allowed_actions:
            # When the agent is clearly stuck in HOLD, allow one-off execution of the
            # originally proposed action (buy/sell) even if gates would normally
            # block it. This keeps exploration alive instead of freezing.
            if proposed_action == "buy" and cash_before > 0:
                action = "buy"
                hold_reason = None
                gate_blocked = False
                timing_blocked = False
                budget_blocked = False
            elif proposed_action == "sell" and position_before > 0:
                action = "sell"
                hold_reason = None
                gate_blocked = False
                timing_blocked = False
                budget_blocked = False

        # --- execution (fees applied; turnover penalty applied) ----------------
        if action == "buy" and cash_before > 0:
            pos_frac = self._dynamic_fraction(buy_margin)
            trade_cash = cash_before * pos_frac
            fee_paid = trade_cash * config.FEE_RATE
            investable_base = trade_cash - fee_paid
            turnover_penalty = investable_base * config.TURNOVER_PENALTY
            investable = investable_base - turnover_penalty
            gross_outlay = trade_cash + turnover_penalty
            if investable > 0 and gross_outlay <= cash_before:
                trade_executed = True
                notional_traded = gross_outlay
                trade_size = investable / price_now
                prior_pos = self.portfolio.position
                prior_cost = prior_pos * self.portfolio.entry_price
                new_pos = prior_pos + trade_size
                total_cost = prior_cost + gross_outlay
                self.portfolio.entry_price = total_cost / max(new_pos, 1e-9)  # why: track fee-adjusted basis
                self.portfolio.position = new_pos
                self.portfolio.cash = cash_before - gross_outlay
                if prior_pos == 0:
                    self.portfolio.entry_value = value_before
                self.last_trade_step = self.steps
                self.last_entry_step = self.steps
            else:
                action = "hold"
                hold_reason = hold_reason or "budget"
                budget_blocked = True
                fee_paid = 0.0
                turnover_penalty = 0.0

        elif action == "sell" and position_before > 0:
            sell_frac = self._dynamic_fraction(sell_margin) if config.PARTIAL_SELLS else 1.0
            trade_size = position_before * min(1.0, max(0.0, sell_frac))
            if trade_size > 0:
                trade_executed = True
                gross_proceeds = trade_size * price_now
                fee_paid = gross_proceeds * config.FEE_RATE
                turnover_penalty = gross_proceeds * config.TURNOVER_PENALTY
                notional_traded = gross_proceeds
                net = gross_proceeds - fee_paid - turnover_penalty
                self.portfolio.cash += net
                self.portfolio.position = position_before - trade_size
                fully_closed = self.portfolio.position <= 1e-12
                if fully_closed:
                    realized_pnl = self.portfolio.cash - self.portfolio.entry_value
                    self.portfolio.entry_price = 0.0
                    self.portfolio.entry_value = 0.0
                    self.last_entry_step = -1
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

        # --- reward & penalties ------------------------------------------------
        value_next = self.portfolio.value(price_next)
        self._log_action(action)
        reward = value_next - value_before
        step_return = reward / max(value_before, 1e-6)
        self._return_history.append(step_return)
        self._turnover_window.append(notional_traded)
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
        risk_penalty_value = dd_penalty_value + to_penalty_value

        trainer_reward = reward - risk_penalty_value
        pct = trainer_reward / max(self.initial_cash, 1e-6)
        scaled_reward = math.tanh(pct * self.reward_scale)

        # --- learning ---------------------------------------------------------
        next_features = build_features(next_row, self.portfolio)
        next_allowed = ["hold"]
        if self.portfolio.position > 0:
            next_allowed.append("sell")
        if self.portfolio.cash > 0:
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
        self._equity_curve.append(value_next)

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
            fee_paid=fee_paid,
            turnover_penalty=turnover_penalty,
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
            "gate_blocks": self.gate_blocks,
            "timing_blocks": self.timing_blocks,
            "budget_blocks": self.budget_blocks,
            "data_is_live": data_is_live,
            "baseline_final_value": baseline_final_value,
            "val_final_value": val_final_value,
            "ma_baseline_final_value": ma_baseline_final_value,
            "executed_trades": executed_trades if executed_trades is not None else len(self.history),
            "reward_scale": self.reward_scale,
            "drawdown_budget": self.drawdown_budget,
            "turnover_budget_multiplier": self.turnover_budget_multiplier,
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
            gate_blocks=self.gate_blocks,
            timing_blocks=self.timing_blocks,
            budget_blocks=self.budget_blocks,
            penalty_profile=self.penalty_profile,
            reward_scale=self.reward_scale,
            drawdown_budget=self.drawdown_budget,
            turnover_budget_multiplier=self.turnover_budget_multiplier,
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

            # simple return history update
            if mtm_delta != 0:
                pnl = mtm_delta - result.fee_paid - result.turnover_penalty
                base = max(before_trade_value, 1e-6)
                self._return_history.append(pnl / base)

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
        trainer.sell_trades = state.sell_trades
        trainer.winning_sells = state.winning_sells
        trainer.last_trade_step = state.last_trade_step
        trainer.last_entry_step = state.last_entry_step
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


