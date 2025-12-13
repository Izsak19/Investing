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
    trainer_reward: float
    scaled_reward: float
    trade_executed: bool
    fee_paid: float
    turnover_penalty: float
    refilled: bool
    realized_pnl: float

@dataclass
class TrainerState:
    version: int = 2
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
    price_scale = max(price, 1e-6)
    vals: list[float] = []
    for col, v in zip(INDICATOR_COLUMNS, raw):
        if col in {"ma","ema","wma","boll_mid","boll_upper","boll_lower","vwap","sar","supertrend"}:
            vals.append((v - price) / price_scale)
        elif col == "atr":
            vals.append(v / price_scale)
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
    def __init__(self, agent: BanditAgent, initial_cash: float = config.INITIAL_CASH,
                 min_cash: float = config.MIN_TRAINING_CASH, timeframe: str | None = None):
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

    # --- helpers --------------------------------------------------------------

    def _log_action(self, action: str) -> None:
        if len(self._recent_actions) == self._recent_actions.maxlen:
            oldest = self._recent_actions[0]
            self._action_counter[oldest] -= 1
            if self._action_counter[oldest] <= 0:
                del self._action_counter[oldest]
        self._recent_actions.append(action)
        self._action_counter[action] += 1

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

    def _maybe_refill_portfolio(self) -> bool:
        # why: prevent learning stalls after burn-down
        if self.portfolio.position > 0:
            return False
        if self.portfolio.cash < self.min_cash:
            self.portfolio.cash = self.initial_cash
            self.portfolio.entry_price = 0.0
            self.portfolio.entry_value = 0.0
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
        refilled = self._maybe_refill_portfolio()

        features = build_features(row, self.portfolio)
        allowed_actions = ["hold"]
        gap_ok = self.last_trade_step < 0 or (self.steps - self.last_trade_step) >= config.MIN_TRADE_GAP_STEPS
        can_sell = (self.portfolio.position > 0) and gap_ok and (self.steps - self.last_entry_step) >= config.MIN_HOLD_STEPS
        if can_sell:
            allowed_actions.append("sell")
        if self.portfolio.cash > 0 and gap_ok:
            allowed_actions.append("buy")

        action, sampled_scores, means = self.agent.act_with_scores(
            features, allowed=allowed_actions, step=self.steps, posterior_scale_override=posterior_scale_override
        )

        # --- edge gate vs costs (use model means, not noisy samples) -----------
        mean_buy = means[ACTIONS.index("buy")]
        mean_hold = means[ACTIONS.index("hold")]
        mean_sell = means[ACTIONS.index("sell")]
        buy_margin = mean_buy - mean_hold
        sell_margin = mean_sell - mean_hold
        if action == "buy" and buy_margin < config.EDGE_THRESHOLD:
            action = "hold"
        if action == "sell" and sell_margin < config.EDGE_THRESHOLD:
            action = "hold"

        trade_executed = False
        fee_paid = 0.0
        turnover_penalty = 0.0
        realized_pnl = 0.0
        notional_traded = 0.0

        value_before = self.portfolio.value(price_now)
        position_before = self.portfolio.position
        cash_before = self.portfolio.cash

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

        # --- reward & penalties ------------------------------------------------
        value_next = self.portfolio.value(price_next)
        self._log_action(action)
        reward = value_next - value_before
        step_return = reward / max(value_before, 1e-6)
        self._return_history.append(step_return)
        self._turnover_window.append(notional_traded)

        prospective_curve = self._equity_curve + [value_next]
        drawdown = compute_max_drawdown(prospective_curve)
        drawdown_over = max(0.0, drawdown - config.DRAWDOWN_BUDGET)
        dd_penalty_value = drawdown_over * self.initial_cash * config.DRAWDOWN_PENALTY

        turnover_budget = self.initial_cash * config.TURNOVER_BUDGET_MULTIPLIER
        turnover_over = max(0.0, sum(self._turnover_window) - turnover_budget)
        to_penalty_value = (turnover_over / max(turnover_budget, 1e-6)) * self.initial_cash * config.TURNOVER_BUDGET_PENALTY
        risk_penalty_value = dd_penalty_value + to_penalty_value

        trainer_reward = reward - risk_penalty_value
        pct = trainer_reward / max(self.initial_cash, 1e-6)
        scaled_reward = math.tanh(pct * config.REWARD_SCALE)

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
        )

    # NOTE: run(), _flush_trades_and_metrics(), _persist_trades(), _persist_metrics(), _save_trainer_state()
    # remain identical to your current version.

