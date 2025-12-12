from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.agent import BanditAgent
from src import config
from src.indicators import INDICATOR_COLUMNS


@dataclass
class Portfolio:
    cash: float = 1000.0
    position: float = 0.0
    entry_price: float = 0.0

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


class Trainer:
    def __init__(
        self,
        agent: BanditAgent,
        initial_cash: float = config.INITIAL_CASH,
        min_cash: float = config.MIN_TRAINING_CASH,
    ):
        self.agent = agent
        self.portfolio = Portfolio(cash=initial_cash)
        self.history: List[Tuple[int, str, float, float]] = []  # step, action, price, reward
        self.total_steps: int = 0
        self.positive_steps: int = 0
        self.initial_cash = initial_cash
        self.min_cash = min_cash
        self.refill_count = 0
        self.total_fee_paid = 0.0
        self.total_turnover_penalty_paid = 0.0
        self.steps = 0
        self.sell_trades = 0
        self.winning_sells = 0

    def _build_features(self, row: pd.Series) -> np.ndarray:
        price = float(row["close"])
        raw_features = row[INDICATOR_COLUMNS].to_numpy(dtype=float)
        price_scale = max(price, 1e-6)
        feature_values: list[float] = []

        for col, value in zip(INDICATOR_COLUMNS, raw_features):
            if col in {
                "ma",
                "ema",
                "wma",
                "boll_mid",
                "boll_upper",
                "boll_lower",
                "vwap",
                "sar",
                "supertrend",
            }:
                feature_values.append((value - price) / price_scale)
            elif col == "atr":
                feature_values.append(value / price_scale)
            elif col == "trix":
                feature_values.append(value / 100.0)
        features = np.clip(np.asarray(feature_values, dtype=float), -config.FEATURE_CLIP, config.FEATURE_CLIP)
        return features

    def step(self, row: pd.Series, next_row: pd.Series, step_idx: int) -> StepResult:
        price_now = float(row["close"])
        price_next = float(next_row["close"])
        refilled = self._maybe_refill_portfolio()
        features = self._build_features(row)
        allowed_actions = ["hold"]
        if self.portfolio.position > 0:
            allowed_actions.append("sell")
        if self.portfolio.cash > 0:
            allowed_actions.append("buy")

        action = self.agent.act(features, allowed=allowed_actions, step=self.steps)
        trade_executed = False
        fee_paid = 0.0
        turnover_penalty = 0.0

        # Naive execution model. Rewards are always computed on net proceeds
        # after fees so the agent learns the true cost of transacting. Buying
        # does not deliver an immediate reward, but the eventual sell reward
        # incorporates both the buy and sell fees because the cost basis is
        # fee-adjusted.
        if action == "buy" and self.portfolio.cash > 0:
            trade_executed = True
            fee_paid = self.portfolio.cash * config.FEE_RATE
            investable = self.portfolio.cash - fee_paid
            turnover_penalty = investable * config.TURNOVER_PENALTY
            self.portfolio.position = investable / price_now
            # Track effective cost basis per unit including the buy fee
            self.portfolio.entry_price = price_now / (1 - config.FEE_RATE)
            self.portfolio.cash = -turnover_penalty
        elif action == "sell" and self.portfolio.position > 0:
            trade_executed = True
            gross_proceeds = self.portfolio.position * price_now
            fee_paid = gross_proceeds * config.FEE_RATE
            net_proceeds = gross_proceeds - fee_paid
            turnover_penalty = gross_proceeds * config.TURNOVER_PENALTY
            net_after_penalty = net_proceeds - turnover_penalty
            self.portfolio.cash = net_after_penalty
            self.portfolio.position = 0.0
            self.portfolio.entry_price = 0.0

        value_now = self.portfolio.value(price_now)
        value_next = self.portfolio.value(price_next)
        reward = value_next - value_now

        # Normalize reward by account value so updates reflect percentage returns
        # and stay bounded during long runs.
        denominator = max(abs(value_now), config.INITIAL_CASH, 1e-6)
        scaled_reward = math.tanh(reward / denominator)
        next_features = self._build_features(next_row)
        next_allowed_actions = ["hold"]
        if self.portfolio.position > 0:
            next_allowed_actions.append("sell")
        if self.portfolio.cash > 0:
            next_allowed_actions.append("buy")
        self.agent.update(
            action,
            scaled_reward,
            features,
            actual_reward=reward,
            trade_executed=trade_executed,
            next_features=next_features,
            allowed_next=next_allowed_actions,
        )
        self.history.append((step_idx, action, price_now, reward))
        self.total_steps += 1
        if reward > 0:
            self.positive_steps += 1
        self.steps += 1
        self.total_fee_paid += fee_paid
        self.total_turnover_penalty_paid += turnover_penalty
        if action == "sell" and trade_executed:
            self.sell_trades += 1
            if reward > 0:
                self.winning_sells += 1

        return StepResult(
            action=action,
            trainer_reward=reward,
            scaled_reward=scaled_reward,
            trade_executed=trade_executed,
            fee_paid=fee_paid,
            turnover_penalty=turnover_penalty,
            refilled=refilled,
        )

    def _maybe_refill_portfolio(self) -> bool:
        """
        Reset the paper trading balance after the agent burns through its cash.

        Early in training the policy can be poor and quickly deplete the
        portfolio. When the agent is out of cash and has no open position it
        cannot take further actions that produce rewards, which stalls
        learning. Replenishing the paper account keeps exploration going while
        still letting the agent experience the consequences of bad trades.
        """

        if self.portfolio.position > 0:
            return False

        if self.portfolio.cash < self.min_cash:
            self.portfolio.cash = self.initial_cash
            self.portfolio.entry_price = 0.0
            self.refill_count += 1
            return True
        return False

    @property
    def success_rate(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return (self.positive_steps / self.total_steps) * 100

    @property
    def step_win_rate(self) -> float:
        return self.success_rate

    @property
    def trade_win_rate(self) -> float:
        return self.winning_sells / max(1, self.sell_trades)

    def run(self, frame: pd.DataFrame, max_steps: int | None = None) -> None:
        steps = max_steps if max_steps is not None else len(frame)
        effective_steps = min(steps, max(0, len(frame) - 1))
        for offset in range(effective_steps):
            row = frame.iloc[offset]
            next_row = frame.iloc[offset + 1]
            self.step(row, next_row, frame.index[offset])
        self.agent.save()
        self._persist_trades()

    def _persist_trades(self) -> None:
        path = Path(config.TRADE_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["step", "action", "price", "reward"])
            for row in self.history:
                writer.writerow(row)
