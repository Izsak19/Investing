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
        self.total_trades: int = 0
        self.successful_trades: int = 0
        self.initial_cash = initial_cash
        self.min_cash = min_cash
        self.refill_count = 0
        self.total_fee_paid = 0.0
        self.total_turnover_penalty_paid = 0.0
        self.steps = 0
        self.sell_trades = 0
        self.winning_sells = 0
        self.prev_price: float | None = None
        self.prev_value: float | None = None

    def step(self, row: pd.Series, step_idx: int) -> StepResult:
        price = float(row["close"])
        refilled = self._maybe_refill_portfolio()
        if self.prev_price is None or refilled:
            self.prev_price = price
            self.prev_value = self.portfolio.value(price)

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

        allowed_actions = ["hold"]
        if self.portfolio.position > 0:
            allowed_actions.append("sell")
        if self.portfolio.cash > 0:
            allowed_actions.append("buy")

        action = self.agent.act(features, allowed=allowed_actions, step=self.steps)
        reward = 0.0
        trade_executed = False
        fee_paid = 0.0
        turnover_penalty = 0.0

        # Naive execution model. Rewards are always computed on net proceeds
        # after fees so the agent learns the true cost of transacting. Buying
        # does not deliver an immediate reward, but the eventual sell reward
        # incorporates both the buy and sell fees because the cost basis is
        # fee-adjusted.
        value_before = self.portfolio.value(self.prev_price)

        if action == "buy" and self.portfolio.cash > 0:
            trade_executed = True
            fee_paid = self.portfolio.cash * config.FEE_RATE
            investable = self.portfolio.cash - fee_paid
            turnover_penalty = investable * config.TURNOVER_PENALTY
            self.portfolio.position = investable / price
            # Track effective cost basis per unit including the buy fee
            self.portfolio.entry_price = price / (1 - config.FEE_RATE)
            self.portfolio.cash = -turnover_penalty
        elif action == "sell" and self.portfolio.position > 0:
            trade_executed = True
            gross_proceeds = self.portfolio.position * price
            fee_paid = gross_proceeds * config.FEE_RATE
            net_proceeds = gross_proceeds - fee_paid
            turnover_penalty = gross_proceeds * config.TURNOVER_PENALTY
            net_after_penalty = net_proceeds - turnover_penalty
            self.portfolio.cash = net_after_penalty
            self.portfolio.position = 0.0
            self.portfolio.entry_price = 0.0

        value_after = self.portfolio.value(price)
        reward = value_after - value_before

        # Normalize reward by account value so updates reflect percentage returns
        # and stay bounded during long runs.
        denominator = max(abs(value_before), config.INITIAL_CASH, 1e-6)
        scaled_reward = math.tanh(reward / denominator)
        self.agent.update(
            action,
            scaled_reward,
            features,
            actual_reward=reward,
            trade_executed=trade_executed,
        )
        self.prev_price = price
        self.prev_value = value_after
        self.history.append((step_idx, action, price, reward))
        self.total_trades += 1
        if reward > 0:
            self.successful_trades += 1
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
        if self.total_trades == 0:
            return 0.0
        return (self.successful_trades / self.total_trades) * 100

    @property
    def trade_win_rate(self) -> float:
        return self.winning_sells / max(1, self.sell_trades)

    def run(self, frame: pd.DataFrame, max_steps: int | None = None) -> None:
        steps = max_steps if max_steps is not None else len(frame)
        for idx, row in frame.head(steps).iterrows():
            self.step(row, idx)
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
