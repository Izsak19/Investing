from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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

    def step(self, row: pd.Series, step_idx: int) -> None:
        price = float(row["close"])
        self._maybe_refill_portfolio()
        features = row[INDICATOR_COLUMNS].to_numpy(dtype=float)
        action = self.agent.act(features)
        reward = 0.0

        # Naive execution model. Rewards are always computed on net proceeds
        # after fees so the agent learns the true cost of transacting. Buying
        # does not deliver an immediate reward, but the eventual sell reward
        # incorporates both the buy and sell fees because the cost basis is
        # fee-adjusted.
        if action == "buy" and self.portfolio.cash > 0:
            fee = self.portfolio.cash * config.FEE_RATE
            investable = self.portfolio.cash - fee
            self.portfolio.position = investable / price
            # Track effective cost basis per unit including the buy fee
            self.portfolio.entry_price = price / (1 - config.FEE_RATE)
            self.portfolio.cash = 0.0
        elif action == "sell" and self.portfolio.position > 0:
            gross_proceeds = self.portfolio.position * price
            fee = gross_proceeds * config.FEE_RATE
            net_proceeds = gross_proceeds - fee
            reward = net_proceeds - self.portfolio.entry_price * self.portfolio.position
            self.portfolio.cash = net_proceeds
            self.portfolio.position = 0.0
            self.portfolio.entry_price = 0.0
        else:
            reward = (price - self.portfolio.entry_price) * self.portfolio.position

        self.agent.update(action, reward, features)
        self.history.append((step_idx, action, price, reward))
        self.total_trades += 1
        if reward > 0:
            self.successful_trades += 1

    def _maybe_refill_portfolio(self) -> None:
        """
        Reset the paper trading balance after the agent burns through its cash.

        Early in training the policy can be poor and quickly deplete the
        portfolio. When the agent is out of cash and has no open position it
        cannot take further actions that produce rewards, which stalls
        learning. Replenishing the paper account keeps exploration going while
        still letting the agent experience the consequences of bad trades.
        """

        if self.portfolio.position > 0:
            return

        if self.portfolio.cash < self.min_cash:
            self.portfolio.cash = self.initial_cash
            self.portfolio.entry_price = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.successful_trades / self.total_trades) * 100

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
