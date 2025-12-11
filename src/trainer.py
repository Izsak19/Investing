from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.agent import BanditAgent
from src import config


@dataclass
class Portfolio:
    cash: float = 1000.0
    position: float = 0.0
    entry_price: float = 0.0

    def value(self, price: float) -> float:
        return self.cash + self.position * price


class Trainer:
    def __init__(self, agent: BanditAgent, initial_cash: float = 1000.0):
        self.agent = agent
        self.portfolio = Portfolio(cash=initial_cash)
        self.history: List[Tuple[int, str, float, float]] = []  # step, action, price, reward
        self.total_trades: int = 0
        self.successful_trades: int = 0

    def step(self, row: pd.Series, step_idx: int) -> None:
        price = float(row["close"])
        action = self.agent.act()
        reward = 0.0

        # Naive execution model
        if action == "buy" and self.portfolio.cash > 0:
            self.portfolio.position = self.portfolio.cash / price
            self.portfolio.entry_price = price
            self.portfolio.cash = 0.0
        elif action == "sell" and self.portfolio.position > 0:
            self.portfolio.cash = self.portfolio.position * price
            reward = self.portfolio.cash - self.portfolio.entry_price * self.portfolio.position
            self.portfolio.position = 0.0
            self.portfolio.entry_price = 0.0
        else:
            reward = (price - self.portfolio.entry_price) * self.portfolio.position

        self.agent.update(action, reward)
        self.history.append((step_idx, action, price, reward))
        self.total_trades += 1
        if reward > 0:
            self.successful_trades += 1

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
