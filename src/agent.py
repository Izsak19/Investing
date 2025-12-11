from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np

from src import config


ACTIONS = ["sell", "hold", "buy"]


@dataclass
class AgentState:
    q_values: List[float]
    total_reward: float = 0.0
    trades: int = 0

    @classmethod
    def default(cls) -> "AgentState":
        return cls(q_values=[0.0, 0.0, 0.0])

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "AgentState":
        if not path.exists():
            return cls.default()
        data = json.loads(path.read_text())
        return cls(**data)


class BanditAgent:
    """Epsilon-greedy multi-armed bandit with additive reward updates."""

    def __init__(self):
        self.state = AgentState.from_json(Path(config.STATE_PATH))

    def act(self) -> str:
        if np.random.random() < config.EPSILON:
            choice = np.random.choice(ACTIONS)
        else:
            choice = ACTIONS[int(np.argmax(self.state.q_values))]
        return choice

    def update(self, action: str, reward: float) -> None:
        idx = ACTIONS.index(action)
        q_old = self.state.q_values[idx]
        self.state.q_values[idx] = q_old + config.ALPHA * (reward - q_old)
        self.state.total_reward += reward
        self.state.trades += 1

    def save(self) -> None:
        self.state.to_json(Path(config.STATE_PATH))
