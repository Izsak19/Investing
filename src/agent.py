from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

from src import config
from src.indicators import INDICATOR_COLUMNS
from src.persistence import atomic_write_json


ACTIONS = ["sell", "hold", "buy"]
FEATURE_COLUMNS = INDICATOR_COLUMNS + ["pos_flag", "cash_frac", "unrealized_ret"]


@dataclass
class AgentState:
    q_values: List[float]
    weights: List[List[float]]
    total_reward: float = 0.0
    trades: int = 0
    steps_seen: int = 0
    last_epsilon: float = 0.0
    version: int = 2
    run_id: str = ""
    symbol: str = ""
    timeframe: str = ""
    saved_at_utc: str = ""
    alpha: float = config.ALPHA
    feature_clip: float = config.FEATURE_CLIP
    weight_clip: float = config.WEIGHT_CLIP
    indicator_columns: list[str] = field(default_factory=lambda: FEATURE_COLUMNS[:])

    @classmethod
    def default(cls) -> "AgentState":
        return cls(
            q_values=[0.0, 0.0, 0.0],
            weights=[[0.0 for _ in FEATURE_COLUMNS] for _ in ACTIONS],
            last_epsilon=config.EPSILON_START,
            indicator_columns=FEATURE_COLUMNS[:],
        )

    def to_json(self, path: Path) -> None:
        atomic_write_json(path, asdict(self))

    @classmethod
    def from_json(cls, path: Path) -> "AgentState":
        if not path.exists():
            return cls.default()
        data = json.loads(path.read_text())
        if "weights" not in data:
            data["weights"] = [[0.0 for _ in FEATURE_COLUMNS] for _ in ACTIONS]
        if "q_values" not in data:
            data["q_values"] = [0.0, 0.0, 0.0]
        if "steps_seen" not in data:
            data["steps_seen"] = 0
        if "last_epsilon" not in data:
            data["last_epsilon"] = config.EPSILON_START
        if "trades" not in data:
            data["trades"] = 0
        if "total_reward" not in data:
            data["total_reward"] = 0.0
        data.setdefault("version", 2)
        data.setdefault("run_id", "")
        data.setdefault("symbol", "")
        data.setdefault("timeframe", "")
        data.setdefault("saved_at_utc", "")
        data.setdefault("alpha", config.ALPHA)
        data.setdefault("feature_clip", config.FEATURE_CLIP)
        data.setdefault("weight_clip", config.WEIGHT_CLIP)
        data.setdefault("indicator_columns", FEATURE_COLUMNS[:])
        return cls(**data)


class BanditAgent:
    """Epsilon-greedy multi-armed bandit with additive reward updates."""

    def __init__(self):
        self.state = AgentState.from_json(Path(config.STATE_PATH))
        self._prepare_state()

    @staticmethod
    def _sanitize(features: np.ndarray) -> np.ndarray:
        """Clamp features to a safe numeric range and replace non-finite values.

        Live market indicators can be large (price-denominated) and, over many
        steps, weight updates can explode. Clipping keeps the dot products used
        for Q-value estimates within floating-point limits.
        """

        clipped = np.clip(
            np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0, copy=False),
            -config.FEATURE_CLIP,
            config.FEATURE_CLIP,
        )
        return clipped

    def _ensure_weight_shape(self) -> None:
        if len(self.state.weights) != len(ACTIONS):
            self.state.weights = [[0.0 for _ in range(self._feature_size)] for _ in ACTIONS]
            return

        for i in range(len(self.state.weights)):
            if len(self.state.weights[i]) != self._feature_size:
                self.state.weights[i] = [0.0 for _ in range(self._feature_size)]

    def _clip_weights(self, weights: list[list[float]]) -> list[list[float]]:
        bounded = np.clip(np.asarray(weights, dtype=float), -config.WEIGHT_CLIP, config.WEIGHT_CLIP)
        return bounded.tolist()

    def _migrate_weights(self) -> None:
        """Align persisted weights with the current feature set."""

        new_columns = FEATURE_COLUMNS
        new_weights = np.zeros((len(ACTIONS), len(new_columns)), dtype=float)
        current_columns = list(self.state.indicator_columns or [])
        base_cols = len(current_columns) if current_columns else len(new_columns)
        source = np.asarray(self.state.weights, dtype=float)
        if source.ndim != 2 or source.shape[0] != len(ACTIONS) or source.shape[1] == 0:
            source = np.zeros((len(ACTIONS), base_cols), dtype=float)
        column_map = {name: idx for idx, name in enumerate(current_columns)}
        for new_idx, name in enumerate(new_columns):
            old_idx = column_map.get(name)
            if old_idx is None:
                continue
            if old_idx >= source.shape[1]:
                continue
            new_weights[:, new_idx] = source[:, old_idx]
        clipped = np.clip(new_weights, -config.WEIGHT_CLIP, config.WEIGHT_CLIP)
        self.state.weights = clipped.tolist()
        self.state.indicator_columns = new_columns[:]
        self._feature_size = len(new_columns)

    def _estimate_rewards(self, features: np.ndarray) -> np.ndarray:
        safe_features = self._sanitize(features)
        return np.dot(np.asarray(self.state.weights), safe_features)

    def _prepare_state(self) -> None:
        self._feature_size = len(self.state.indicator_columns) or len(FEATURE_COLUMNS)
        if not self.state.indicator_columns:
            self.state.indicator_columns = FEATURE_COLUMNS[:]
        if self.state.indicator_columns != FEATURE_COLUMNS:
            self._migrate_weights()
        self._feature_size = len(self.state.indicator_columns) or len(FEATURE_COLUMNS)
        self.state.weights = self._clip_weights(self.state.weights)
        self._ensure_weight_shape()

    def current_epsilon(self, step: int | None = None) -> float:
        t = step if step is not None else self.state.steps_seen
        progress = min(1.0, t / config.EPSILON_DECAY_STEPS)
        epsilon = max(
            config.EPSILON_END,
            config.EPSILON_START + (config.EPSILON_END - config.EPSILON_START) * progress,
        )
        return float(epsilon)

    def act(
        self,
        features: np.ndarray,
        *,
        allowed: list[str] | None = None,
        step: int | None = None,
        epsilon_override: float | None = None,
    ) -> str:
        estimates = self._estimate_rewards(features)
        self.state.q_values = list(estimates)

        actions = allowed if allowed is not None else ACTIONS

        epsilon = epsilon_override if epsilon_override is not None else self.current_epsilon(step)
        is_flat = "buy" in actions and "sell" not in actions
        if epsilon_override is None and is_flat and self.state.steps_seen < config.FLAT_EXPLORATION_WARMUP_STEPS:
            epsilon = max(epsilon, config.EPSILON_WHEN_FLAT)
        self.state.last_epsilon = epsilon

        if np.random.random() < epsilon:
            return str(np.random.choice(actions))

        allowed_estimates = {action: estimates[ACTIONS.index(action)] for action in actions}
        best_estimate = max(allowed_estimates.values())
        candidates = [action for action, value in allowed_estimates.items() if value == best_estimate]
        return str(np.random.choice(candidates))

    def update(
        self,
        action: str,
        reward: float,
        features: np.ndarray,
        *,
        actual_reward: float | None = None,
        trade_executed: bool = True,
        next_features: np.ndarray | None = None,
        allowed_next: list[str] | None = None,
    ) -> None:
        safe_features = self._sanitize(features)
        idx = ACTIONS.index(action)
        prediction = float(np.dot(self.state.weights[idx], safe_features))
        target = reward
        if config.USE_TD and next_features is not None:
            safe_next = self._sanitize(next_features)
            next_actions = allowed_next if allowed_next is not None else ACTIONS
            next_idxs = [ACTIONS.index(a) for a in next_actions]
            q_next = float(np.max(np.dot(np.asarray(self.state.weights)[next_idxs], safe_next)))
            target = reward + config.GAMMA * q_next
        error = float(np.clip(target - prediction, -config.ERROR_CLIP, config.ERROR_CLIP))
        updated = np.asarray(self.state.weights[idx]) + config.ALPHA * error * safe_features
        bounded = np.clip(updated, -config.WEIGHT_CLIP, config.WEIGHT_CLIP)
        self.state.weights[idx] = list(bounded)
        self.state.q_values = list(self._estimate_rewards(safe_features))
        self.state.total_reward += reward if actual_reward is None else actual_reward
        if trade_executed:
            self.state.trades += 1
        self.state.steps_seen += 1

    def _prune_checkpoints(self, directory: Path, keep_last: int) -> None:
        checkpoints: list[Path] = []
        for candidate in directory.glob("agent_state_step_*.json"):
            try:
                int(candidate.stem.split("_")[-1])
            except (ValueError, IndexError):
                continue
            checkpoints.append(candidate)
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[-1]))
        if keep_last <= 0:
            keep_last = 0
        for old in checkpoints[:-keep_last]:
            try:
                old.unlink()
            except OSError:
                pass

    def save(
        self,
        *,
        run_dir: Path | None = None,
        checkpoint: bool = False,
        keep_last: int = config.DEFAULT_KEEP_LAST_CHECKPOINTS,
    ) -> Path:
        target_dir = run_dir or Path(config.STATE_PATH).parent
        target_dir.mkdir(parents=True, exist_ok=True)
        self.state.saved_at_utc = datetime.utcnow().isoformat()

        latest_path = target_dir / "agent_state_latest.json"
        self.state.to_json(latest_path)
        pointer_path = Path(config.STATE_PATH)
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(pointer_path, asdict(self.state))

        saved_path = latest_path
        if checkpoint:
            checkpoint_path = target_dir / f"agent_state_step_{self.state.steps_seen}.json"
            self.state.to_json(checkpoint_path)
            saved_path = checkpoint_path
            self._prune_checkpoints(target_dir, keep_last)
        return saved_path
