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
FEATURE_COLUMNS = INDICATOR_COLUMNS + ["pos_flag", "cash_frac", "unrealized_ret", "bias", "pos_frac"]


@dataclass
class AgentState:
    q_values: List[float]
    weights: List[List[float]]
    cov_inv_matrices: List[List[List[float]]]
    bias_vectors: List[List[float]]
    total_reward: float = 0.0
    trades: int = 0
    steps_seen: int = 0
    last_epsilon: float = 0.0
    version: int = 4
    run_id: str = ""
    symbol: str = ""
    timeframe: str = ""
    saved_at_utc: str = ""
    alpha: float = config.ALPHA
    feature_clip: float = config.FEATURE_CLIP
    weight_clip: float = config.WEIGHT_CLIP
    ridge_factor: float = config.RIDGE_FACTOR
    posterior_scale: float = config.POSTERIOR_SCALE
    indicator_columns: list[str] = field(default_factory=lambda: FEATURE_COLUMNS[:])

    @classmethod
    def default(cls) -> "AgentState":
        return cls(
            q_values=[0.0, 0.0, 0.0],
            weights=[[0.0 for _ in FEATURE_COLUMNS] for _ in ACTIONS],
            cov_inv_matrices=[
                (np.eye(len(FEATURE_COLUMNS), dtype=float) / config.RIDGE_FACTOR).tolist()
                for _ in ACTIONS
            ],
            bias_vectors=[[0.0 for _ in FEATURE_COLUMNS] for _ in ACTIONS],
            last_epsilon=config.POSTERIOR_SCALE,
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
        data.setdefault(
            "cov_inv_matrices",
            [
                (np.eye(len(FEATURE_COLUMNS), dtype=float) / config.RIDGE_FACTOR).tolist()
                for _ in ACTIONS
            ],
        )
        data.setdefault("bias_vectors", [[0.0 for _ in FEATURE_COLUMNS] for _ in ACTIONS])
        if "q_values" not in data:
            data["q_values"] = [0.0, 0.0, 0.0]
        if "steps_seen" not in data:
            data["steps_seen"] = 0
        if "last_epsilon" not in data:
            data["last_epsilon"] = config.POSTERIOR_SCALE
        if "trades" not in data:
            data["trades"] = 0
        if "total_reward" not in data:
            data["total_reward"] = 0.0
        data.setdefault("version", 4)
        data.setdefault("run_id", "")
        data.setdefault("symbol", "")
        data.setdefault("timeframe", "")
        data.setdefault("saved_at_utc", "")
        data.setdefault("alpha", config.ALPHA)
        data.setdefault("feature_clip", config.FEATURE_CLIP)
        data.setdefault("weight_clip", config.WEIGHT_CLIP)
        data.setdefault("ridge_factor", config.RIDGE_FACTOR)
        data.setdefault("posterior_scale", config.POSTERIOR_SCALE)
        data.pop("ucb_scale", None)
        data.setdefault("indicator_columns", FEATURE_COLUMNS[:])
        return cls(**data)


class BanditAgent:
    """Contextual bandit that uses Thompson sampling for exploration."""

    def __init__(
        self,
        *,
        posterior_scale: float | None = None,
    ):
        self.state = AgentState.from_json(Path(config.STATE_PATH))
        self.posterior_scale = (
            posterior_scale if posterior_scale is not None else self.state.posterior_scale
        )
        self.state.posterior_scale = self.posterior_scale
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

    def _clip_weights(self, weights: list[list[float]]) -> list[list[float]]:
        bounded = np.clip(np.asarray(weights, dtype=float), -config.WEIGHT_CLIP, config.WEIGHT_CLIP)
        return bounded.tolist()

    def _ensure_covariance_shape(self) -> None:
        identity_scale = self.state.ridge_factor if self.state.ridge_factor > 0 else 1.0
        base = np.eye(self._feature_size, dtype=float) / identity_scale
        if len(self.state.cov_inv_matrices) != len(ACTIONS):
            self.state.cov_inv_matrices = [base.tolist() for _ in ACTIONS]
            return

        for i in range(len(self.state.cov_inv_matrices)):
            mat = np.asarray(self.state.cov_inv_matrices[i], dtype=float)
            if mat.shape != (self._feature_size, self._feature_size):
                self.state.cov_inv_matrices[i] = base.tolist()

    def _ensure_bias_shape(self) -> None:
        if len(self.state.bias_vectors) != len(ACTIONS):
            self.state.bias_vectors = [[0.0 for _ in range(self._feature_size)] for _ in ACTIONS]
            return

        for i in range(len(self.state.bias_vectors)):
            if len(self.state.bias_vectors[i]) != self._feature_size:
                self.state.bias_vectors[i] = [0.0 for _ in range(self._feature_size)]

    def _recompute_weights(self, action_idx: int | None = None) -> None:
        indices = range(len(ACTIONS)) if action_idx is None else [action_idx]
        for idx in indices:
            cov_inv = np.asarray(self.state.cov_inv_matrices[idx], dtype=float)
            bias = np.asarray(self.state.bias_vectors[idx], dtype=float)
            weights = cov_inv @ bias
            bounded = np.clip(weights, -config.WEIGHT_CLIP, config.WEIGHT_CLIP)
            self.state.weights[idx] = bounded.tolist()

    def _migrate_weights(self) -> None:
        """Align persisted state with the current feature set."""

        new_columns = FEATURE_COLUMNS
        new_size = len(new_columns)
        current_columns = list(self.state.indicator_columns or [])
        column_map = {name: idx for idx, name in enumerate(current_columns)}

        mapped_bias = np.zeros((len(ACTIONS), new_size), dtype=float)
        mapped_weights = np.zeros((len(ACTIONS), new_size), dtype=float)
        for action_idx in range(len(ACTIONS)):
            bias_source = (
                np.asarray(self.state.bias_vectors[action_idx], dtype=float)
                if action_idx < len(self.state.bias_vectors)
                else np.zeros(len(current_columns), dtype=float)
            )
            weight_source = (
                np.asarray(self.state.weights[action_idx], dtype=float)
                if action_idx < len(self.state.weights)
                else np.zeros(len(current_columns), dtype=float)
            )
            for new_idx, name in enumerate(new_columns):
                old_idx = column_map.get(name)
                if old_idx is None:
                    continue
                if old_idx < bias_source.shape[0]:
                    mapped_bias[action_idx, new_idx] = bias_source[old_idx]
                elif old_idx < weight_source.shape[0]:
                    mapped_bias[action_idx, new_idx] = weight_source[old_idx]
                if old_idx < weight_source.shape[0]:
                    mapped_weights[action_idx, new_idx] = weight_source[old_idx]

        identity_scale = self.state.ridge_factor if self.state.ridge_factor > 0 else 1.0
        base_cov_inv = np.eye(new_size, dtype=float) / identity_scale
        self.state.cov_inv_matrices = [base_cov_inv.tolist() for _ in ACTIONS]
        self.state.bias_vectors = mapped_bias.tolist()
        self.state.weights = self._clip_weights(mapped_weights.tolist())
        self.state.indicator_columns = new_columns[:]
        self._feature_size = new_size
        self._recompute_weights()

    def _estimate_rewards(self, features: np.ndarray) -> np.ndarray:
        safe_features = self._sanitize(features)
        return np.dot(np.asarray(self.state.weights), safe_features)

    def _thompson_scores(self, features: np.ndarray, *, scale: float) -> tuple[np.ndarray, np.ndarray]:
        safe_features = self._sanitize(features)
        means = np.dot(np.asarray(self.state.weights), safe_features)
        sampled_scores: list[float] = []

        for idx in range(len(ACTIONS)):
            mean = np.asarray(self.state.weights[idx], dtype=float)
            cov_inv = np.asarray(self.state.cov_inv_matrices[idx], dtype=float)
            covariance = cov_inv * max(scale, 0.0)
            covariance = 0.5 * (covariance + covariance.T)
            if scale <= 0:
                draw = mean
            else:
                jitter = np.eye(self._feature_size, dtype=float) * 1e-6
                try:
                    draw = np.random.multivariate_normal(mean, covariance)
                except np.linalg.LinAlgError:
                    draw = np.random.multivariate_normal(mean, covariance + jitter)
            sampled_scores.append(float(np.dot(draw, safe_features)))

        return np.asarray(sampled_scores), means

    def _prepare_state(self) -> None:
        self._feature_size = len(self.state.indicator_columns) or len(FEATURE_COLUMNS)
        if not self.state.indicator_columns:
            self.state.indicator_columns = FEATURE_COLUMNS[:]
        if self.state.indicator_columns != FEATURE_COLUMNS:
            self._migrate_weights()
        else:
            self._feature_size = len(self.state.indicator_columns) or len(FEATURE_COLUMNS)
        self._ensure_covariance_shape()
        self._ensure_bias_shape()
        if len(self.state.weights) != len(ACTIONS):
            self.state.weights = [[0.0 for _ in range(self._feature_size)] for _ in ACTIONS]
        self._recompute_weights()

    def act(
        self,
        features: np.ndarray,
        *,
        allowed: list[str] | None = None,
        step: int | None = None,
        posterior_scale_override: float | None = None,
    ) -> str:
        actions = allowed if allowed is not None else ACTIONS

        sample_scale = (
            posterior_scale_override
            if posterior_scale_override is not None
            else self.posterior_scale
        )
        sample_scale = max(sample_scale, config.POSTERIOR_SCALE_MIN)
        sampled_scores, means = self._thompson_scores(features, scale=sample_scale)
        self.state.q_values = list(means)
        self.state.last_epsilon = sample_scale

        allowed_scores = {action: sampled_scores[ACTIONS.index(action)] for action in actions}
        best_score = max(allowed_scores.values())
        candidates = [action for action, value in allowed_scores.items() if value == best_score]
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
        cov_inv = np.asarray(self.state.cov_inv_matrices[idx], dtype=float)
        feature_vec = safe_features.reshape(-1, 1)
        denom = float(1.0 + (feature_vec.T @ cov_inv @ feature_vec))
        adjustment = (cov_inv @ feature_vec @ feature_vec.T @ cov_inv) / denom
        updated_cov_inv = cov_inv - adjustment
        updated_bias = np.asarray(self.state.bias_vectors[idx], dtype=float) + reward * safe_features

        self.state.cov_inv_matrices[idx] = updated_cov_inv.tolist()
        self.state.bias_vectors[idx] = updated_bias.tolist()
        self._recompute_weights(idx)
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
