# src/agent.py
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
    feature_clip: float = config.FEATURE_CLIP
    weight_clip: float = config.WEIGHT_CLIP
    ridge_factor: float = config.RIDGE_FACTOR
    forgetting_factor: float = config.FORGETTING_FACTOR
    posterior_scale: float = config.POSTERIOR_SCALE
    indicator_columns: list[str] = field(default_factory=lambda: FEATURE_COLUMNS[:])

    @classmethod
    def default(cls) -> "AgentState":
        size = len(FEATURE_COLUMNS)
        return cls(
            q_values=[0.0, 0.0, 0.0],
            weights=[[0.0 for _ in FEATURE_COLUMNS] for _ in ACTIONS],
            cov_inv_matrices=[(np.eye(size, dtype=float) / config.RIDGE_FACTOR).tolist() for _ in ACTIONS],
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
        size = len(FEATURE_COLUMNS)
        # Backward compatibility: older checkpoints stored an `alpha` exploration
        # parameter instead of `posterior_scale`.
        if "alpha" in data and "posterior_scale" not in data:
            data["posterior_scale"] = data.get("alpha", config.POSTERIOR_SCALE)
        data.setdefault("weights", [[0.0 for _ in FEATURE_COLUMNS] for _ in ACTIONS])
        data.setdefault("cov_inv_matrices", [(np.eye(size, dtype=float) / config.RIDGE_FACTOR).tolist() for _ in ACTIONS])
        data.setdefault("bias_vectors", [[0.0 for _ in FEATURE_COLUMNS] for _ in ACTIONS])
        data.setdefault("q_values", [0.0, 0.0, 0.0])
        data.setdefault("steps_seen", 0)
        data.setdefault("last_epsilon", config.POSTERIOR_SCALE)
        data.setdefault("trades", 0)
        data.setdefault("total_reward", 0.0)
        data.setdefault("version", 4)
        for k in ("run_id", "symbol", "timeframe", "saved_at_utc"):
            data.setdefault(k, "")
        data.setdefault("feature_clip", config.FEATURE_CLIP)
        data.setdefault("weight_clip", config.WEIGHT_CLIP)
        data.setdefault("ridge_factor", config.RIDGE_FACTOR)
        data.setdefault("forgetting_factor", config.FORGETTING_FACTOR)
        data.setdefault("posterior_scale", config.POSTERIOR_SCALE)
        data.setdefault("indicator_columns", FEATURE_COLUMNS[:])
        # Ignore any unknown fields from legacy state files.
        allowed = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)


class BanditAgent:
    """Linear contextual bandit with Thompson sampling; now supports optional TD(0) targets."""

    def __init__(self, *, posterior_scale: float | None = None, forgetting_factor: float | None = None):
        self.state = AgentState.from_json(Path(config.STATE_PATH))
        self.posterior_scale = posterior_scale if posterior_scale is not None else self.state.posterior_scale
        self.state.posterior_scale = self.posterior_scale
        self.forgetting_factor = (
            forgetting_factor if forgetting_factor is not None else self.state.forgetting_factor
        )
        self.state.forgetting_factor = self.forgetting_factor
        self._prepare_state()

    # --- utilities ------------------------------------------------------------

    @staticmethod
    def _sanitize(features: np.ndarray) -> np.ndarray:
        # why: indicators can spike; keep dot products numerically safe
        return np.clip(np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0, copy=False),
                       -config.FEATURE_CLIP, config.FEATURE_CLIP)

    def _clip_weights(self, weights: list[list[float]]) -> list[list[float]]:
        return np.clip(np.asarray(weights, dtype=float), -config.WEIGHT_CLIP, config.WEIGHT_CLIP).tolist()

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
            w = cov_inv @ bias
            self.state.weights[idx] = np.clip(w, -config.WEIGHT_CLIP, config.WEIGHT_CLIP).tolist()

    def _migrate_weights(self) -> None:
        # align persisted state with current feature set
        new_cols = FEATURE_COLUMNS
        new_size = len(new_cols)
        old_cols = list(self.state.indicator_columns or [])
        col_map = {name: i for i, name in enumerate(old_cols)}

        mapped_bias = np.zeros((len(ACTIONS), new_size), dtype=float)
        mapped_weights = np.zeros((len(ACTIONS), new_size), dtype=float)

        for a in range(len(ACTIONS)):
            bias_src = np.asarray(self.state.bias_vectors[a], dtype=float) if a < len(self.state.bias_vectors) else np.zeros(len(old_cols))
            w_src = np.asarray(self.state.weights[a], dtype=float) if a < len(self.state.weights) else np.zeros(len(old_cols))
            for j, name in enumerate(new_cols):
                oi = col_map.get(name)
                if oi is None:
                    continue
                if oi < bias_src.shape[0]:
                    mapped_bias[a, j] = bias_src[oi]
                if oi < w_src.shape[0]:
                    mapped_weights[a, j] = w_src[oi]

        base_cov_inv = (np.eye(new_size, dtype=float) / (self.state.ridge_factor if self.state.ridge_factor > 0 else 1.0)).tolist()
        self.state.cov_inv_matrices = [base_cov_inv for _ in ACTIONS]
        self.state.bias_vectors = mapped_bias.tolist()
        self.state.weights = self._clip_weights(mapped_weights.tolist())
        self.state.indicator_columns = new_cols[:]
        self._feature_size = new_size
        self._recompute_weights()

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

    # --- scoring / acting -----------------------------------------------------

    def _estimate_rewards(self, features: np.ndarray) -> np.ndarray:
        f = self._sanitize(features)
        return np.dot(np.asarray(self.state.weights), f)

    def _thompson_scores(self, features: np.ndarray, *, scale: float) -> tuple[np.ndarray, np.ndarray]:
        f = self._sanitize(features)
        means = np.dot(np.asarray(self.state.weights), f)
        draws: list[float] = []
        for a in range(len(ACTIONS)):
            mean = np.asarray(self.state.weights[a], dtype=float)
            cov_inv = np.asarray(self.state.cov_inv_matrices[a], dtype=float)
            cov = cov_inv * max(scale, 0.0)
            cov = 0.5 * (cov + cov.T)
            if scale <= 0:
                w = mean
            else:
                jitter = np.eye(self._feature_size, dtype=float) * 1e-6
                try:
                    w = np.random.multivariate_normal(mean, cov)
                except np.linalg.LinAlgError:
                    w = np.random.multivariate_normal(mean, cov + jitter)
            draws.append(float(np.dot(w, f)))
        return np.asarray(draws), means

    @staticmethod
    def _half_life_decay(step: int | None) -> float:
        if not step or config.POSTERIOR_DECAY_HALF_LIFE_STEPS <= 0:
            return 1.0
        return 0.5 ** (step / float(config.POSTERIOR_DECAY_HALF_LIFE_STEPS))

    def act_with_scores(
        self,
        features: np.ndarray,
        *,
        allowed: list[str] | None = None,
        step: int | None = None,
        posterior_scale_override: float | None = None,
    ) -> tuple[str, np.ndarray, np.ndarray]:
        actions = allowed if allowed is not None else ACTIONS
        base = posterior_scale_override if posterior_scale_override is not None else self.posterior_scale
        scale = max(base * self._half_life_decay(step), config.POSTERIOR_SCALE_MIN)
        sampled, means = self._thompson_scores(features, scale=scale)
        self.state.q_values = list(means)
        self.state.last_epsilon = scale

        idxs = [ACTIONS.index(a) for a in actions]
        best_val = np.max(sampled[idxs])
        cands = [a for a in actions if sampled[ACTIONS.index(a)] == best_val]
        choice = str(np.random.choice(cands))
        return choice, sampled, means

    def act(
        self,
        features: np.ndarray,
        *,
        allowed: list[str] | None = None,
        step: int | None = None,
        posterior_scale_override: float | None = None,
    ) -> str:
        action, _, _ = self.act_with_scores(
            features, allowed=allowed, step=step, posterior_scale_override=posterior_scale_override
        )
        return action

    # --- learning -------------------------------------------------------------

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
        x = self._sanitize(features)
        a_idx = ACTIONS.index(action)

        # TD(0) target if enabled; else immediate reward
        y = reward
        if config.USE_TD and next_features is not None and allowed_next:
            next_means = self._estimate_rewards(next_features)
            next_idxs = [ACTIONS.index(a) for a in allowed_next]
            boot = float(np.max(next_means[next_idxs])) if len(next_idxs) else 0.0
            y = reward + config.TD_GAMMA * boot  # why: propagate credit forward

        cov_inv = np.asarray(self.state.cov_inv_matrices[a_idx], dtype=float)
        x_col = x.reshape(-1, 1)
        denom = float(1.0 + (x_col.T @ cov_inv @ x_col))
        cov_update = (cov_inv @ x_col @ x_col.T @ cov_inv) / denom
        new_cov_inv = cov_inv - cov_update
        new_bias = np.asarray(self.state.bias_vectors[a_idx], dtype=float) + y * x

        self.state.cov_inv_matrices[a_idx] = new_cov_inv.tolist()
        self.state.bias_vectors[a_idx] = new_bias.tolist()
        self._recompute_weights(a_idx)
        self.state.q_values = list(self._estimate_rewards(x))
        self.state.total_reward += reward if actual_reward is None else actual_reward
        if trade_executed:
            self.state.trades += 1
        self.state.steps_seen += 1
    def save(self, *, run_dir: Path | None = None, checkpoint: bool = False,
             keep_last: int = 5) -> Path:
        target_dir = run_dir or Path(config.STATE_PATH).parent
        target_dir.mkdir(parents=True, exist_ok=True)
        self.state.saved_at_utc = datetime.utcnow().isoformat()

        latest_path = target_dir / "agent_state_latest.json"
        self.state.to_json(latest_path)
        pointer = Path(config.STATE_PATH)
        pointer.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(pointer, asdict(self.state))

        saved = latest_path
        if checkpoint:
            ck = target_dir / f"agent_state_step_{self.state.steps_seen}.json"
            self.state.to_json(ck)
            self._prune_checkpoints(target_dir, keep_last)
            saved = ck
        return saved

    def _prune_checkpoints(self, directory: Path, keep_last: int) -> None:
        ckpts: list[Path] = []
        for cand in directory.glob("agent_state_step_*.json"):
            try:
                int(cand.stem.split("_")[-1])
            except Exception:
                continue
            ckpts.append(cand)
        ckpts.sort(key=lambda p: int(p.stem.split("_")[-1]))
        keep_last = max(0, keep_last)
        for old in ckpts[:-keep_last]:
            try:
                old.unlink()
            except OSError:
                pass


class RLSForgettingAgent(BanditAgent):
    """Recursive least squares agent with exponential forgetting."""

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
        x = self._sanitize(features)
        a_idx = ACTIONS.index(action)

        y = reward
        if config.USE_TD and next_features is not None and allowed_next:
            next_means = self._estimate_rewards(next_features)
            next_idxs = [ACTIONS.index(a) for a in allowed_next]
            boot = float(np.max(next_means[next_idxs])) if len(next_idxs) else 0.0
            y = reward + config.TD_GAMMA * boot

        lam = float(self.forgetting_factor)
        if lam <= 0:
            lam = 1e-6
        elif lam > 1:
            lam = 1.0

        cov_inv = np.asarray(self.state.cov_inv_matrices[a_idx], dtype=float)
        scaled_precision = cov_inv / lam
        x_col = x.reshape(-1, 1)
        denom = float(1.0 + (x_col.T @ scaled_precision @ x_col))
        cov_update = (scaled_precision @ x_col @ x_col.T @ scaled_precision) / denom
        new_cov_inv = scaled_precision - cov_update

        bias_prev = np.asarray(self.state.bias_vectors[a_idx], dtype=float)
        new_bias = (bias_prev * lam) + (y * x)

        self.state.cov_inv_matrices[a_idx] = new_cov_inv.tolist()
        self.state.bias_vectors[a_idx] = new_bias.tolist()
        self._recompute_weights(a_idx)
        self.state.q_values = list(self._estimate_rewards(x))
        self.state.total_reward += reward if actual_reward is None else actual_reward
        if trade_executed:
            self.state.trades += 1
        self.state.steps_seen += 1
