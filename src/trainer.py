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

from src.agent import AgentState, BanditAgent
from src import config
from src.indicators import INDICATOR_COLUMNS
from src.metrics import compute_max_drawdown, compute_sharpe_ratio, rolling_volatility, total_return
from src.persistence import atomic_write_json


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

    pos_flag = 1.0 if portfolio.position > 0 else 0.0
    portfolio_value = portfolio.value(price)
    cash_frac = portfolio.cash / max(portfolio_value, 1e-6)
    unrealized_ret = (price / portfolio.entry_price) - 1.0 if portfolio.position > 0 and portfolio.entry_price > 0 else 0.0

    position_value = portfolio.position * price
    pos_frac = position_value / max(portfolio_value, 1e-6)

    feature_values.extend([pos_flag, cash_frac, unrealized_ret, 1.0, pos_frac])

    features = np.clip(np.asarray(feature_values, dtype=float), -config.FEATURE_CLIP, config.FEATURE_CLIP)
    return features


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
        self._last_flushed_trade_idx = 0
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
        self._last_price: float | None = None
        self._last_value: float | None = None
        self.last_trade_step = -1
        self.last_entry_step = -1
        self.last_data_is_live: bool | None = None
        self._equity_curve: list[float] = []
        self._return_history: deque[float] = deque(maxlen=config.RISK_VOL_WINDOW)
        self._recent_actions: deque[str] = deque(maxlen=config.ACTION_HISTORY_WINDOW)
        self._action_counter: Counter[str] = Counter()

    def reset_portfolio(self) -> None:
        self.portfolio.cash = self.initial_cash
        self.portfolio.position = 0.0
        self.portfolio.entry_price = 0.0
        self.portfolio.entry_value = 0.0
        self.last_trade_step = -1
        self.last_entry_step = -1
        self._equity_curve.clear()
        self._return_history.clear()

    def export_state(self, run_id: str) -> TrainerState:
        return TrainerState(
            run_id=run_id,
            steps=self.steps,
            prev_price=self._last_price,
            prev_value=self._last_value,
            refill_count=self.refill_count,
            total_fee_paid=self.total_fee_paid,
            total_turnover_penalty_paid=self.total_turnover_penalty_paid,
            total_trades=self.agent.state.trades,
            successful_trades=self.positive_steps,
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
        )

    def import_state(self, state: TrainerState) -> None:
        self.steps = state.steps
        self._last_price = state.prev_price
        self._last_value = state.prev_value
        self.refill_count = state.refill_count
        self.total_fee_paid = state.total_fee_paid
        self.total_turnover_penalty_paid = state.total_turnover_penalty_paid
        self.sell_trades = state.sell_trades
        self.winning_sells = state.winning_sells
        self.portfolio.cash = state.portfolio_cash
        self.portfolio.position = state.portfolio_position
        self.portfolio.entry_price = state.portfolio_entry_price
        self.portfolio.entry_value = state.portfolio_entry_value
        self.total_steps = state.total_steps
        self.positive_steps = state.positive_steps
        self.last_trade_step = state.last_trade_step
        self.last_entry_step = state.last_entry_step

    def _build_features(self, row: pd.Series) -> np.ndarray:
        return build_features(row, self.portfolio)

    def _log_action(self, action: str) -> None:
        if len(self._recent_actions) == self._recent_actions.maxlen:
            oldest = self._recent_actions[0]
            self._action_counter[oldest] -= 1
            if self._action_counter[oldest] <= 0:
                del self._action_counter[oldest]
        self._recent_actions.append(action)
        self._action_counter[action] += 1

    def step(
        self,
        row: pd.Series,
        next_row: pd.Series,
        step_idx: int,
        *,
        train: bool = True,
        epsilon_override: float | None = None,
    ) -> StepResult:
        price_now = float(row["close"])
        price_next = float(next_row["close"])
        refilled = self._maybe_refill_portfolio()
        features = self._build_features(row)
        allowed_actions = ["hold"]
        gap_respected = self.last_trade_step < 0 or (self.steps - self.last_trade_step) >= config.MIN_TRADE_GAP_STEPS
        can_sell = (
            self.portfolio.position > 0
            and gap_respected
            and (self.steps - self.last_entry_step) >= config.MIN_HOLD_STEPS
        )
        if can_sell:
            allowed_actions.append("sell")
        if self.portfolio.cash > 0 and gap_respected:
            allowed_actions.append("buy")

        action = self.agent.act(
            features, allowed=allowed_actions, step=self.steps, epsilon_override=epsilon_override
        )
        trade_executed = False
        fee_paid = 0.0
        turnover_penalty = 0.0
        realized_pnl = 0.0

        value_before = self.portfolio.value(price_now)
        position_before = self.portfolio.position
        cash_before = self.portfolio.cash

        # Naive execution model. Rewards are always computed on net proceeds
        # after fees so the agent learns the true cost of transacting. Buying
        # does not deliver an immediate reward, but the eventual sell reward
        # incorporates both the buy and sell fees because the cost basis is
        # fee-adjusted.
        if action == "buy" and cash_before > 0:
            trade_cash = cash_before * config.POSITION_FRACTION
            fee_paid = trade_cash * config.FEE_RATE
            investable_base = trade_cash - fee_paid
            turnover_penalty = investable_base * config.TURNOVER_PENALTY
            investable = investable_base - turnover_penalty
            if investable > 0:
                trade_executed = True
                trade_size = investable / price_now
                prior_position = self.portfolio.position
                prior_cost_basis = prior_position * self.portfolio.entry_price
                new_position = prior_position + trade_size
                total_cost_basis = prior_cost_basis + trade_cash
                # Track effective cost basis per unit including fees/penalties
                self.portfolio.entry_price = total_cost_basis / max(new_position, 1e-9)
                self.portfolio.position = new_position
                self.portfolio.cash = cash_before - trade_cash
                if prior_position == 0:
                    self.portfolio.entry_value = value_before
                self.last_trade_step = self.steps
                self.last_entry_step = self.steps
            else:
                action = "hold"
                fee_paid = 0.0
                turnover_penalty = 0.0
        elif action == "sell" and position_before > 0:
            trade_executed = True
            gross_proceeds = position_before * price_now
            fee_paid = gross_proceeds * config.FEE_RATE
            turnover_penalty = gross_proceeds * config.TURNOVER_PENALTY
            net = gross_proceeds - fee_paid - turnover_penalty
            self.portfolio.cash = net
            self.portfolio.position = 0.0
            realized_pnl = self.portfolio.cash - self.portfolio.entry_value
            self.portfolio.entry_price = 0.0
            self.portfolio.entry_value = 0.0
            self.last_trade_step = self.steps
            self.last_entry_step = -1

        value_next = self.portfolio.value(price_next)
        self._log_action(action)
        reward = value_next - value_before
        step_return = reward / max(value_before, 1e-6)
        self._return_history.append(step_return)
        vol_penalty = rolling_volatility(self._return_history) * config.RISK_VOL_PENALTY
        risk_penalty_value = vol_penalty * self.initial_cash
        trainer_reward = reward - risk_penalty_value

        # Normalize reward by initial capital to avoid runaway scales on long episodes.
        pct = trainer_reward / max(self.initial_cash, 1e-6)
        scaled_reward = math.tanh(pct * config.REWARD_SCALE)
        next_features = self._build_features(next_row)
        next_allowed_actions = ["hold"]
        if self.portfolio.position > 0:
            next_allowed_actions.append("sell")
        if self.portfolio.cash > 0:
            next_allowed_actions.append("buy")
        if train:
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
            self.portfolio.entry_value = 0.0
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

    @property
    def action_distribution(self) -> dict[str, float]:
        total = sum(self._action_counter.values())
        if total <= 0:
            return {}
        return {action: count / total for action, count in self._action_counter.items()}

    @property
    def sharpe_ratio(self) -> float:
        return compute_sharpe_ratio(list(self._return_history))

    @property
    def max_drawdown(self) -> float:
        return compute_max_drawdown(self._equity_curve)

    @property
    def total_return(self) -> float:
        price = self._last_price if self._last_price is not None else 0.0
        return total_return(self.initial_cash, self.portfolio.value(price))

    def run(
        self,
        frame: pd.DataFrame,
        max_steps: int | None = None,
        *,
        run_id: str,
        run_dir: Path,
        checkpoint_every: int,
        flush_trades_every: int,
        keep_last: int,
        data_is_live: bool | None = None,
    ) -> None:
        steps = max_steps if max_steps is not None else len(frame)
        effective_steps = min(steps, max(0, len(frame) - 1))
        interrupted = False
        self.last_data_is_live = data_is_live
        try:
            for offset in range(effective_steps):
                row = frame.iloc[offset]
                next_row = frame.iloc[offset + 1]
                self.step(row, next_row, frame.index[offset])

                if flush_trades_every > 0 and self.steps % flush_trades_every == 0:
                    self._flush_trades_and_metrics(run_dir, data_is_live=data_is_live)
                if checkpoint_every > 0 and self.steps % checkpoint_every == 0:
                    self.agent.save(run_dir=run_dir, checkpoint=True, keep_last=keep_last)
                    self._save_trainer_state(run_dir, run_id, checkpoint=True, keep_last=keep_last)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            self._flush_trades_and_metrics(run_dir, force=True, data_is_live=data_is_live)
            self.agent.save(run_dir=run_dir, checkpoint=False, keep_last=keep_last)
            self._save_trainer_state(run_dir, run_id, checkpoint=not interrupted, keep_last=keep_last)

    def _flush_trades_and_metrics(
        self,
        run_dir: Path,
        force: bool = False,
        *,
        data_is_live: bool | None = None,
        baseline_final_value: float | None = None,
        val_final_value: float | None = None,
        max_drawdown: float | None = None,
        executed_trades: int | None = None,
        ma_baseline_final_value: float | None = None,
    ) -> None:
        pending = self.history[self._last_flushed_trade_idx :]
        if pending:
            self._persist_trades(run_dir, pending)
            self._last_flushed_trade_idx = len(self.history)
            if self._last_flushed_trade_idx > 0:
                self.history = []
                self._last_flushed_trade_idx = 0
        if force or pending:
            self._persist_metrics(
                run_dir,
                data_is_live=data_is_live,
                baseline_final_value=baseline_final_value,
                val_final_value=val_final_value,
                max_drawdown=max_drawdown,
                executed_trades=executed_trades,
                ma_baseline_final_value=ma_baseline_final_value,
            )

    def _persist_trades(self, run_dir: Path, rows: list[tuple[int, str, float, float]]) -> None:
        path = run_dir / "trades.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["step", "action", "price", "reward"])
            for row in rows:
                writer.writerow(row)

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
        path = run_dir / "metrics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        price = self._last_price if self._last_price is not None else 0.0
        portfolio_value = self.portfolio.value(price)
        total_ret = total_return(self.initial_cash, portfolio_value)
        realized_max_drawdown = max_drawdown if max_drawdown is not None else compute_max_drawdown(self._equity_curve)
        sharpe = compute_sharpe_ratio(list(self._return_history))
        distribution = self.action_distribution
        with path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "wall_time_utc",
                        "steps",
                        "steps_seen",
                        "total_reward",
                        "portfolio_value",
                        "cash",
                        "position",
                        "success_rate",
                        "sell_win_rate",
                        "refill_count",
                        "total_fee_paid",
                        "total_turnover_penalty_paid",
                        "data_is_live",
                        "baseline_final_value",
                        "val_final_value",
                        "max_drawdown",
                        "executed_trades",
                        "ma_baseline_final_value",
                        "total_return",
                        "sharpe_ratio",
                        "action_hold",
                        "action_buy",
                        "action_sell",
                    ]
                )
            writer.writerow(
                [
                    datetime.utcnow().isoformat(),
                    self.steps,
                    self.agent.state.steps_seen,
                    self.agent.state.total_reward,
                    portfolio_value,
                    self.portfolio.cash,
                    self.portfolio.position,
                    self.success_rate,
                    self.trade_win_rate,
                    self.refill_count,
                    self.total_fee_paid,
                    self.total_turnover_penalty_paid,
                    data_is_live if data_is_live is not None else self.last_data_is_live,
                    baseline_final_value,
                    val_final_value,
                    realized_max_drawdown,
                    executed_trades,
                    ma_baseline_final_value,
                    total_ret,
                    sharpe,
                    distribution.get("hold", 0.0),
                    distribution.get("buy", 0.0),
                    distribution.get("sell", 0.0),
                ]
            )

    def _save_trainer_state(
        self,
        run_dir: Path,
        run_id: str,
        *,
        checkpoint: bool,
        keep_last: int,
    ) -> None:
        state = self.export_state(run_id)
        latest_path = run_dir / "trainer_state_latest.json"
        state.to_json(latest_path)
        if checkpoint:
            checkpoint_path = run_dir / f"trainer_state_step_{self.steps}.json"
            state.to_json(checkpoint_path)
            self._prune_trainer_checkpoints(run_dir, keep_last)

    def _prune_trainer_checkpoints(self, run_dir: Path, keep_last: int) -> None:
        checkpoints: list[Path] = []
        for candidate in run_dir.glob("trainer_state_step_*.json"):
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


def load_latest_run(run_dir: Path) -> tuple[AgentState, TrainerState]:
    agent_state = AgentState.from_json(run_dir / "agent_state_latest.json")
    trainer_state = TrainerState.from_json(run_dir / "trainer_state_latest.json")
    return agent_state, trainer_state


def resume_from(run_dir: Path, agent: BanditAgent, trainer: Trainer) -> None:
    agent_state, trainer_state = load_latest_run(run_dir)
    agent.state = agent_state
    agent._prepare_state()
    trainer.import_state(trainer_state)
