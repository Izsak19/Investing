import math
import unittest

import numpy as np
import pandas as pd

from src import config
from src.agent import ACTIONS, AgentState
from src.trainer import Trainer


class DummyAgent:
    """Deterministic agent that returns a fixed action sequence."""

    def __init__(self, actions: list[str]):
        self.actions = actions[:]
        self.state = AgentState.default()
        self.posterior_scale = 0.0

    def act_with_scores(self, features: np.ndarray, *, allowed=None, step=None, posterior_scale_override=None):
        action = self.actions.pop(0)
        scores = np.zeros(len(ACTIONS))
        # Keep margins tiny so strong-edge overrides do not force full sells.
        scores[ACTIONS.index(action)] = 0.001
        return action, scores, scores

    def update(
        self,
        action: str,
        reward: float,
        features: np.ndarray,
        *,
        actual_reward: float | None = None,
        trade_executed: bool = True,
        next_features: np.ndarray | None = None,
        allowed_next=None,
    ) -> None:
        if trade_executed:
            self.state.trades += 1
        self.state.steps_seen += 1


class DeterministicTrainer(Trainer):
    """Trainer with deterministic position sizing for predictable leg sizes."""

    def __init__(self, agent, *, fractions: list[float]):
        super().__init__(agent, initial_cash=100.0, min_cash=0.0)
        self._fractions = fractions[:]

    def _dynamic_fraction(self, margin: float) -> float:
        if self._fractions:
            return self._fractions.pop(0)
        return super()._dynamic_fraction(margin)


class SellWinRateTest(unittest.TestCase):
    def test_per_leg_win_rate_counts_partial_sells(self) -> None:
        # Keep cooldowns and gates from blocking deterministic actions.
        orig_gap, orig_hold = config.MIN_TRADE_GAP_STEPS, config.MIN_HOLD_STEPS
        try:
            config.MIN_TRADE_GAP_STEPS = 0
            config.MIN_HOLD_STEPS = 0

            agent = DummyAgent(actions=["buy", "sell", "sell"])
            trainer = DeterministicTrainer(agent, fractions=[1.0, 0.5, 1.0])

            data = {
                "timestamp": [0, 1, 2, 3],
                "open": [100.0, 110.0, 90.0, 90.0],
                "high": [100.0, 110.0, 90.0, 90.0],
                "low": [100.0, 110.0, 90.0, 90.0],
                "close": [100.0, 110.0, 90.0, 90.0],
                "volume": [1.0, 1.0, 1.0, 1.0],
            }
            # Indicator columns expected by build_features; zeros keep the test deterministic.
            for col in (
                "ret_1m",
                "rv_1m",
                "ofi_l1",
                "imb1",
                "micro_bias",
                "rel_spread",
                "aggr_imb",
                "dw_spread",
                "cvd_1m",
                "whale_net_rate_1m",
                "liq_net_rate_1m",
                "oi_delta_1m",
                "basis_pct",
                "funding_x_time",
                "rv1m_pct_5m",
                "spread_pct_5m",
                "tod_sin",
                "tod_cos",
            ):
                data[col] = 0.0
            frame = pd.DataFrame(data)

            for i in range(3):
                row = frame.iloc[i]
                nxt = frame.iloc[i + 1]
                trainer.step(row, nxt, i, train=True)

            self.assertEqual(trainer.sell_legs, 2)
            self.assertEqual(trainer.winning_sell_legs, 1)
            self.assertTrue(math.isclose(trainer.trade_win_rate, 0.5, rel_tol=1e-9))
        finally:
            config.MIN_TRADE_GAP_STEPS = orig_gap
            config.MIN_HOLD_STEPS = orig_hold


if __name__ == "__main__":
    unittest.main()
