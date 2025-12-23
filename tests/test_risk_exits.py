import unittest

import numpy as np
import pandas as pd

from src import config
from src.agent import ACTIONS, AgentState
from src.trainer import Trainer


class DummyAgent:
    def __init__(self, actions: list[str]):
        self.actions = actions[:]
        self.state = AgentState.default()
        self.posterior_scale = 0.0

    def act_with_scores(self, features: np.ndarray, *, allowed=None, step=None, posterior_scale_override=None):
        action = self.actions.pop(0)
        scores = np.zeros(len(ACTIONS))
        scores[ACTIONS.index(action)] = 0.01
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


class FixedFractionTrainer(Trainer):
    def __init__(self, agent, fraction: float):
        super().__init__(agent, initial_cash=1000.0, min_cash=0.0)
        self._fraction = fraction

    def _dynamic_fraction(self, margin: float) -> float:
        return self._fraction


class RiskExitTest(unittest.TestCase):
    def test_stop_loss_and_trailing_tp_force_exit(self) -> None:
        orig_gap, orig_hold = config.MIN_TRADE_GAP_STEPS, config.MIN_HOLD_STEPS
        orig_stop = config.STOP_LOSS_PCT
        orig_tp = config.TAKE_PROFIT_PCT
        orig_trail = config.TRAILING_TP_PCT
        orig_use_trail = config.USE_TRAILING_TP
        orig_force_full = config.FORCE_FULL_EXIT_ON_RISK
        try:
            config.MIN_TRADE_GAP_STEPS = 0
            config.MIN_HOLD_STEPS = 0
            config.STOP_LOSS_PCT = 0.003
            # Keep TP above the path so trailing TP triggers on the retrace.
            config.TAKE_PROFIT_PCT = 0.01
            config.TRAILING_TP_PCT = 0.003
            config.USE_TRAILING_TP = True
            config.FORCE_FULL_EXIT_ON_RISK = True

            prices = [100.0, 99.6, 100.0, 100.6, 100.2, 100.2]
            data = {
                "timestamp": list(range(len(prices))),
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume": [1.0] * len(prices),
            }
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

            agent = DummyAgent(actions=["buy", "hold", "buy", "hold", "hold"])
            trainer = FixedFractionTrainer(agent, fraction=1.0)

            # Entry 1
            trainer.step(frame.iloc[0], frame.iloc[1], 0, train=True)
            pos_before_stop = trainer.portfolio.position
            stop_exit = trainer.step(frame.iloc[1], frame.iloc[2], 1, train=True)
            self.assertEqual(stop_exit.action, "sell")
            self.assertTrue(stop_exit.trade_executed)
            self.assertTrue(stop_exit.forced_exit)
            self.assertEqual(stop_exit.forced_exit_reason, "stop_loss")
            self.assertAlmostEqual(stop_exit.trade_size, pos_before_stop, places=8)
            self.assertAlmostEqual(trainer.portfolio.position, 0.0, places=8)

            # Entry 2
            trainer.step(frame.iloc[2], frame.iloc[3], 2, train=True)
            pos_before_trail = trainer.portfolio.position
            # Price peaks then retraces to trigger trailing TP.
            trainer.step(frame.iloc[3], frame.iloc[4], 3, train=True)
            trail_exit = trainer.step(frame.iloc[4], frame.iloc[5], 4, train=True)
            self.assertEqual(trail_exit.action, "sell")
            self.assertTrue(trail_exit.trade_executed)
            self.assertTrue(trail_exit.forced_exit)
            self.assertEqual(trail_exit.forced_exit_reason, "trailing_tp")
            self.assertAlmostEqual(trail_exit.trade_size, pos_before_trail, places=8)
            self.assertAlmostEqual(trainer.portfolio.position, 0.0, places=8)
        finally:
            config.MIN_TRADE_GAP_STEPS = orig_gap
            config.MIN_HOLD_STEPS = orig_hold
            config.STOP_LOSS_PCT = orig_stop
            config.TAKE_PROFIT_PCT = orig_tp
            config.TRAILING_TP_PCT = orig_trail
            config.USE_TRAILING_TP = orig_use_trail
            config.FORCE_FULL_EXIT_ON_RISK = orig_force_full


if __name__ == "__main__":
    unittest.main()
