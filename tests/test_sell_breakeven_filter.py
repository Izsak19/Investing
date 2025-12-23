import math
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
        # Tiny margins so no strong-edge bypass.
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


class FixedFractionTrainer(Trainer):
    def __init__(self, agent, fraction: float):
        super().__init__(agent, initial_cash=1000.0, min_cash=0.0)
        self._fraction = fraction

    def _dynamic_fraction(self, margin: float) -> float:
        return self._fraction


class SellBreakEvenFilterTest(unittest.TestCase):
    def test_sell_requires_break_even_profit(self) -> None:
        # Save/override config knobs that impact gating.
        orig_gap, orig_hold = config.MIN_TRADE_GAP_STEPS, config.MIN_HOLD_STEPS
        orig_min_pct = config.MIN_PROFIT_TO_SELL_PCT
        orig_min_mult = config.MIN_PROFIT_TO_SELL_MULT_OF_COST
        try:
            config.MIN_TRADE_GAP_STEPS = 0
            config.MIN_HOLD_STEPS = 0
            config.MIN_PROFIT_TO_SELL_PCT = 0.0
            config.MIN_PROFIT_TO_SELL_MULT_OF_COST = 1.0

            agent = DummyAgent(actions=["buy", "sell", "sell"])
            trainer = FixedFractionTrainer(agent, fraction=1.0)

            fee = config.FEE_RATE
            slip = config.SLIPPAGE_RATE
            est_roundtrip = (2 * (fee + slip)) + config.GATE_SAFETY_MARGIN
            entry_est = 100.0 / (1 - (fee + slip))  # buy entry price includes frictions
            breakeven_price = entry_est * (1 + est_roundtrip)
            prices = [
                100.0,  # buy
                100.0 * (1 + 0.0005),  # below breakeven -> should HOLD
                breakeven_price + 0.2,  # comfortably above breakeven -> should SELL
                breakeven_price + 0.2,
            ]

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

            # Buy
            trainer.step(frame.iloc[0], frame.iloc[1], 0, train=True)
            # First sell attempt should be blocked by break-even filter.
            first_sell = trainer.step(frame.iloc[1], frame.iloc[2], 1, train=True)
            self.assertEqual(first_sell.action, "hold")
            self.assertEqual(first_sell.hold_reason, "sell_breakeven")
            self.assertFalse(first_sell.trade_executed)
            # Second sell clears breakeven and should execute.
            second_sell = trainer.step(frame.iloc[2], frame.iloc[3], 2, train=True)
            self.assertEqual(second_sell.action, "sell")
            self.assertTrue(second_sell.trade_executed)
        finally:
            config.MIN_TRADE_GAP_STEPS = orig_gap
            config.MIN_HOLD_STEPS = orig_hold
            config.MIN_PROFIT_TO_SELL_PCT = orig_min_pct
            config.MIN_PROFIT_TO_SELL_MULT_OF_COST = orig_min_mult


if __name__ == "__main__":
    unittest.main()
