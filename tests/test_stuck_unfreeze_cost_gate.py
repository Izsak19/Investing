import math
import unittest

import numpy as np
import pandas as pd

from src import config
from src.agent import ACTIONS, AgentState
from src.indicators import INDICATOR_COLUMNS
from src.trainer import Trainer


class ScriptedAgent:
    def __init__(self, steps: list[tuple[str, float]]):
        self.steps = steps[:]
        self.state = AgentState.default()
        self.posterior_scale = 0.0

    def act_with_scores(self, features: np.ndarray, *, allowed=None, step=None, posterior_scale_override=None):
        action, raw_margin = self.steps.pop(0)
        scores = np.full(len(ACTIONS), -1e-6)
        hold_idx = ACTIONS.index("hold")
        scores[hold_idx] = 0.0
        if action == "buy":
            scores[ACTIONS.index("buy")] = scores[hold_idx] + raw_margin
        elif action == "sell":
            scores[ACTIONS.index("sell")] = scores[hold_idx] + raw_margin
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


class StuckUnfreezeCostGateTest(unittest.TestCase):
    def test_stuck_unfreeze_does_not_bypass_cost_gate(self) -> None:
        orig_gap = config.MIN_TRADE_GAP_STEPS
        orig_hold = config.MIN_HOLD_STEPS
        orig_warmup = config.WARMUP_TRADES_BEFORE_GATING
        orig_stuck = config.ENABLE_STUCK_UNFREEZE
        orig_window = config.STUCK_HOLD_WINDOW
        orig_ratio = config.STUCK_HOLD_RATIO
        orig_boost = config.STUCK_POSTERIOR_BOOST
        orig_allow_buy = getattr(config, "STUCK_ALLOW_BUY", False)
        orig_max_legs = config.MAX_BUY_LEGS_PER_POSITION
        orig_edge = config.EDGE_THRESHOLD
        orig_cost_mult = config.COST_EDGE_MULT
        orig_edge_margin = config.EDGE_SAFETY_MARGIN
        orig_gate_margin = config.GATE_SAFETY_MARGIN
        orig_cost_gate = config.COST_AWARE_GATING
        try:
            config.MIN_TRADE_GAP_STEPS = 0
            config.MIN_HOLD_STEPS = 0
            config.WARMUP_TRADES_BEFORE_GATING = 0
            config.ENABLE_STUCK_UNFREEZE = True
            config.STUCK_HOLD_WINDOW = 50
            config.STUCK_HOLD_RATIO = 0.9
            config.STUCK_POSTERIOR_BOOST = 0.0
            config.STUCK_ALLOW_BUY = True
            config.MAX_BUY_LEGS_PER_POSITION = 2
            config.EDGE_THRESHOLD = 0.0
            config.COST_EDGE_MULT = 2.0
            config.EDGE_SAFETY_MARGIN = 0.0
            config.GATE_SAFETY_MARGIN = 0.0
            config.COST_AWARE_GATING = True

            est_cost = config.FEE_RATE + config.SLIPPAGE_RATE + config.GATE_SAFETY_MARGIN
            old_edge = est_cost + config.EDGE_SAFETY_MARGIN
            new_edge = max(config.EDGE_THRESHOLD, config.COST_EDGE_MULT * est_cost) + config.EDGE_SAFETY_MARGIN
            target_margin = 0.5 * (old_edge + new_edge)
            margin_scale = config.WEIGHT_CLIP * float(getattr(config, "MARGIN_SCALE_MULT", 1.0))
            raw_margin = math.atanh(target_margin) * margin_scale

            margin_scale = config.WEIGHT_CLIP * float(getattr(config, "MARGIN_SCALE_MULT", 1.0))
            entry_margin = margin_scale * 3.0
            steps = [("buy", entry_margin)] + [("hold", 0.0)] * 25 + [("buy", raw_margin)]
            agent = ScriptedAgent(steps=steps)
            trainer = FixedFractionTrainer(agent, fraction=1.0)

            rows = len(steps) + 1
            prices = [100.0] * rows
            data = {
                "timestamp": list(range(rows)),
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume": [1.0] * rows,
            }
            for col in INDICATOR_COLUMNS:
                data[col] = 0.0
            frame = pd.DataFrame(data)

            for i in range(26):
                trainer.step(frame.iloc[i], frame.iloc[i + 1], i, train=True)

            result = trainer.step(frame.iloc[26], frame.iloc[27], 26, train=True)
            self.assertTrue(result.stuck_relax)
            self.assertEqual(result.action, "hold")
            self.assertFalse(result.trade_executed)
        finally:
            config.MIN_TRADE_GAP_STEPS = orig_gap
            config.MIN_HOLD_STEPS = orig_hold
            config.WARMUP_TRADES_BEFORE_GATING = orig_warmup
            config.ENABLE_STUCK_UNFREEZE = orig_stuck
            config.STUCK_HOLD_WINDOW = orig_window
            config.STUCK_HOLD_RATIO = orig_ratio
            config.STUCK_POSTERIOR_BOOST = orig_boost
            config.STUCK_ALLOW_BUY = orig_allow_buy
            config.MAX_BUY_LEGS_PER_POSITION = orig_max_legs
            config.EDGE_THRESHOLD = orig_edge
            config.COST_EDGE_MULT = orig_cost_mult
            config.EDGE_SAFETY_MARGIN = orig_edge_margin
            config.GATE_SAFETY_MARGIN = orig_gate_margin
            config.COST_AWARE_GATING = orig_cost_gate

    def test_stuck_unfreeze_blocks_buy_actions(self) -> None:
        orig_gap = config.MIN_TRADE_GAP_STEPS
        orig_hold = config.MIN_HOLD_STEPS
        orig_warmup = config.WARMUP_TRADES_BEFORE_GATING
        orig_stuck = config.ENABLE_STUCK_UNFREEZE
        orig_window = config.STUCK_HOLD_WINDOW
        orig_ratio = config.STUCK_HOLD_RATIO
        orig_allow_buy = getattr(config, "STUCK_ALLOW_BUY", False)
        orig_max_legs = config.MAX_BUY_LEGS_PER_POSITION
        orig_edge = config.EDGE_THRESHOLD
        orig_cost_mult = config.COST_EDGE_MULT
        orig_edge_margin = config.EDGE_SAFETY_MARGIN
        orig_gate_margin = config.GATE_SAFETY_MARGIN
        orig_cost_gate = config.COST_AWARE_GATING
        try:
            config.MIN_TRADE_GAP_STEPS = 0
            config.MIN_HOLD_STEPS = 0
            config.WARMUP_TRADES_BEFORE_GATING = 0
            config.ENABLE_STUCK_UNFREEZE = True
            config.STUCK_HOLD_WINDOW = 30
            config.STUCK_HOLD_RATIO = 0.9
            config.STUCK_ALLOW_BUY = False
            config.MAX_BUY_LEGS_PER_POSITION = 2
            config.EDGE_THRESHOLD = 0.0
            config.COST_EDGE_MULT = 1.0
            config.EDGE_SAFETY_MARGIN = 0.0
            config.GATE_SAFETY_MARGIN = 0.0
            config.COST_AWARE_GATING = True

            margin_scale = config.WEIGHT_CLIP * float(getattr(config, "MARGIN_SCALE_MULT", 1.0))
            raw_margin = margin_scale * 3.0

            steps = [("buy", raw_margin)] + [("hold", 0.0)] * 25 + [("buy", raw_margin)]
            agent = ScriptedAgent(steps=steps)
            trainer = FixedFractionTrainer(agent, fraction=1.0)

            rows = len(steps) + 1
            prices = [100.0] * rows
            data = {
                "timestamp": list(range(rows)),
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume": [1.0] * rows,
            }
            for col in INDICATOR_COLUMNS:
                data[col] = 0.0
            frame = pd.DataFrame(data)

            for i in range(26):
                trainer.step(frame.iloc[i], frame.iloc[i + 1], i, train=True)

            result = trainer.step(frame.iloc[26], frame.iloc[27], 26, train=True)
            self.assertTrue(result.stuck_relax)
            self.assertEqual(result.action, "hold")
            self.assertFalse(result.trade_executed)
        finally:
            config.MIN_TRADE_GAP_STEPS = orig_gap
            config.MIN_HOLD_STEPS = orig_hold
            config.WARMUP_TRADES_BEFORE_GATING = orig_warmup
            config.ENABLE_STUCK_UNFREEZE = orig_stuck
            config.STUCK_HOLD_WINDOW = orig_window
            config.STUCK_HOLD_RATIO = orig_ratio
            config.STUCK_ALLOW_BUY = orig_allow_buy
            config.MAX_BUY_LEGS_PER_POSITION = orig_max_legs
            config.EDGE_THRESHOLD = orig_edge
            config.COST_EDGE_MULT = orig_cost_mult
            config.EDGE_SAFETY_MARGIN = orig_edge_margin
            config.GATE_SAFETY_MARGIN = orig_gate_margin
            config.COST_AWARE_GATING = orig_cost_gate


if __name__ == "__main__":
    unittest.main()
