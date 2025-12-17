from __future__ import annotations

from typing import Iterable

from rich.console import Console
from rich.live import Live
from rich.table import Table

from src import config
from src.agent import BanditAgent
from src.trainer import Portfolio, StepResult

console = Console()


def build_table(
    step: int,
    price: float,
    result: StepResult,
    portfolio_value: float,
    mtm_delta: float,
    trade_impact: float,
    portfolio: Portfolio,
    agent: BanditAgent,
    step_win_rate: float,
    refill_count: int,
    sell_win_rate: float,
    sharpe_ratio: float,
    total_return: float,
    max_drawdown: float,
    action_distribution: dict[str, float],
    gate_blocks: int,
    timing_blocks: int,
    budget_blocks: int,
    avg_win_pnl: float,
    avg_loss_pnl: float,
    win_loss_ratio: float,
    expectancy_pnl_per_sell_leg: float,
) -> Table:
    """Render a compact dashboard (<= 10 rows).

    The goal is to keep the live view focused on the few numbers that most
    directly explain: (1) what action was taken, (2) why, (3) how it impacted
    equity, and (4) whether risk/guardrails are binding.
    """

    table = Table(title="Lightweight BTC/USDT Trainer", expand=True)
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")

    # 1) Step
    table.add_row("Step", str(step))

    # 2) Price
    table.add_row("Close", f"{price:,.2f}")

    # 3) Action context (proposed vs executed + hold reason)
    proposed = getattr(result, "proposed_action", result.action)
    hold_reason = getattr(result, "hold_reason", None) or "-"
    action_str = result.action.upper()
    if proposed != result.action:
        action_str = f"{action_str} (proposed {proposed.upper()})"
    if hold_reason != "-":
        action_str = f"{action_str} | hold={hold_reason}"
    table.add_row("Action", action_str)

    # 4) Edge margin (gate signal strength)
    table.add_row("Edge", f"{result.edge_margin:+.3f} vs {config.EDGE_THRESHOLD:.3f}")

    # 5) Portfolio value and MTM delta
    table.add_row("Portfolio", f"{portfolio_value:,.2f} | MTM {mtm_delta:+.2f}")

    # 6) Exposure
    table.add_row("Exposure", f"cash {portfolio.cash:,.2f} | pos {portfolio.position:.6f}")

    # 7) Reward (raw trainer reward + scaled reward used by the agent)
    table.add_row("Reward", f"raw {result.trainer_reward:+.2f} | scaled {result.scaled_reward:+.3f}")

    # 8) Costs & execution impact
    slip = getattr(result, 'slippage_paid', 0.0)
    table.add_row("Costs", f"fee {result.fee_paid:+.2f} | turnover {result.turnover_penalty:+.2f} | slip {slip:+.2f} | impact {trade_impact:+.2f}")

    # 9) Performance snapshot
    perf = (
        f"ret {total_return:+.2%} | sharpe {sharpe_ratio:.2f} | dd {max_drawdown:.2%}"
        f" | trades {agent.state.trades} | win {sell_win_rate:.0%}"
    )
    table.add_row("Perf", perf)

    # 9b) Realized trade PnL diagnostics (per SELL leg, net of fees/slippage)
    table.add_row(
        "Trade PnL",
        f"avgW {avg_win_pnl:+.2f} | avgL {avg_loss_pnl:+.2f} | W/L {win_loss_ratio:.2f} | exp {expectancy_pnl_per_sell_leg:+.2f}",
    )

    # 10) Guardrail diagnostics (why actions were blocked)
    table.add_row("Blocks", f"gate {gate_blocks} | timing {timing_blocks} | budget {budget_blocks}")

    return table


def live_dashboard(
    events: Iterable[
        tuple[
            int,
            float,
            StepResult,
            float,
            float,
            float,
            Portfolio,
            BanditAgent,
            float,
            int,
            float,
            float,
            float,
            float,
            dict[str, float],
            int,
            int,
            int,
            float,
            float,
            float,
            float,
        ]
    ]
):
    with Live(refresh_per_second=int(1 / config.DASHBOARD_REFRESH), console=console, screen=False) as live:
        for (
            step,
            price,
            result,
            portfolio_value,
            mtm_delta,
            trade_impact,
            portfolio,
            agent,
            step_win_rate,
            refill_count,
            sell_win_rate,
            sharpe_ratio,
            total_return,
            max_drawdown,
            action_distribution,
            gate_blocks,
            timing_blocks,
            budget_blocks,
            avg_win_pnl,
            avg_loss_pnl,
            win_loss_ratio,
            expectancy_pnl_per_sell_leg,
        ) in events:
            live.update(
                build_table(
                    step,
                    price,
                    result,
                    portfolio_value,
                    mtm_delta,
                    trade_impact,
                    portfolio,
                    agent,
                    step_win_rate,
                    refill_count,
                    sell_win_rate,
                    sharpe_ratio,
                    total_return,
                    max_drawdown,
                    action_distribution,
                    gate_blocks,
                    timing_blocks,
                    budget_blocks,
                    avg_win_pnl,
                    avg_loss_pnl,
                    win_loss_ratio,
                    expectancy_pnl_per_sell_leg,
                )
            )
