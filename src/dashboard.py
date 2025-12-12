from __future__ import annotations

from typing import Iterable

from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.table import Table

from src.agent import ACTIONS, BanditAgent
from src.trainer import Portfolio, StepResult
from src import config

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
) -> Table:
    table = Table(title="Lightweight BTC/USDT Trainer", expand=True)
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")

    table.add_row("Step", str(step))
    table.add_row("Close Price", f"{price:,.2f}")
    table.add_row("Action", result.action.upper())
    table.add_row("Trainer Reward", f"{result.trainer_reward:+.4f}")
    table.add_row("Portfolio Δ (MTM)", f"{mtm_delta:+.4f}")
    table.add_row("Trade Impact", f"{trade_impact:+.4f}")
    table.add_row("Fee", f"{result.fee_paid:+.4f}")
    table.add_row("Turnover Penalty", f"{result.turnover_penalty:+.4f}")
    table.add_row("Refilled", "Yes" if result.refilled else "No")
    table.add_row("Refill Count", str(refill_count))
    table.add_row("Cash", f"{portfolio.cash:,.2f}")
    table.add_row("Position", f"{portfolio.position:.6f}")
    table.add_row("Portfolio Value", f"{portfolio_value:,.2f}")
    table.add_row("Executed Trades", str(agent.state.trades))
    table.add_row("Sell Win Rate", f"{sell_win_rate:.2%}")
    table.add_row("Step Win Rate", f"{step_win_rate:,.2f}%")
    table.add_row("Total Reward", f"{agent.state.total_reward:+.2f}")

    q_values = Table.grid(expand=True)
    q_values.add_row("Q-values", ", ".join(f"{a}:{v:.3f}" for a, v in zip(ACTIONS, agent.state.q_values)))
    epsilon = agent.state.last_epsilon if hasattr(agent, "state") else config.EPSILON
    table.add_row("Exploration (eps)", f"{epsilon:.2f}")
    table.add_row("", "")
    table.add_row("Q", Align.left(q_values))
    return table


def live_dashboard(
    events: Iterable[
        tuple[int, float, StepResult, float, float, float, Portfolio, BanditAgent, float, int, float]
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
                )
            )
