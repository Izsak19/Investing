from __future__ import annotations

from typing import Iterable

from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.table import Table

from src.agent import ACTIONS, BanditAgent
from src.trainer import Portfolio
from src import config

console = Console()


def build_table(step: int, price: float, action: str, reward: float, portfolio: Portfolio, agent: BanditAgent) -> Table:
    table = Table(title="Lightweight BTC/USDT Trainer", expand=True)
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")

    table.add_row("Step", str(step))
    table.add_row("Close Price", f"{price:,.2f}")
    table.add_row("Action", action.upper())
    table.add_row("Reward", f"{reward:+.4f}")
    table.add_row("Cash", f"{portfolio.cash:,.2f}")
    table.add_row("Position", f"{portfolio.position:.6f}")
    table.add_row("Portfolio Value", f"{portfolio.value(price):,.2f}")
    table.add_row("Total Reward", f"{agent.state.total_reward:+.2f}")

    q_values = Table.grid(expand=True)
    q_values.add_row("Q-values", ", ".join(f"{a}:{v:.3f}" for a, v in zip(ACTIONS, agent.state.q_values)))
    table.add_row("Exploration (eps)", f"{config.EPSILON:.2f}")
    table.add_row("", "")
    table.add_row("Q", Align.left(q_values))
    return table


def live_dashboard(events: Iterable[tuple[int, float, str, float, Portfolio, BanditAgent]]):
    with Live(refresh_per_second=int(1 / config.DASHBOARD_REFRESH), console=console, screen=False) as live:
        for step, price, action, reward, portfolio, agent in events:
            live.update(build_table(step, price, action, reward, portfolio, agent))
