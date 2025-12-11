from __future__ import annotations

import argparse
from typing import Iterable

import pandas as pd

from src import config
from src.agent import BanditAgent
from src.data_feed import DataFeed, MarketConfig
from src.indicators import compute_indicators
from src.dashboard import live_dashboard
from src.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight BTC/USDT learner")
    parser.add_argument("--symbol", default=config.DEFAULT_SYMBOL)
    parser.add_argument("--timeframe", default=config.DEFAULT_TIMEFRAME)
    parser.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT)
    parser.add_argument("--steps", type=int, default=50, help="max steps to run")
    parser.add_argument("--offline", action="store_true", help="use synthetic data instead of live exchange")
    parser.add_argument("--dashboard", action="store_true", help="enable live dashboard rendering")
    return parser.parse_args()


def run_loop(trainer: Trainer, frame: pd.DataFrame, steps: int) -> Iterable[tuple[int, float, str, float]]:
    for idx, row in frame.head(steps).iterrows():
        before_trade_value = trainer.portfolio.value(float(row["close"]))
        trainer.step(row, idx)
        after_trade_value = trainer.portfolio.value(float(row["close"]))
        delta = after_trade_value - before_trade_value
        yield idx, float(row["close"]), trainer.history[-1][1], delta


def main() -> None:
    args = parse_args()
    feed = DataFeed(MarketConfig(symbol=args.symbol, timeframe=args.timeframe, limit=args.limit, offline=args.offline))
    raw_frame = feed.fetch()
    feature_frame = compute_indicators(raw_frame)

    agent = BanditAgent()
    trainer = Trainer(agent)

    loop = run_loop(trainer, feature_frame, args.steps)

    if args.dashboard:
        from src.dashboard import live_dashboard as render

        def enrich(events):
            for step, price, action, reward in events:
                yield step, price, action, reward, trainer.portfolio, agent

        render(enrich(loop))
    else:
        for _ in loop:
            pass

    trainer.agent.save()


if __name__ == "__main__":
    main()
