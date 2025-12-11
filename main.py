from __future__ import annotations

import argparse
import math
import time
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
    parser.add_argument("--steps", type=int, default=50, help="max steps to run (ignored when --duration is set)")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="seconds to keep training; also interpreted as a minute lookback window when offline",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=config.DASHBOARD_REFRESH,
        help="pause (seconds) between trading events to make learning visible",
    )
    parser.add_argument("--offline", action="store_true", help="use synthetic data instead of live exchange")
    parser.add_argument("--dashboard", action="store_true", help="enable live dashboard rendering")
    parser.add_argument("--continuous", action="store_true", help="keep fetching live data until interrupted")
    return parser.parse_args()


def run_loop(
    trainer: Trainer, frame: pd.DataFrame, steps: int, duration: float | None, delay: float
) -> Iterable[tuple[int, float, str, float]]:
    """
    Generate trading events either for a fixed number of steps or until a duration elapses.

    When a duration is provided the data frame is cycled to keep producing events and a
    small delay is applied between steps so the dashboard can render progress over time.
    """

    start = time.monotonic()
    idx = 0
    row_count = len(frame)
    if row_count == 0:
        return

    while True:
        if duration is None and idx >= steps:
            break
        if duration is not None and time.monotonic() - start >= duration:
            break

        row = frame.iloc[idx % row_count]
        price = float(row["close"])

        before_trade_value = trainer.portfolio.value(price)
        trainer.step(row, idx)
        after_trade_value = trainer.portfolio.value(price)
        delta = after_trade_value - before_trade_value
        yield idx, price, trainer.history[-1][1], delta

        idx += 1
        if delay > 0:
            time.sleep(delay)


def stream_live(
    trainer: Trainer, feed: DataFeed, delay: float
) -> Iterable[tuple[int, float, str, float]]:
    """Continuously fetch new market data and yield trading events indefinitely."""

    last_ts = None
    idx = 0

    while True:
        frame = compute_indicators(feed.fetch())
        if frame.empty:
            if delay > 0:
                time.sleep(delay)
            continue

        for _, row in frame.iterrows():
            ts = row.get("timestamp")
            if last_ts is not None and ts is not None and ts <= last_ts:
                continue

            price = float(row["close"])
            before_trade_value = trainer.portfolio.value(price)
            trainer.step(row, idx)
            after_trade_value = trainer.portfolio.value(price)
            delta = after_trade_value - before_trade_value

            yield idx, price, trainer.history[-1][1], delta

            idx += 1
            last_ts = ts if ts is not None else last_ts

            if delay > 0:
                time.sleep(delay)


def timeframe_to_minutes(timeframe: str) -> float:
    """Convert timeframe strings like "1m", "5m", or "1h" to minutes."""

    if not timeframe:
        return 1.0

    unit = timeframe[-1].lower()
    try:
        quantity = float(timeframe[:-1])
    except ValueError:
        return 1.0

    multiplier = {"m": 1, "h": 60, "d": 24 * 60}.get(unit)
    return quantity * multiplier if multiplier else 1.0


def main() -> None:
    args = parse_args()
    limit = args.limit
    if args.offline and args.duration:
        candles_for_window = math.ceil(args.duration / timeframe_to_minutes(args.timeframe))
        limit = max(limit, candles_for_window)

    feed = DataFeed(
        MarketConfig(symbol=args.symbol, timeframe=args.timeframe, limit=limit, offline=args.offline)
    )

    agent = BanditAgent()
    trainer = Trainer(agent)

    if args.continuous:
        loop = stream_live(trainer, feed, args.delay)
    else:
        raw_frame = feed.fetch()
        feature_frame = compute_indicators(raw_frame)
        loop = run_loop(trainer, feature_frame, args.steps, args.duration, args.delay)

    try:
        if args.dashboard:
            from src.dashboard import live_dashboard as render

            def enrich(events):
                for step, price, action, reward in events:
                    yield step, price, action, reward, trainer.portfolio, agent, trainer.success_rate

            render(enrich(loop))
        else:
            for _ in loop:
                pass
    except KeyboardInterrupt:
        trainer.agent.save()
        raise
    else:
        trainer.agent.save()


if __name__ == "__main__":
    main()
