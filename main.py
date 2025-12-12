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
from src.trainer import StepResult, Trainer
from src.webapp import WebDashboard


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
    parser.add_argument("--web-dashboard", action="store_true", help="serve a Plotly HTML dashboard on port 8000")
    parser.add_argument("--web-port", type=int, default=8000, help="port for the web dashboard")
    parser.add_argument("--continuous", action="store_true", help="keep fetching live data until interrupted")
    return parser.parse_args()


def run_loop(
    trainer: Trainer, frame: pd.DataFrame, steps: int, duration: float | None, delay: float
) -> Iterable[tuple[int, pd.Series, StepResult, float, float, float]]:
    """
    Generate trading events either for a fixed number of steps or until a duration elapses.

    When a duration is provided the data frame is cycled to keep producing events and a
    small delay is applied between steps so the dashboard can render progress over time.
    """

    start = time.monotonic()
    idx = 0
    row_count = len(frame)
    if row_count < 2:
        return []

    first_price = float(frame.iloc[0]["close"])
    pv_prev_after = trainer.portfolio.value(first_price)

    while True:
        if duration is None and idx >= steps:
            break
        if duration is not None and time.monotonic() - start >= duration:
            break

        row = frame.iloc[idx % row_count]
        next_row = frame.iloc[(idx + 1) % row_count]
        price = float(row["close"])

        before_trade_value = trainer.portfolio.value(price)
        result = trainer.step(row, next_row, idx)
        after_trade_value = trainer.portfolio.value(price)
        trade_impact = after_trade_value - before_trade_value
        mtm_delta = after_trade_value - pv_prev_after
        pv_prev_after = after_trade_value
        yield idx, row, result, after_trade_value, mtm_delta, trade_impact

        idx += 1
        if delay > 0:
            time.sleep(delay)


def stream_live(
    trainer: Trainer, feed: DataFeed, delay: float
) -> Iterable[tuple[int, pd.Series, StepResult, float, float, float]]:
    """Continuously fetch new market data and yield trading events indefinitely."""

    last_ts = None
    idx = 0
    pv_prev_after = trainer.portfolio.value(0.0)

    while True:
        raw_frame, _ = feed.fetch()
        frame = compute_indicators(raw_frame)
        if frame.empty:
            if delay > 0:
                time.sleep(delay)
            continue

        for i in range(max(0, len(frame) - 1)):
            row = frame.iloc[i]
            next_row = frame.iloc[i + 1]
            ts = row.get("timestamp")
            if last_ts is not None and ts is not None and ts <= last_ts:
                continue

            price = float(row["close"])
            before_trade_value = trainer.portfolio.value(price)
            result = trainer.step(row, next_row, idx)
            after_trade_value = trainer.portfolio.value(price)
            trade_impact = after_trade_value - before_trade_value
            mtm_delta = after_trade_value - pv_prev_after
            pv_prev_after = after_trade_value

            yield idx, row, result, after_trade_value, mtm_delta, trade_impact

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
    web_dashboard = WebDashboard(port=args.web_port) if args.web_dashboard else None
    if web_dashboard:
        web_dashboard.start()

    if args.continuous:
        loop = stream_live(trainer, feed, args.delay)
    else:
        raw_frame, _ = feed.fetch()
        feature_frame = compute_indicators(raw_frame)
        loop = run_loop(trainer, feature_frame, args.steps, args.duration, args.delay)

    try:
        if args.dashboard:
            from src.dashboard import live_dashboard as render

            def enrich(events):
                for step, row, result, portfolio_value, mtm_delta, trade_impact in events:
                    price = float(row["close"])
                    if web_dashboard:
                        web_dashboard.publish_event(
                            step=step,
                            timestamp=row.get("timestamp"),
                            ohlc={
                                "open": float(row.get("open", price)),
                                "high": float(row.get("high", price)),
                                "low": float(row.get("low", price)),
                                "close": price,
                            },
                            action=result.action,
                            reward=result.trainer_reward,
                            portfolio_value=portfolio_value,
                            cash=trainer.portfolio.cash,
                            position=trainer.portfolio.position,
                            success_rate=trainer.success_rate,
                            step_win_rate=trainer.success_rate,
                            total_reward=trainer.agent.state.total_reward,
                            trainer_reward=result.trainer_reward,
                            mtm_delta=mtm_delta,
                            trade_impact=trade_impact,
                            fee_paid=result.fee_paid,
                            turnover_penalty=result.turnover_penalty,
                            refilled=result.refilled,
                            refill_count=trainer.refill_count,
                            executed_trades=trainer.agent.state.trades,
                            sell_win_rate=trainer.trade_win_rate,
                        )
                    yield (
                        step,
                        price,
                        result,
                        portfolio_value,
                        mtm_delta,
                        trade_impact,
                        trainer.portfolio,
                        agent,
                        trainer.success_rate,
                        trainer.refill_count,
                        trainer.trade_win_rate,
                    )

            render(enrich(loop))
        else:

            def emit(events):
                for step, row, result, portfolio_value, mtm_delta, trade_impact in events:
                    price = float(row["close"])
                    if web_dashboard:
                        web_dashboard.publish_event(
                            step=step,
                            timestamp=row.get("timestamp"),
                            ohlc={
                                "open": float(row.get("open", price)),
                                "high": float(row.get("high", price)),
                                "low": float(row.get("low", price)),
                                "close": price,
                            },
                            action=result.action,
                            reward=result.trainer_reward,
                            portfolio_value=portfolio_value,
                            cash=trainer.portfolio.cash,
                            position=trainer.portfolio.position,
                            success_rate=trainer.success_rate,
                            step_win_rate=trainer.success_rate,
                            total_reward=trainer.agent.state.total_reward,
                            trainer_reward=result.trainer_reward,
                            mtm_delta=mtm_delta,
                            trade_impact=trade_impact,
                            fee_paid=result.fee_paid,
                            turnover_penalty=result.turnover_penalty,
                            refilled=result.refilled,
                            refill_count=trainer.refill_count,
                            executed_trades=trainer.agent.state.trades,
                            sell_win_rate=trainer.trade_win_rate,
                        )
                    yield

            for _ in emit(loop):
                pass
    except KeyboardInterrupt:
        print("Interrupted; saving agent state before exit...")
    finally:
        trainer.agent.save()

        if web_dashboard:
            web_dashboard.stop()


if __name__ == "__main__":
    main()
