from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from src import config
from src.agent import BanditAgent
from src.data_feed import DataFeed, MarketConfig
from src.indicators import compute_indicators
from src.dashboard import live_dashboard
from src.trainer import StepResult, Trainer, resume_from
from src.webapp import WebDashboard


def generate_run_id(symbol: str, timeframe: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_symbol = symbol.replace("/", "_")
    safe_timeframe = timeframe.replace("/", "_")
    return f"{safe_symbol}_{safe_timeframe}_{timestamp}"


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
    parser.add_argument("--run-id", default=None, help="identifier for this training run")
    parser.add_argument("--run-dir", default=None, help="directory to store run artifacts")
    parser.add_argument("--resume", action="store_true", help="resume from the latest or specified run directory")
    parser.add_argument("--checkpoint-every", type=int, default=config.DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--flush-trades-every", type=int, default=config.DEFAULT_FLUSH_TRADES_EVERY)
    parser.add_argument("--keep-last", type=int, default=config.DEFAULT_KEEP_LAST_CHECKPOINTS)
    parser.add_argument(
        "--warmup-hours",
        type=float,
        default=0.0,
        help="Optional preflight duration (hours) on the last 24h window before switching to live streaming.",
    )
    parser.add_argument(
        "--warmup-profit-target",
        type=float,
        default=0.0,
        help="Percentage gain on initial cash required before enabling --continuous mode (0 means breakeven).",
    )
    return parser.parse_args()


def run_loop(
    trainer: Trainer,
    frame: pd.DataFrame,
    steps: int,
    duration: float | None,
    delay: float,
    *,
    run_id: str,
    run_dir: Path,
    checkpoint_every: int,
    flush_trades_every: int,
    keep_last: int,
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

    episode_len = row_count - 1
    if duration is None and steps > episode_len:
        print(
            "Warning: requested steps exceed available window; repeating episodes over the same data."
        )

    first_price = float(frame.iloc[0]["close"])
    pv_prev_after = trainer.portfolio.value(first_price)

    while True:
        if duration is None and idx >= steps:
            break
        if duration is not None and time.monotonic() - start >= duration:
            break

        i = idx % episode_len
        row = frame.iloc[i]
        next_row = frame.iloc[i + 1]

        if i == 0 and idx > 0:
            trainer.reset_portfolio()
            pv_prev_after = trainer.portfolio.value(float(row["close"]))
        price = float(row["close"])

        before_trade_value = trainer.portfolio.value(price)
        result = trainer.step(row, next_row, idx)
        after_trade_value = trainer.portfolio.value(price)
        trade_impact = after_trade_value - before_trade_value
        mtm_delta = after_trade_value - pv_prev_after
        pv_prev_after = after_trade_value

        if flush_trades_every > 0 and trainer.steps % flush_trades_every == 0:
            trainer._flush_trades_and_metrics(run_dir)
        if checkpoint_every > 0 and trainer.steps % checkpoint_every == 0:
            trainer.agent.save(run_dir=run_dir, checkpoint=True, keep_last=keep_last)
            trainer._save_trainer_state(run_dir, run_id, checkpoint=True, keep_last=keep_last)
        yield idx, row, result, after_trade_value, mtm_delta, trade_impact

        idx += 1
        if delay > 0:
            time.sleep(delay)


def stream_live(
    trainer: Trainer,
    feed: DataFeed,
    delay: float,
    *,
    run_id: str,
    run_dir: Path,
    checkpoint_every: int,
    flush_trades_every: int,
    keep_last: int,
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

            if flush_trades_every > 0 and trainer.steps % flush_trades_every == 0:
                trainer._flush_trades_and_metrics(run_dir)
            if checkpoint_every > 0 and trainer.steps % checkpoint_every == 0:
                trainer.agent.save(run_dir=run_dir, checkpoint=True, keep_last=keep_last)
                trainer._save_trainer_state(run_dir, run_id, checkpoint=True, keep_last=keep_last)

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
    if args.warmup_hours > 0:
        candles_for_day = math.ceil((24 * 60) / timeframe_to_minutes(args.timeframe))
        limit = max(limit, candles_for_day)

    base_run_dir = Path(config.RUNS_DIR)
    base_run_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id.replace("/", "_") if args.run_id else None
    run_dir = Path(args.run_dir) if args.run_dir else None
    if args.resume:
        if run_dir is None:
            candidates = [d for d in base_run_dir.iterdir() if d.is_dir()]
            run_dir = max(candidates, key=lambda d: d.stat().st_mtime) if candidates else None
        if run_dir is not None:
            run_id = run_id or run_dir.name
        else:
            run_id = run_id or generate_run_id(args.symbol, args.timeframe)
            run_dir = base_run_dir / run_id
    else:
        run_id = run_id or generate_run_id(args.symbol, args.timeframe)
        run_dir = run_dir or base_run_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    feed = DataFeed(
        MarketConfig(symbol=args.symbol, timeframe=args.timeframe, limit=limit, offline=args.offline)
    )

    agent = BanditAgent()
    agent.state.run_id = agent.state.run_id or run_id
    agent.state.symbol = agent.state.symbol or args.symbol
    agent.state.timeframe = agent.state.timeframe or args.timeframe
    trainer = Trainer(agent)
    if args.resume:
        resume_from(run_dir, agent, trainer)
    web_dashboard = WebDashboard(port=args.web_port) if args.web_dashboard else None
    if web_dashboard:
        web_dashboard.start()

    if args.continuous:
        raw_frame, _ = feed.fetch()
        feature_frame = compute_indicators(raw_frame)

        warmup_target = trainer.initial_cash * (1 + args.warmup_profit_target / 100)
        warmup_seconds = args.warmup_hours * 3600
        def warmup_then_stream():
            warmup_hit = False

            if args.warmup_hours > 0:
                print(
                    f"Starting warmup for up to {args.warmup_hours:.2f}h on the last 24h window "
                    f"(target portfolio >= {warmup_target:.2f})."
                )
                for event in run_loop(
                    trainer,
                    feature_frame,
                    args.steps,
                    warmup_seconds,
                    args.delay,
                    run_id=run_id,
                    run_dir=run_dir,
                    checkpoint_every=args.checkpoint_every,
                    flush_trades_every=args.flush_trades_every,
                    keep_last=args.keep_last,
                ):
                    yield event
                    _, _, _, portfolio_value, _, _ = event
                    if portfolio_value >= warmup_target:
                        warmup_hit = True
                        print(
                            f"Warmup profit target reached (portfolio {portfolio_value:.2f} >= {warmup_target:.2f}). "
                            "Switching to live stream..."
                        )
                        break

                if not warmup_hit:
                    print(
                        f"Warmup ended without reaching the profit target ({warmup_target:.2f}). "
                        "Continuous mode will not start."
                    )
                    return

            print("Starting continuous live stream...")
            yield from stream_live(
                trainer,
                feed,
                args.delay,
                run_id=run_id,
                run_dir=run_dir,
                checkpoint_every=args.checkpoint_every,
                flush_trades_every=args.flush_trades_every,
                keep_last=args.keep_last,
            )

        loop = warmup_then_stream()
    else:
        raw_frame, _ = feed.fetch()
        feature_frame = compute_indicators(raw_frame)
        loop = run_loop(
            trainer,
            feature_frame,
            args.steps,
            args.duration,
            args.delay,
            run_id=run_id,
            run_dir=run_dir,
            checkpoint_every=args.checkpoint_every,
            flush_trades_every=args.flush_trades_every,
            keep_last=args.keep_last,
        )

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
                            realized_pnl=result.realized_pnl,
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
                            realized_pnl=result.realized_pnl,
                        )
                    yield

            for _ in emit(loop):
                pass
    except KeyboardInterrupt:
        print("Interrupted; saving agent state before exit...")
    finally:
        trainer._flush_trades_and_metrics(run_dir, force=True)
        trainer.agent.save(run_dir=run_dir, keep_last=args.keep_last)
        trainer._save_trainer_state(run_dir, run_id, checkpoint=False, keep_last=args.keep_last)

        if web_dashboard:
            web_dashboard.stop()


if __name__ == "__main__":
    main()
