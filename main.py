from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from src import config
from src.agent import AgentState, RLSForgettingAgent
from src.data_feed import DataFeed, MarketConfig
from src.dashboard import live_dashboard
from src.timeframe import timeframe_to_minutes
from src.trainer import StepResult, Trainer, resume_from
from src.webapp import WebDashboard


def generate_run_id(symbol: str, timeframe: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
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
    parser.add_argument("--cache", action="store_true", help="cache fetched datasets for reproducibility")
    parser.add_argument("--cache-only", action="store_true", help="load data exclusively from the cache")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"), help="dataset cache directory")
    parser.add_argument(
        "--window-end",
        default=None,
        help="Cache window key/ISO timestamp to force a distinct cached dataset (default: latest).",
    )
    parser.add_argument(
        "--derivatives",
        action="store_true",
        help="Include derivatives signals (funding rate, open interest, order book imbalance) when available.",
    )
    parser.add_argument(
        "--derivatives-exchange",
        default="binanceusdm",
        help="ccxt exchange id for derivatives data (default: binanceusdm).",
    )
    parser.add_argument(
        "--history-candles",
        type=int,
        default=0,
        help="Number of candles to fetch for backtests (0 uses --limit).",
    )
    parser.add_argument("--dashboard", action="store_true", help="enable live dashboard rendering")
    parser.add_argument("--web-dashboard", action="store_true", help="serve a Plotly HTML dashboard on port 8000")
    parser.add_argument("--web-port", type=int, default=8000, help="port for the web dashboard")
    parser.add_argument(
        "--web-window-hours",
        type=float,
        default=48.0,
        help="Time window to display on the web dashboard (hours).",
    )
    parser.add_argument("--continuous", action="store_true", help="keep fetching live data until interrupted")
    parser.add_argument("--run-id", default=None, help="identifier for this training run")
    parser.add_argument("--run-dir", default=None, help="directory to store run artifacts")
    parser.add_argument("--resume", action="store_true", help="resume from the latest or specified run directory")
    parser.add_argument("--checkpoint-every", type=int, default=config.DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--flush-trades-every", type=int, default=config.DEFAULT_FLUSH_TRADES_EVERY)
    parser.add_argument("--keep-last", type=int, default=config.DEFAULT_KEEP_LAST_CHECKPOINTS)
    parser.add_argument(
        "--fresh-agent",
        action="store_true",
        help="Start with a fresh agent state instead of loading data/state.json.",
    )
    parser.add_argument(
        "--cycle-window",
        action="store_true",
        help="Cycle the candle window when steps exceed available history (default: clamp to history length).",
    )
    parser.add_argument(
        "--warmup-hours",
        type=float,
        default=0.0,
        help="Optional preflight duration (hours) on a lookback window before switching to live streaming. The lookback is max(24h, warmup-hours).",
    )
    parser.add_argument(
        "--warmup-profit-target",
        type=float,
        default=0.0,
        help="Percentage gain on initial cash required before enabling --continuous mode (0 means breakeven).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Single-pass warmup steps before going live (0 uses --steps).",
    )
    parser.add_argument(
        "--posterior-scale",
        type=float,
        default=None,
        help="Override POSTERIOR_SCALE used for Thompson sampling (0 disables sampling).",
    )
    parser.add_argument(
        "--forgetting-factor",
        type=float,
        default=None,
        help="Forgetting factor (lambda) for RLS updates; 1.0 falls back to no forgetting.",
    )
    parser.add_argument("--eval", action="store_true", help="Run one evaluation pass without training or saving state")
    parser.add_argument(
        "--profile",
        choices=sorted(config.PROFILES.keys()),
        default=None,
        help="Preset of exploration/penalty knobs (see config.PROFILES)",
    )
    parser.add_argument(
        "--penalty-profile",
        choices=sorted(config.PENALTY_PROFILES.keys()),
        default="train",
        help="Penalty weighting profile for reward shaping",
    )
    parser.add_argument(
        "--warmup-trades",
        type=int,
        default=config.WARMUP_TRADES_BEFORE_GATING,
        help="Number of executed trades before enabling gating and timing locks",
    )
    parser.add_argument(
        "--walkforward",
        action="store_true",
        help="Run walk-forward evaluation on the fetched dataset (non-continuous only).",
    )
    parser.add_argument(
        "--walkforward-folds",
        type=int,
        default=config.WALKFORWARD_FOLDS,
        help="Number of walk-forward folds (expanding train, rolling eval).",
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
    data_is_live: bool = False,
    train: bool = True,
    posterior_scale_override: float | None = None,
    cycle_window: bool = False,
) -> Iterable[tuple[int, pd.Series, StepResult, float, float, float]]:
    """
    Generate trading events either for a fixed number of steps or until a duration elapses.

    When a duration is provided the data frame is cycled to keep producing events and a
    small delay is applied between steps so the dashboard can render progress over time.
    """

    start = time.monotonic()
    trainer.last_data_is_live = data_is_live
    idx = 0
    row_count = len(frame)
    if row_count < 2:
        return []

    episode_len = row_count - 1
    # When the requested step count exceeds the available candle window, cycle the
    # window rather than silently truncating. This is useful for "learning cycles"
    # where you want a fixed number of trades/updates.
    cycle_window = bool(duration is not None or cycle_window)
    if duration is None and steps > episode_len:
        if cycle_window:
            print("Info: requested steps exceed available window; cycling window to reach requested steps.")
        else:
            print("Info: requested steps exceed available window; clamping steps to available history.")
            steps = episode_len

    first_price = float(frame.iloc[0]["close"])
    pv_prev_after = trainer.portfolio.value(first_price)

    while True:
        if duration is None and idx >= steps:
            break
        if duration is not None and time.monotonic() - start >= duration:
            break

        i = idx % episode_len if cycle_window else idx
        row = frame.iloc[i]
        next_row = frame.iloc[i + 1]

        if i == 0 and idx > 0:
            # When the window loops, keep the existing portfolio so performance can
            # compound instead of resetting to the initial cash every cycle.
            pv_prev_after = trainer.portfolio.value(float(row["close"]))
        price = float(row["close"])

        before_trade_value = trainer.portfolio.value(price)
        edge_horizon = max(1, int(getattr(config, "EDGE_GATE_TARGET_HORIZON", 1) or 1))
        future_idx = i + edge_horizon
        if cycle_window:
            future_idx = future_idx % episode_len
        else:
            future_idx = min(future_idx, episode_len)
        future_price = float(frame.iloc[future_idx]["close"])

        result = trainer.step(
            row,
            next_row,
            idx,
            train=train,
            posterior_scale_override=posterior_scale_override,
            future_price=future_price,
        )
        after_trade_value = trainer.portfolio.value(price)
        trade_impact = after_trade_value - before_trade_value
        mtm_delta = after_trade_value - pv_prev_after
        pv_prev_after = after_trade_value

        if train:
            if flush_trades_every > 0 and trainer.steps % flush_trades_every == 0:
                trainer._flush_trades_and_metrics(run_dir, data_is_live=data_is_live)
            if checkpoint_every > 0 and trainer.steps % checkpoint_every == 0:
                trainer.agent.save(run_dir=run_dir, checkpoint=True, keep_last=keep_last)
                trainer._save_trainer_state(run_dir, run_id, checkpoint=True, keep_last=keep_last)
        # Kill-switch: stop early if the strategy is structurally losing (expectancy) and drawdown is bad.
        if (
            getattr(config, 'ENABLE_KILL_SWITCH', False)
            and train
            and getattr(trainer, 'sell_legs', 0) >= getattr(config, 'KILL_SWITCH_MIN_SELL_LEGS', 0)
            and getattr(trainer, 'expectancy_pnl_per_sell_leg', 0.0) <= getattr(config, 'KILL_SWITCH_EXPECTANCY_PNL_PER_SELL_LEG', -1e9)
            and getattr(trainer, 'max_drawdown', 0.0) >= getattr(config, 'KILL_SWITCH_MAX_DRAWDOWN', 1.0)
        ):
            print(
                f"Kill-switch triggered at step {idx}: "
                f"expectancy={getattr(trainer,'expectancy_pnl_per_sell_leg',0.0):+.4f}, "
                f"dd={getattr(trainer,'max_drawdown',0.0):.2%}, sells={getattr(trainer,'sell_legs',0)}"
            )
            trainer._flush_trades_and_metrics(run_dir, force=True, data_is_live=data_is_live)
            break

        # Hard guardrail on absolute trade count (prevents runaway churn loops).
        if (
            getattr(config, 'ENABLE_KILL_SWITCH', False)
            and train
            and getattr(trainer, 'executed_trade_count', 0) >= getattr(config, 'KILL_SWITCH_MAX_TRADES', 10**9)
        ):
            print(
                f"Kill-switch trade cap reached at step {idx}: executed_trades={getattr(trainer,'executed_trade_count',0)}"
            )
            trainer._flush_trades_and_metrics(run_dir, force=True, data_is_live=data_is_live)
            break

        yield idx, row, result, after_trade_value, mtm_delta, trade_impact

        idx += 1
        if delay > 0:
            time.sleep(delay)


def run_walkforward(
    frame: pd.DataFrame,
    folds: int,
    *,
    build_trainer,
    run_dir: Path,
    data_is_live: bool,
    posterior_scale_override: float | None,
    delay: float,
    event_callback=None,
) -> list[dict[str, float]]:
    if folds < 2:
        raise ValueError("walkforward requires at least 2 folds")
    fold_size = len(frame) // folds
    if fold_size < 2:
        raise ValueError("walkforward requires more data per fold")

    results: list[dict[str, float]] = []
    for k in range(1, folds):
        start = k * fold_size
        end = (k + 1) * fold_size if k < folds - 1 else len(frame)
        train_frame = frame.iloc[:start]
        eval_frame = frame.iloc[start:end]

        agent, trainer = build_trainer()
        if len(train_frame) > 1:
            for event in run_loop(
                trainer,
                train_frame,
                len(train_frame) - 1,
                None,
                0.0,
                run_id=f"wf_train_{k}",
                run_dir=run_dir,
                checkpoint_every=0,
                flush_trades_every=0,
                keep_last=0,
                data_is_live=data_is_live,
                train=True,
                posterior_scale_override=posterior_scale_override,
                cycle_window=False,
            ):
                if event_callback:
                    event_callback(event, trainer, agent)

        trainer.reset_portfolio()
        if len(eval_frame) > 1:
            for event in run_loop(
                trainer,
                eval_frame,
                len(eval_frame) - 1,
                None,
                delay,
                run_id=f"wf_eval_{k}",
                run_dir=run_dir,
                checkpoint_every=0,
                flush_trades_every=0,
                keep_last=0,
                data_is_live=data_is_live,
                train=False,
                posterior_scale_override=posterior_scale_override,
                cycle_window=False,
            ):
                if event_callback:
                    event_callback(event, trainer, agent)

        results.append(
            {
                "fold": float(k),
                "steps": float(trainer.total_steps),
                "trade_win_rate": float(trainer.trade_win_rate),
                "avg_win_pnl": float(trainer.avg_win_pnl),
                "avg_loss_pnl": float(trainer.avg_loss_pnl),
                "expectancy_pnl_per_sell_leg": float(trainer.expectancy_pnl_per_sell_leg),
                "total_return": float(trainer.total_return),
                "max_drawdown": float(trainer.max_drawdown),
                "forced_exit_count": float(trainer.forced_exit_count),
            }
        )
    return results


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
    posterior_scale_override: float | None = None,
) -> Iterable[tuple[int, pd.Series, StepResult, float, float, float]]:
    """Continuously fetch new market data and yield trading events indefinitely."""

    last_ts = None
    idx = 0
    pv_prev_after = trainer.portfolio.value(0.0)
    trainer.last_data_is_live = None

    while True:
        raw_frame, is_live = feed.fetch(include_indicators=True)
        trainer.last_data_is_live = is_live
        frame = raw_frame
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
            edge_horizon = max(1, int(getattr(config, "EDGE_GATE_TARGET_HORIZON", 1) or 1))
            future_idx = min(i + edge_horizon, len(frame) - 1)
            future_price = float(frame.iloc[future_idx]["close"])
            result = trainer.step(
                row,
                next_row,
                idx,
                posterior_scale_override=posterior_scale_override,
                future_price=future_price,
            )
            after_trade_value = trainer.portfolio.value(price)
            trade_impact = after_trade_value - before_trade_value
            mtm_delta = after_trade_value - pv_prev_after
            pv_prev_after = after_trade_value

            if flush_trades_every > 0 and trainer.steps % flush_trades_every == 0:
                trainer._flush_trades_and_metrics(run_dir, data_is_live=is_live)
            if checkpoint_every > 0 and trainer.steps % checkpoint_every == 0:
                trainer.agent.save(run_dir=run_dir, checkpoint=True, keep_last=keep_last)
                trainer._save_trainer_state(run_dir, run_id, checkpoint=True, keep_last=keep_last)

            # Kill-switch: stop early if expectancy is clearly negative and drawdown is bad.
            if (
                getattr(config, "ENABLE_KILL_SWITCH", False)
                and getattr(trainer, "sell_legs", 0) >= getattr(config, "KILL_SWITCH_MIN_SELL_LEGS", 0)
                and getattr(trainer, "expectancy_pnl_per_sell_leg", 0.0)
                <= getattr(config, "KILL_SWITCH_EXPECTANCY_PNL_PER_SELL_LEG", -1e9)
                and getattr(trainer, "max_drawdown", 0.0) >= getattr(config, "KILL_SWITCH_MAX_DRAWDOWN", 1.0)
            ):
                print(
                    f"Kill-switch triggered at step {idx}: "
                    f"expectancy={getattr(trainer,'expectancy_pnl_per_sell_leg',0.0):+.4f}, "
                    f"dd={getattr(trainer,'max_drawdown',0.0):.2%}, sells={getattr(trainer,'sell_legs',0)}"
                )
                trainer._flush_trades_and_metrics(run_dir, force=True, data_is_live=is_live)
                return

            # Hard guardrail on absolute trade count (prevents runaway churn loops).
            if (
                getattr(config, "ENABLE_KILL_SWITCH", False)
                and getattr(trainer, "executed_trade_count", 0) >= getattr(config, "KILL_SWITCH_MAX_TRADES", 10**9)
            ):
                print(
                    f"Kill-switch trade cap reached at step {idx}: "
                    f"executed_trades={getattr(trainer,'executed_trade_count',0)}"
                )
                trainer._flush_trades_and_metrics(run_dir, force=True, data_is_live=is_live)
                return

            yield idx, row, result, after_trade_value, mtm_delta, trade_impact

            idx += 1
            last_ts = ts if ts is not None else last_ts

            if delay > 0:
                time.sleep(delay)


def _initialize_offline_seeds(seed: int, run_dir: Path) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    seeds_path = run_dir / "seeds.json"
    seeds_path.write_text(json.dumps({"seed": seed, "mode": "offline"}, indent=2))


def main() -> None:
    args = parse_args()
    # Apply requested profile first.
    config.apply_profile(args.profile)
    # Professional default: for 5m and above, be conservative unless user explicitly chose a profile.
    # This prevents fee-death-by-churn when users forget --profile.
    try:
        tf_min = timeframe_to_minutes(args.timeframe)
    except Exception:
        tf_min = None
    if args.profile is None and tf_min is not None and tf_min >= 5:
        # 5m+: default to a profitability-focused preset to avoid churn and to
        # enforce hard risk exits by default.
        config.apply_profile('tf_5m_profit_focus')
    eval_mode = args.eval
    if eval_mode and args.continuous:
        print("--eval disables --continuous; running a bounded evaluation pass instead.")
        args.continuous = False
    if args.warmup_trades is not None:
        config.WARMUP_TRADES_BEFORE_GATING = max(0, args.warmup_trades)
    warmup_steps = max(0, int(getattr(args, "warmup_steps", 0) or 0))
    if warmup_steps <= 0 and args.continuous and (args.warmup_hours > 0 or args.warmup_profit_target != 0):
        warmup_steps = max(0, int(args.steps))
    limit = args.limit
    if args.offline and args.duration:
        candles_for_window = math.ceil(args.duration / timeframe_to_minutes(args.timeframe))
        limit = max(limit, candles_for_window)
    if args.warmup_hours > 0:
        # Fetch enough history for the warmup. Historically this was fixed at 24h;
        # we now scale the lookback with warmup duration (minimum 24h for stability).
        warmup_lookback_hours = max(24.0, float(args.warmup_hours))
        candles_for_warmup = math.ceil((warmup_lookback_hours * 60) / timeframe_to_minutes(args.timeframe))
        limit = max(limit, candles_for_warmup)
    history_candles = max(0, int(getattr(args, "history_candles", 0) or 0))
    if history_candles > 0:
        limit = max(limit, history_candles)
    if args.continuous and warmup_steps > 0:
        limit = max(limit, warmup_steps + 1)
    if args.duration is None and args.steps > limit:
        limit = args.steps + 1

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

    if args.offline:
        _initialize_offline_seeds(config.DEFAULT_RANDOM_SEED, run_dir)

    feed = DataFeed(
        MarketConfig(
            symbol=args.symbol,
            timeframe=args.timeframe,
            limit=limit,
            offline=args.offline,
            cache=args.cache,
            cache_only=args.cache_only,
            cache_dir=args.cache_dir,
            window_end=args.window_end,
            derivatives=args.derivatives,
            derivatives_exchange=args.derivatives_exchange,
            include_orderbook=True,
        )
    )

    penalty_profile = "eval" if eval_mode else args.penalty_profile

    def build_trainer():
        agent = RLSForgettingAgent(
            posterior_scale=args.posterior_scale, forgetting_factor=args.forgetting_factor
        )
        if args.fresh_agent:
            agent.state = AgentState.default()
            agent._prepare_state()
        agent.state.run_id = agent.state.run_id or run_id
        agent.state.symbol = agent.state.symbol or args.symbol
        agent.state.timeframe = agent.state.timeframe or args.timeframe
        trainer = Trainer(agent, timeframe=args.timeframe, penalty_profile=penalty_profile)
        return agent, trainer

    agent, trainer = build_trainer()
    eval_scale_floor = max(config.POSTERIOR_SCALE_MIN, 1e-3)
    posterior_override = (
        args.posterior_scale if args.posterior_scale is not None else (eval_scale_floor if eval_mode else None)
    )
    if posterior_override is not None:
        posterior_override = max(posterior_override, eval_scale_floor)
    if args.resume and not args.walkforward and not args.fresh_agent:
        resume_from(run_dir, agent, trainer)
    web_dashboard = None
    if args.web_dashboard:
        try:
            tf_min_for_web = timeframe_to_minutes(args.timeframe)
        except Exception:
            tf_min_for_web = None
        if tf_min_for_web and tf_min_for_web > 0:
            web_history = max(1, math.ceil((args.web_window_hours * 60.0) / tf_min_for_web))
        else:
            web_history = 500
        web_dashboard = WebDashboard(
            port=args.web_port,
            history=web_history,
            window_hours=args.web_window_hours,
        )
    if web_dashboard:
        web_dashboard.start()

    def publish_web_event(event, trainer: Trainer, agent: RLSForgettingAgent) -> None:
        if not web_dashboard:
            return
        step, row, result, portfolio_value, mtm_delta, trade_impact = event
        price = float(row["close"])
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
            trade_size=result.trade_size,
            notional_traded=result.notional_traded,
            refilled=result.refilled,
            refill_count=trainer.refill_count,
            executed_trades=trainer.agent.state.trades,
            sell_win_rate=trainer.trade_win_rate,
            realized_pnl=result.realized_pnl,
            data_is_live=trainer.last_data_is_live or False,
            total_return=trainer.total_return,
            sharpe_ratio=trainer.sharpe_ratio,
            max_drawdown=trainer.max_drawdown,
            action_distribution=trainer.action_distribution,
            avg_win_pnl=getattr(trainer, 'avg_win_pnl', 0.0),
            avg_loss_pnl=getattr(trainer, 'avg_loss_pnl', 0.0),
            win_loss_ratio=getattr(trainer, 'win_loss_ratio', 0.0),
            expectancy_pnl_per_sell_leg=getattr(trainer, 'expectancy_pnl_per_sell_leg', 0.0),
        )

    if args.walkforward:
        if args.continuous:
            print("--walkforward disables --continuous; running offline-style walk-forward instead.")
            args.continuous = False
        feature_frame, is_live = feed.fetch(include_indicators=True)
        folds = max(2, int(args.walkforward_folds or 0) or config.WALKFORWARD_FOLDS)
        if args.dashboard:
            print("Note: walk-forward streams only to the web dashboard; terminal dashboard is disabled.")
        try:
            results = run_walkforward(
                feature_frame,
                folds,
                build_trainer=build_trainer,
                run_dir=run_dir,
                data_is_live=is_live,
                posterior_scale_override=posterior_override,
                delay=args.delay,
                event_callback=publish_web_event if web_dashboard else None,
            )
        except ValueError as exc:
            print(f"Walk-forward aborted: {exc}")
            return
        out_path = run_dir / "walkforward.json"
        out_path.write_text(json.dumps(results, indent=2))
        if results:
            win_rates = [r["trade_win_rate"] for r in results]
            expectancies = [r["expectancy_pnl_per_sell_leg"] for r in results]
            avg_losses = [r["avg_loss_pnl"] for r in results]
            print(
                "Walk-forward summary: "
                f"folds={len(results)}, "
                f"win_rate={statistics.fmean(win_rates):.4f}, "
                f"avg_loss={statistics.fmean(avg_losses):+.4f}, "
                f"expectancy={statistics.fmean(expectancies):+.4f}"
            )
        return

    if args.continuous:
        feature_frame, is_live = feed.fetch(include_indicators=True)

        warmup_target = trainer.initial_cash * (1 + args.warmup_profit_target / 100)
        warmup_steps_effective = warmup_steps
        def warmup_then_stream():
            warmup_hit = False

            if warmup_steps_effective > 0:
                print(
                    f"Starting warmup for {warmup_steps_effective} steps on the latest window "
                    f"(target portfolio >= {warmup_target:.2f})."
                )
                for event in run_loop(
                    trainer,
                    feature_frame,
                    warmup_steps_effective,
                    None,
                    args.delay,
                    run_id=run_id,
                    run_dir=run_dir,
                    checkpoint_every=args.checkpoint_every,
                    flush_trades_every=args.flush_trades_every,
                    keep_last=args.keep_last,
                    data_is_live=is_live,
                    posterior_scale_override=posterior_override,
                    cycle_window=args.cycle_window,
                ):
                    yield event
                    _, _, _, portfolio_value, _, _ = event
                    if portfolio_value >= warmup_target:
                        warmup_hit = True
                        print(
                            f"Warmup profit target reached (portfolio {portfolio_value:.2f} >= {warmup_target:.2f})."
                        )

                if not warmup_hit:
                    print(
                        f"Warmup ended without reaching the profit target ({warmup_target:.2f}). "
                        "Continuing to live stream anyway."
                    )

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
                posterior_scale_override=posterior_override,
            )

        loop = warmup_then_stream()
    else:
        feature_frame, is_live = feed.fetch(include_indicators=True)
        loop_checkpoint_every = 0 if eval_mode else args.checkpoint_every
        loop_flush_every = 0 if eval_mode else args.flush_trades_every
        loop = run_loop(
            trainer,
            feature_frame,
            args.steps,
            args.duration,
            args.delay,
            run_id=run_id,
            run_dir=run_dir,
            checkpoint_every=loop_checkpoint_every,
            flush_trades_every=loop_flush_every,
            keep_last=args.keep_last,
            data_is_live=is_live,
            train=not eval_mode,
            posterior_scale_override=posterior_override,
            cycle_window=args.cycle_window,
        )

    try:
        if eval_mode:
            executed_trades = 0
            final_value = trainer.portfolio.value(float(feature_frame.iloc[-1]["close"])) if len(feature_frame) else 0.0
            for _, row, result, portfolio_value, _, _ in loop:
                final_value = portfolio_value
                if result.trade_executed:
                    executed_trades += 1
            sell_win_rate = trainer.trade_win_rate * 100
            print(
                f"Evaluation complete: portfolio value={final_value:.2f}, executed trades={executed_trades}, "
                f"sell win rate={sell_win_rate:.2f}%"
            )
        elif args.dashboard:
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
                            trade_size=result.trade_size,
                            notional_traded=result.notional_traded,
                            refilled=result.refilled,
                            refill_count=trainer.refill_count,
                            executed_trades=trainer.agent.state.trades,
                            sell_win_rate=trainer.trade_win_rate,
                            realized_pnl=result.realized_pnl,
                            data_is_live=trainer.last_data_is_live or False,
                            total_return=trainer.total_return,
                            sharpe_ratio=trainer.sharpe_ratio,
                            max_drawdown=trainer.max_drawdown,
                            action_distribution=trainer.action_distribution,
                            avg_win_pnl=getattr(trainer, 'avg_win_pnl', 0.0),
                            avg_loss_pnl=getattr(trainer, 'avg_loss_pnl', 0.0),
                            win_loss_ratio=getattr(trainer, 'win_loss_ratio', 0.0),
                            expectancy_pnl_per_sell_leg=getattr(trainer, 'expectancy_pnl_per_sell_leg', 0.0),
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
                        trainer.sharpe_ratio,
                        trainer.total_return,
                        trainer.max_drawdown,
                        trainer.action_distribution,
                        trainer.gate_blocks,
                        trainer.timing_blocks,
                        trainer.budget_blocks,
                        getattr(trainer, 'avg_win_pnl', 0.0),
                        getattr(trainer, 'avg_loss_pnl', 0.0),
                        getattr(trainer, 'win_loss_ratio', 0.0),
                        getattr(trainer, 'expectancy_pnl_per_sell_leg', 0.0),
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
                            trade_size=result.trade_size,
                            notional_traded=result.notional_traded,
                            refilled=result.refilled,
                            refill_count=trainer.refill_count,
                            executed_trades=trainer.agent.state.trades,
                            sell_win_rate=trainer.trade_win_rate,
                            realized_pnl=result.realized_pnl,
                            data_is_live=trainer.last_data_is_live or False,
                            total_return=trainer.total_return,
                            sharpe_ratio=trainer.sharpe_ratio,
                            max_drawdown=trainer.max_drawdown,
                            action_distribution=trainer.action_distribution,
                        )
                    yield

            for _ in emit(loop):
                pass
    except KeyboardInterrupt:
        print("Interrupted; saving agent state before exit...")
    finally:
        if not eval_mode:
            trainer._flush_trades_and_metrics(run_dir, force=True, data_is_live=trainer.last_data_is_live)
            trainer.agent.save(run_dir=run_dir, keep_last=args.keep_last)
            trainer._save_trainer_state(run_dir, run_id, checkpoint=False, keep_last=args.keep_last)

        if web_dashboard:
            web_dashboard.stop()


if __name__ == "__main__":
    main()
