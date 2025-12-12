from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd

from src import config
from src.agent import BanditAgent
from src.data_feed import DataFeed, MarketConfig
from src.metrics import compute_max_drawdown
from src.trainer import Trainer
from src.persistence import atomic_write_text


@dataclass
class Metrics:
    final_value: float
    executed_trades: int
    total_reward: float
    max_drawdown: float
    realized_pnl: float
    baseline_final_value: float
    ma_baseline_final_value: float


def timeframe_to_minutes(timeframe: str) -> float:
    if not timeframe:
        return 1.0

    unit = timeframe[-1].lower()
    try:
        quantity = float(timeframe[:-1])
    except ValueError:
        return 1.0

    multiplier = {"m": 1, "h": 60, "d": 24 * 60}.get(unit)
    return quantity * multiplier if multiplier else 1.0


def fetch_last_24h(symbol: str, timeframe: str, offline: bool, cache: bool, cache_only: bool, cache_dir: Path) -> Tuple[pd.DataFrame, bool]:
    minutes = timeframe_to_minutes(timeframe)
    candles_needed = math.ceil((24 * 60) / minutes)
    feed = DataFeed(
        MarketConfig(
            symbol=symbol,
            timeframe=timeframe,
            limit=candles_needed,
            offline=offline,
            cache=cache,
            cache_only=cache_only,
            cache_dir=cache_dir,
        )
    )
    frame, is_live = feed.fetch(include_indicators=True)
    return frame, is_live


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    expected = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in fetched data: {sorted(missing)}")

    cleaned = frame.dropna(subset=list(expected)).copy()
    if "timestamp" in cleaned.columns:
        cleaned = cleaned.sort_values("timestamp").drop_duplicates(subset="timestamp")
    return cleaned.reset_index(drop=True)


def buy_and_hold_baseline(frame: pd.DataFrame, initial_cash: float) -> float:
    if frame.empty:
        return initial_cash
    start_price = float(frame.iloc[0]["close"])
    end_price = float(frame.iloc[-1]["close"])

    buy_fee = initial_cash * config.FEE_RATE
    investable = initial_cash - buy_fee
    turnover_penalty = investable * config.TURNOVER_PENALTY
    position_size = max(0.0, investable - turnover_penalty) / max(start_price, 1e-6)

    gross_proceeds = position_size * end_price
    sell_fee = gross_proceeds * config.FEE_RATE
    sell_turnover = gross_proceeds * config.TURNOVER_PENALTY
    final_cash = gross_proceeds - sell_fee - sell_turnover
    return final_cash


def moving_average_crossover_baseline(
    frame: pd.DataFrame, initial_cash: float, fast: int = 10, slow: int = 30
) -> float:
    if frame.empty or len(frame) < slow:
        return initial_cash

    prices = frame["close"].astype(float)
    fast_ma = prices.rolling(fast, min_periods=1).mean()
    slow_ma = prices.rolling(slow, min_periods=1).mean()

    cash = initial_cash
    position = 0.0
    entry_price = 0.0

    for idx in range(1, len(frame)):
        price = float(prices.iloc[idx])
        prev_fast, prev_slow = float(fast_ma.iloc[idx - 1]), float(slow_ma.iloc[idx - 1])
        curr_fast, curr_slow = float(fast_ma.iloc[idx]), float(slow_ma.iloc[idx])

        cross_up = prev_fast <= prev_slow and curr_fast > curr_slow
        cross_down = prev_fast >= prev_slow and curr_fast < curr_slow

        if cross_up and cash > 0:
            fee = cash * config.FEE_RATE
            investable_base = cash - fee
            turnover_penalty = investable_base * config.TURNOVER_PENALTY
            investable = max(0.0, investable_base - turnover_penalty)
            position = investable / max(price, 1e-6)
            cash = 0.0
            entry_price = price
        elif cross_down and position > 0:
            gross = position * price
            fee = gross * config.FEE_RATE
            turnover_penalty = gross * config.TURNOVER_PENALTY
            cash = gross - fee - turnover_penalty
            position = 0.0
            entry_price = 0.0

    if position > 0:
        gross = position * float(prices.iloc[-1])
        fee = gross * config.FEE_RATE
        turnover_penalty = gross * config.TURNOVER_PENALTY
        cash = gross - fee - turnover_penalty
    return cash


def evaluate_agent(
    trainer: Trainer,
    frame: pd.DataFrame,
    *,
    initial_cash: float,
    run_dir: Path,
    run_id: str,
    data_is_live: bool,
    baseline_final_value: float | None = None,
    ma_baseline_final_value: float | None = None,
) -> Metrics:
    eval_trainer = Trainer(trainer.agent, initial_cash=initial_cash)
    eval_trainer.last_data_is_live = data_is_live

    equity_curve: list[float] = []
    executed_trades = 0
    realized_pnl = 0.0

    for idx in range(max(0, len(frame) - 1)):
        row = frame.iloc[idx]
        next_row = frame.iloc[idx + 1]
        price_now = float(row["close"])
        result = eval_trainer.step(row, next_row, idx, train=False, posterior_scale_override=0.0)
        executed_trades += 1 if result.trade_executed else 0
        realized_pnl += result.realized_pnl
        equity_curve.append(eval_trainer.portfolio.value(price_now))

    final_price = float(frame.iloc[-1]["close"]) if not frame.empty else 0.0
    final_value = eval_trainer.portfolio.value(final_price)
    max_drawdown = compute_max_drawdown(equity_curve)

    eval_trainer._flush_trades_and_metrics(
        run_dir,
        force=True,
        data_is_live=data_is_live,
        baseline_final_value=baseline_final_value,
        val_final_value=final_value if baseline_final_value is not None else None,
        max_drawdown=max_drawdown,
        executed_trades=executed_trades,
        ma_baseline_final_value=ma_baseline_final_value,
    )

    total_reward = sum(r for *_, r in eval_trainer.history)
    return Metrics(
        final_value=final_value,
        executed_trades=executed_trades,
        total_reward=total_reward,
        max_drawdown=max_drawdown,
        realized_pnl=realized_pnl,
        baseline_final_value=baseline_final_value or 0.0,
        ma_baseline_final_value=ma_baseline_final_value or 0.0,
    )


def snapshot_state(path: Path) -> str | None:
    return path.read_text() if path.exists() else None


def restore_state(path: Path, snapshot: str | None) -> None:
    if snapshot is None:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, snapshot)


def split_frame(frame: pd.DataFrame, validation_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = max(1, int(len(frame) * (1 - validation_fraction)))
    return frame.iloc[:cutoff], frame.iloc[cutoff:]


def train_agent(
    feature_frame: pd.DataFrame,
    steps: int | None,
    initial_cash: float,
    *,
    run_id: str,
    run_dir: Path,
    data_is_live: bool,
) -> Trainer:
    agent = BanditAgent()
    trainer = Trainer(agent, initial_cash=initial_cash)
    trainer.run(
        feature_frame,
        max_steps=steps,
        run_id=run_id,
        run_dir=run_dir,
        checkpoint_every=config.DEFAULT_CHECKPOINT_EVERY,
        flush_trades_every=config.DEFAULT_FLUSH_TRADES_EVERY,
        keep_last=config.DEFAULT_KEEP_LAST_CHECKPOINTS,
        data_is_live=data_is_live,
    )
    return trainer


def run_cycle(args: argparse.Namespace, cycle: int, run_dir: Path, run_id: str) -> bool:
    print(f"\n[cycle {cycle}] Fetching last 24h of {args.symbol} ({args.timeframe})...")
    raw_frame, is_live = fetch_last_24h(args.symbol, args.timeframe, args.offline, args.cache, args.cache_only, args.cache_dir)
    validated = validate_frame(raw_frame)

    if validated.empty:
        raise RuntimeError("No rows available after indicator computation; try increasing --timeframe limit.")

    train_frame, val_frame = split_frame(validated, args.validation_fraction)
    if val_frame.empty:
        raise RuntimeError("Validation split produced an empty frame; reduce --validation-fraction.")

    state_path = Path(config.STATE_PATH)
    backup = snapshot_state(state_path)

    trainer = train_agent(
        train_frame,
        steps=args.train_steps,
        initial_cash=args.initial_cash,
        run_id=run_id,
        run_dir=run_dir,
        data_is_live=is_live,
    )

    val_baseline = buy_and_hold_baseline(val_frame, args.initial_cash)
    val_ma_baseline = moving_average_crossover_baseline(val_frame, args.initial_cash)
    val_metrics = evaluate_agent(
        trainer,
        val_frame,
        initial_cash=args.initial_cash,
        run_dir=run_dir,
        run_id=run_id,
        data_is_live=is_live,
        baseline_final_value=val_baseline,
        ma_baseline_final_value=val_ma_baseline,
    )
    backtest_metrics = evaluate_agent(
        trainer,
        validated,
        initial_cash=args.initial_cash,
        run_dir=run_dir,
        run_id=run_id,
        data_is_live=is_live,
        baseline_final_value=buy_and_hold_baseline(validated, args.initial_cash),
        ma_baseline_final_value=moving_average_crossover_baseline(validated, args.initial_cash),
    )

    threshold_value = val_baseline * (1 + args.min_profit_threshold / 100)
    print(
        "  Val:      "
        f"trades={val_metrics.executed_trades} | final_value={val_metrics.final_value:.2f} | "
        f"buy&hold={val_baseline:.2f} | ma_crossover={val_ma_baseline:.2f} | "
        f"max_drawdown={val_metrics.max_drawdown:.3f}"
    )
    print(
        "  Backtest: "
        f"trades={backtest_metrics.executed_trades} | final_value={backtest_metrics.final_value:.2f} | "
        f"buy&hold={backtest_metrics.baseline_final_value:.2f} | "
        f"ma_crossover={backtest_metrics.ma_baseline_final_value:.2f} | "
        f"max_drawdown={backtest_metrics.max_drawdown:.3f}"
    )

    drawdown_ok = args.max_drawdown is None or val_metrics.max_drawdown <= args.max_drawdown
    trades_ok = args.max_trades is None or val_metrics.executed_trades <= args.max_trades
    profitable = val_metrics.final_value >= threshold_value

    if profitable and drawdown_ok and trades_ok:
        print(
            f"Cycle {cycle} passed threshold (val {val_metrics.final_value:.2f} >= baseline target {threshold_value:.2f}). "
            "Keeping updated agent state."
        )
        return True

    print(
        f"Cycle {cycle} below threshold (val {val_metrics.final_value:.2f} < target {threshold_value:.2f}"
        f" or drawdown/trade limits exceeded). Restoring previous agent state and trying again."
    )
    restore_state(state_path, backup)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-retrain on the last 24h of market data until profit target is met.")
    parser.add_argument("--symbol", default=config.DEFAULT_SYMBOL, help="Market pair to train on (e.g., BTC/USDT)")
    parser.add_argument("--timeframe", default=config.DEFAULT_TIMEFRAME, help="Candle timeframe (e.g., 1m, 5m, 1h)")
    parser.add_argument("--offline", action="store_true", help="Use synthetic candles instead of live exchange data")
    parser.add_argument(
        "--min-profit-threshold",
        type=float,
        default=0.0,
        help="Validation uplift over baseline (percent) required",
    )
    parser.add_argument("--max-drawdown", type=float, default=None, help="Maximum acceptable validation drawdown (fraction)")
    parser.add_argument("--max-trades", type=int, default=None, help="Optional cap on validation trades")
    parser.add_argument("--max-cycles", type=int, default=5, help="Maximum retraining attempts before giving up")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="Portion of data reserved for validation")
    parser.add_argument("--train-steps", type=int, default=None, help="Optional cap on training steps (defaults to full window)")
    parser.add_argument("--initial-cash", type=float, default=1000.0, help="Starting cash balance for simulations")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Pause between cycles to avoid API pressure")
    parser.add_argument("--cache", action="store_true", help="Cache fetched datasets for reproducibility")
    parser.add_argument("--cache-only", action="store_true", help="Load data exclusively from cache")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"), help="Directory to store cached datasets")
    parser.add_argument("--run-id", default=None, help="Identifier for this auto-retrain run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_run_dir = Path(config.RUNS_DIR)
    base_run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"auto_{args.symbol.replace('/', '_')}_{args.timeframe}_{timestamp}"
    run_dir = base_run_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    for cycle in range(1, args.max_cycles + 1):
        try:
            if run_cycle(args, cycle, run_dir, run_id):
                break
        except Exception as exc:  # pragma: no cover - defensive logging path
            print(f"Cycle {cycle} failed: {exc}")
            if cycle >= args.max_cycles:
                raise

        if cycle < args.max_cycles and args.sleep_seconds > 0:
            print(f"Sleeping {args.sleep_seconds:.1f}s before next cycle...")
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
