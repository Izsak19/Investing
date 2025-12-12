from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd

from src import config
from src.agent import BanditAgent
from src.data_feed import DataFeed, MarketConfig
from src.indicators import INDICATOR_COLUMNS, compute_indicators
from src.trainer import Portfolio, Trainer


@dataclass
class Metrics:
    success_rate: float
    trades: int
    total_reward: float
    final_value: float


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


def fetch_last_24h(symbol: str, timeframe: str, offline: bool) -> pd.DataFrame:
    minutes = timeframe_to_minutes(timeframe)
    candles_needed = math.ceil((24 * 60) / minutes)
    feed = DataFeed(MarketConfig(symbol=symbol, timeframe=timeframe, limit=candles_needed, offline=offline))
    return feed.fetch()


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    expected = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in fetched data: {sorted(missing)}")

    cleaned = frame.dropna(subset=list(expected)).copy()
    if "timestamp" in cleaned.columns:
        cleaned = cleaned.sort_values("timestamp").drop_duplicates(subset="timestamp")
    return cleaned.reset_index(drop=True)


def train_agent(feature_frame: pd.DataFrame, steps: int | None, initial_cash: float) -> Trainer:
    agent = BanditAgent()
    trainer = Trainer(agent, initial_cash=initial_cash)
    trainer.run(feature_frame, max_steps=steps)
    return trainer


def simulate_agent(agent: BanditAgent, frame: pd.DataFrame, initial_cash: float) -> Metrics:
    portfolio = Portfolio(cash=initial_cash)
    rewards: list[float] = []
    successful = 0

    for _, row in frame.iterrows():
        price = float(row["close"])
        features = row[INDICATOR_COLUMNS].to_numpy(dtype=float)
        action = agent.act(features)
        reward = 0.0

        if action == "buy" and portfolio.cash > 0:
            fee = portfolio.cash * config.FEE_RATE
            investable = portfolio.cash - fee
            portfolio.position = investable / price
            portfolio.entry_price = price / (1 - config.FEE_RATE)
            portfolio.cash = 0.0
        elif action == "sell" and portfolio.position > 0:
            gross_proceeds = portfolio.position * price
            fee = gross_proceeds * config.FEE_RATE
            net_proceeds = gross_proceeds - fee
            reward = net_proceeds - portfolio.entry_price * portfolio.position
            portfolio.cash = net_proceeds
            portfolio.position = 0.0
            portfolio.entry_price = 0.0
        else:
            reward = (price - portfolio.entry_price) * portfolio.position

        rewards.append(reward)
        if reward > 0:
            successful += 1

    final_value = portfolio.value(float(frame.iloc[-1]["close"])) if not frame.empty else portfolio.cash
    success_rate = (successful / len(rewards) * 100) if rewards else 0.0
    return Metrics(success_rate=success_rate, trades=len(rewards), total_reward=sum(rewards), final_value=final_value)


def snapshot_state(path: Path) -> str | None:
    return path.read_text() if path.exists() else None


def restore_state(path: Path, snapshot: str | None) -> None:
    if snapshot is None:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(snapshot)


def split_frame(frame: pd.DataFrame, validation_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = max(1, int(len(frame) * (1 - validation_fraction)))
    return frame.iloc[:cutoff], frame.iloc[cutoff:]


def run_cycle(args: argparse.Namespace, cycle: int) -> bool:
    print(f"\n[cycle {cycle}] Fetching last 24h of {args.symbol} ({args.timeframe})...")
    raw_frame = fetch_last_24h(args.symbol, args.timeframe, args.offline)
    validated = validate_frame(raw_frame)
    features = compute_indicators(validated)

    if features.empty:
        raise RuntimeError("No rows available after indicator computation; try increasing --timeframe limit.")

    train_frame, val_frame = split_frame(features, args.validation_fraction)
    if val_frame.empty:
        raise RuntimeError("Validation split produced an empty frame; reduce --validation-fraction.")

    state_path = Path(config.STATE_PATH)
    backup = snapshot_state(state_path)

    trainer = train_agent(train_frame, steps=args.train_steps, initial_cash=args.initial_cash)
    train_metrics = Metrics(
        success_rate=trainer.success_rate,
        trades=len(trainer.history),
        total_reward=sum(r for *_, r in trainer.history),
        final_value=trainer.portfolio.value(float(train_frame.iloc[-1]["close"])) if not train_frame.empty else trainer.portfolio.cash,
    )

    val_metrics = simulate_agent(trainer.agent, val_frame, args.initial_cash)
    backtest_metrics = simulate_agent(trainer.agent, features, args.initial_cash)

    print(
        "  Train:    "
        f"success={train_metrics.success_rate:.2f}% | trades={train_metrics.trades} | "
        f"total_reward={train_metrics.total_reward:.4f} | final_value={train_metrics.final_value:.2f}"
    )
    print(
        "  Val:      "
        f"success={val_metrics.success_rate:.2f}% | trades={val_metrics.trades} | "
        f"total_reward={val_metrics.total_reward:.4f} | final_value={val_metrics.final_value:.2f}"
    )
    print(
        "  Backtest: "
        f"success={backtest_metrics.success_rate:.2f}% | trades={backtest_metrics.trades} | "
        f"total_reward={backtest_metrics.total_reward:.4f} | final_value={backtest_metrics.final_value:.2f}"
    )

    if val_metrics.success_rate >= args.min_success_rate:
        print(
            f"Cycle {cycle} passed threshold ({val_metrics.success_rate:.2f}% >= {args.min_success_rate}%). "
            "Keeping updated agent state."
        )
        return True

    print(
        f"Cycle {cycle} below threshold ({val_metrics.success_rate:.2f}% < {args.min_success_rate}%). "
        "Restoring previous agent state and trying again."
    )
    restore_state(state_path, backup)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-retrain on the last 24h of market data until success target is met.")
    parser.add_argument("--symbol", default=config.DEFAULT_SYMBOL, help="Market pair to train on (e.g., BTC/USDT)")
    parser.add_argument("--timeframe", default=config.DEFAULT_TIMEFRAME, help="Candle timeframe (e.g., 1m, 5m, 1h)")
    parser.add_argument("--offline", action="store_true", help="Use synthetic candles instead of live exchange data")
    parser.add_argument("--min-success-rate", type=float, default=55.0, help="Validation success rate required to stop retraining")
    parser.add_argument("--max-cycles", type=int, default=5, help="Maximum retraining attempts before giving up")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="Portion of data reserved for validation")
    parser.add_argument("--train-steps", type=int, default=None, help="Optional cap on training steps (defaults to full window)")
    parser.add_argument("--initial-cash", type=float, default=1000.0, help="Starting cash balance for simulations")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Pause between cycles to avoid API pressure")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for cycle in range(1, args.max_cycles + 1):
        try:
            if run_cycle(args, cycle):
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
