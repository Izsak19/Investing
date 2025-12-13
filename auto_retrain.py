# src/auto_retrain.py
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple, Iterable

import numpy as np
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
        qty = float(timeframe[:-1])
    except ValueError:
        return 1.0
    mult = {"m": 1, "h": 60, "d": 24 * 60}.get(unit)
    return qty * mult if mult else 1.0

def fetch_last_24h(symbol: str, timeframe: str, offline: bool, cache: bool, cache_only: bool, cache_dir: Path) -> Tuple[pd.DataFrame, bool]:
    minutes = timeframe_to_minutes(timeframe)
    candles_needed = math.ceil((24 * 60) / minutes)
    feed = DataFeed(MarketConfig(symbol=symbol, timeframe=timeframe, limit=candles_needed,
                                 offline=offline, cache=cache, cache_only=cache_only, cache_dir=cache_dir))
    return feed.fetch(include_indicators=True)

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
    turn = investable * config.TURNOVER_PENALTY
    size = max(0.0, investable - turn) / max(start_price, 1e-6)
    gross = size * end_price
    sell_fee = gross * config.FEE_RATE
    sell_turn = gross * config.TURNOVER_PENALTY
    return gross - sell_fee - sell_turn

def moving_average_crossover_baseline(frame: pd.DataFrame, initial_cash: float, fast: int = 10, slow: int = 30) -> float:
    if frame.empty or len(frame) < slow:
        return initial_cash
    prices = frame["close"].astype(float)
    fast_ma = prices.rolling(fast, min_periods=1).mean()
    slow_ma = prices.rolling(slow, min_periods=1).mean()
    cash = initial_cash
    position = 0.0
    for i in range(1, len(frame)):
        price = float(prices.iloc[i])
        prev_fast, prev_slow = float(fast_ma.iloc[i-1]), float(slow_ma.iloc[i-1])
        curr_fast, curr_slow = float(fast_ma.iloc[i]), float(slow_ma.iloc[i])
        cross_up = prev_fast <= prev_slow and curr_fast > curr_slow
        cross_down = prev_fast >= prev_slow and curr_fast < curr_slow
        if cross_up and cash > 0:
            fee = cash * config.FEE_RATE
            base = cash - fee
            turn = base * config.TURNOVER_PENALTY
            investable = max(0.0, base - turn)
            position = investable / max(price, 1e-6)
            cash = 0.0
        elif cross_down and position > 0:
            gross = position * price
            fee = gross * config.FEE_RATE
            turn = gross * config.TURNOVER_PENALTY
            cash = gross - fee - turn
            position = 0.0
    if position > 0:
        gross = position * float(prices.iloc[-1])
        fee = gross * config.FEE_RATE
        turn = gross * config.TURNOVER_PENALTY
        cash = gross - fee - turn
    return cash

def _walk_forward_returns_from_curve(equity_curve: Iterable[float], folds: int) -> list[float]:
    curve = list(equity_curve)
    if folds <= 1 or len(curve) < folds + 1:
        return []
    n = len(curve) // folds
    if n <= 0:
        return []
    out: list[float] = []
    s = 0
    for k in range(folds):
        e = (k + 1) * n if k < folds - 1 else len(curve)
        if e - s < 2:
            continue
        out.append((curve[e-1] - curve[s]) / max(curve[s], 1e-9))
        s = e
    return out

def evaluate_agent(trainer: Trainer, frame: pd.DataFrame, *,
                   initial_cash: float, run_dir: Path, run_id: str, data_is_live: bool,
                   baseline_final_value: float | None = None,
                   ma_baseline_final_value: float | None = None) -> Metrics:
    eval_tr = Trainer(trainer.agent, initial_cash=initial_cash)
    eval_tr.last_data_is_live = data_is_live
    equity_curve: list[float] = []
    executed_trades = 0
    realized_pnl = 0.0
    for i in range(max(0, len(frame) - 1)):
        row = frame.iloc[i]
        nxt = frame.iloc[i + 1]
        price_now = float(row["close"])
        result = eval_tr.step(row, nxt, i, train=False, posterior_scale_override=0.0)
        executed_trades += 1 if result.trade_executed else 0
        realized_pnl += result.realized_pnl
        equity_curve.append(eval_tr.portfolio.value(price_now))
    final_price = float(frame.iloc[-1]["close"]) if not frame.empty else 0.0
    final_value = eval_tr.portfolio.value(final_price)
    max_dd = compute_max_drawdown(equity_curve)

    # persist run metrics
    eval_tr._flush_trades_and_metrics(
        run_dir, force=True, data_is_live=data_is_live,
        baseline_final_value=baseline_final_value,
        val_final_value=final_value if baseline_final_value is not None else None,
        max_drawdown=max_dd, executed_trades=executed_trades,
        ma_baseline_final_value=ma_baseline_final_value,
    )
    total_reward = sum(r for *_, r in eval_tr.history)
    return Metrics(
        final_value=final_value, executed_trades=executed_trades,
        total_reward=total_reward, max_drawdown=max_dd, realized_pnl=realized_pnl,
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

def train_agent(feature_frame: pd.DataFrame, steps: int | None, initial_cash: float, *,
                run_id: str, run_dir: Path, data_is_live: bool) -> Trainer:
    agent = BanditAgent()
    trainer = Trainer(agent, initial_cash=initial_cash)
    trainer.run(
        feature_frame, max_steps=steps, run_id=run_id, run_dir=run_dir,
        checkpoint_every=config.DEFAULT_CHECKPOINT_EVERY,
        flush_trades_every=config.DEFAULT_FLUSH_TRADES_EVERY,
        keep_last=config.DEFAULT_KEEP_LAST_CHECKPOINTS,
        data_is_live=data_is_live,
    )
    return trainer

def run_cycle(args: argparse.Namespace, cycle: int, run_dir: Path, run_id: str) -> bool:
    print(f"\n[cycle {cycle}] Fetching last 24h of {args.symbol} ({args.timeframe}).")
    raw, is_live = fetch_last_24h(args.symbol, args.timeframe, args.offline, args.cache, args.cache_only, args.cache_dir)
    validated = validate_frame(raw)
    if validated.empty:
        raise RuntimeError("No rows available after indicator computation; try increasing --timeframe limit.")
    train_frame, val_frame = split_frame(validated, args.validation_fraction)
    if val_frame.empty:
        raise RuntimeError("Validation split produced an empty frame; reduce --validation-fraction.")

    state_path = Path(config.STATE_PATH)
    backup = snapshot_state(state_path)

    trainer = train_agent(train_frame, steps=args.train_steps, initial_cash=args.initial_cash,
                          run_id=run_id, run_dir=run_dir, data_is_live=is_live)

    val_bh = buy_and_hold_baseline(val_frame, args.initial_cash)
    val_ma = moving_average_crossover_baseline(val_frame, args.initial_cash)
    val_m = evaluate_agent(trainer, val_frame, initial_cash=args.initial_cash, run_dir=run_dir,
                           run_id=run_id, data_is_live=is_live,
                           baseline_final_value=val_bh, ma_baseline_final_value=val_ma)
    back_m = evaluate_agent(trainer, validated, initial_cash=args.initial_cash, run_dir=run_dir,
                            run_id=run_id, data_is_live=is_live,
                            baseline_final_value=buy_and_hold_baseline(validated, args.initial_cash),
                            ma_baseline_final_value=moving_average_crossover_baseline(validated, args.initial_cash))

    thr_value = val_bh * (1 + args.min_profit_threshold / 100.0)

    # walk-forward guard on validation equity
    # (re-use the validation pass equity curve by recomputing quickly)
    # For speed/clarity we compute folds on price-proxy curve using the agent again with TD disabled.
    wf_equity = []
    probe = Trainer(trainer.agent, initial_cash=args.initial_cash)
    for i in range(max(0, len(val_frame) - 1)):
        row = val_frame.iloc[i]
        nxt = val_frame.iloc[i + 1]
        price_now = float(row["close"])
        probe.step(row, nxt, i, train=False, posterior_scale_override=0.0)
        wf_equity.append(probe.portfolio.value(price_now))
    wf = _walk_forward_returns_from_curve(wf_equity, config.WALKFORWARD_FOLDS)
    wf_min = float(min(wf)) if wf else 0.0

    print(
        "  Val:      "
        f"trades={val_m.executed_trades} | final_value={val_m.final_value:.2f} | "
        f"buy&hold={val_bh:.2f} | ma_crossover={val_ma:.2f} | max_drawdown={val_m.max_drawdown:.3f} | "
        f"walkforward_min={wf_min:+.2%}"
    )
    print(
        "  Backtest: "
        f"trades={back_m.executed_trades} | final_value={back_m.final_value:.2f} | "
        f"buy&hold={back_m.baseline_final_value:.2f} | ma_crossover={back_m.ma_baseline_final_value:.2f} | "
        f"max_drawdown={back_m.max_drawdown:.3f}"
    )

    drawdown_ok = args.max_drawdown is None or val_m.max_drawdown <= args.max_drawdown
    trades_ok = args.max_trades is None or val_m.executed_trades <= args.max_trades
    profitable = val_m.final_value >= thr_value
    wf_ok = (not args.require_walkforward_nonnegative) or (wf_min >= 0.0)

    if profitable and drawdown_ok and trades_ok and wf_ok:
        print(f"Cycle {cycle} passed threshold (val {val_m.final_value:.2f} >= target {thr_value:.2f}). Keeping updated agent state.")
        return True

    print(
        f"Cycle {cycle} below threshold (val {val_m.final_value:.2f} < target {thr_value:.2f} "
        f"or drawdown/trade/walkforward limits exceeded). Restoring previous agent state."
    )
    restore_state(state_path, backup)
    return False

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-retrain on the last 24h of market data until profit target is met.")
    p.add_argument("--symbol", default=config.DEFAULT_SYMBOL)
    p.add_argument("--timeframe", default=config.DEFAULT_TIMEFRAME)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--min-profit-threshold", type=float, default=0.0)
    p.add_argument("--max-drawdown", type=float, default=None)
    p.add_argument("--max-trades", type=int, default=None)
    p.add_argument("--max-cycles", type=int, default=5)
    p.add_argument("--validation-fraction", type=float, default=0.2)
    p.add_argument("--train-steps", type=int, default=None)
    p.add_argument("--initial-cash", type=float, default=1000.0)
    p.add_argument("--sleep-seconds", type=float, default=0.0)
    p.add_argument("--cache", action="store_true")
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument("--run-id", default=None)
    p.add_argument("--require-walkforward-nonnegative", action="store_true",
                   help="Only accept updates if min walk-forward fold return on validation is >= 0.")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    base = Path(config.RUNS_DIR); base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"auto_{args.symbol.replace('/', '_')}_{args.timeframe}_{timestamp}"
    run_dir = base / run_id; run_dir.mkdir(parents=True, exist_ok=True)
    for cycle in range(1, args.max_cycles + 1):
        try:
            if run_cycle(args, cycle, run_dir, run_id):
                break
        except Exception as exc:
            print(f"Cycle {cycle} failed: {exc}")
            if cycle >= args.max_cycles:
                raise
        if cycle < args.max_cycles and args.sleep_seconds > 0:
            print(f"Sleeping {args.sleep_seconds:.1f}s before next cycle.")
            time.sleep(args.sleep_seconds)

if __name__ == "__main__":
    main()
