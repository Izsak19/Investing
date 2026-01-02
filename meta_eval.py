from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data_feed import DataFeed, MarketConfig
import baseline_eval


@dataclass
class SelectorResult:
    total_return: float
    trade_count: int
    win_rate: float
    avg_win_pnl: float
    avg_loss_pnl: float
    expectancy: float
    max_drawdown: float
    total_fee_paid: float
    total_slippage_paid: float


def _simulate_curve(
    prices: np.ndarray,
    targets: np.ndarray,
    *,
    steps: int,
    fee_rate: float,
    slippage_rate: float,
    initial_cash: float,
) -> list[float]:
    equity = float(initial_cash)
    position = 0
    entry_price = 0.0
    entry_equity = equity
    curve: list[float] = []

    for step in range(steps):
        price = float(prices[step])
        target = int(targets[step])
        target = 1 if target > 0 else -1 if target < 0 else 0

        if target != position:
            if position != 0:
                if position > 0:
                    gross = equity * (price / max(entry_price, 1e-9))
                else:
                    gross = equity * (1.0 + (entry_price - price) / max(entry_price, 1e-9))
                fee_exit = gross * fee_rate
                slippage_exit = gross * slippage_rate
                equity = gross - fee_exit - slippage_exit
                position = 0
                entry_price = 0.0

            if target != 0:
                fee_entry = equity * fee_rate
                slippage_entry = equity * slippage_rate
                equity -= (fee_entry + slippage_entry)
                entry_equity = equity
                entry_price = price
                position = target

        if position == 0:
            curve.append(equity)
        elif position > 0:
            curve.append(equity * (price / max(entry_price, 1e-9)))
        else:
            curve.append(equity * (1.0 + (entry_price - price) / max(entry_price, 1e-9)))

    return curve


def _max_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _simulate_selector(
    prices: np.ndarray,
    signals: dict[str, np.ndarray],
    *,
    steps: int,
    window: int,
    fee_rate: float,
    slippage_rate: float,
    initial_cash: float,
) -> SelectorResult:
    curves = {
        name: _simulate_curve(prices, target, steps=steps, fee_rate=fee_rate,
                              slippage_rate=slippage_rate, initial_cash=initial_cash)
        for name, target in signals.items()
    }

    equity = float(initial_cash)
    position = 0
    entry_price = 0.0
    entry_equity = equity

    trade_count = 0
    win_count = 0
    win_sum = 0.0
    loss_sum = 0.0
    fee_paid_total = 0.0
    slippage_paid_total = 0.0
    equity_curve: list[float] = []

    for step in range(steps):
        start = max(0, step - window)
        best_name = None
        best_return = -1e9
        for name, curve in curves.items():
            base = curve[start]
            ret = (curve[step] / base) - 1.0 if base > 0 else -1e9
            if ret > best_return:
                best_return = ret
                best_name = name
        target = int(signals[best_name][step])
        target = 1 if target > 0 else -1 if target < 0 else 0
        price = float(prices[step])

        if target != position:
            if position != 0:
                if position > 0:
                    gross = equity * (price / max(entry_price, 1e-9))
                else:
                    gross = equity * (1.0 + (entry_price - price) / max(entry_price, 1e-9))
                fee_exit = gross * fee_rate
                slippage_exit = gross * slippage_rate
                equity = gross - fee_exit - slippage_exit
                fee_paid_total += fee_exit
                slippage_paid_total += slippage_exit
                pnl = equity - entry_equity
                trade_count += 1
                if pnl > 0:
                    win_count += 1
                    win_sum += pnl
                else:
                    loss_sum += pnl
                position = 0
                entry_price = 0.0

            if target != 0:
                fee_entry = equity * fee_rate
                slippage_entry = equity * slippage_rate
                equity -= (fee_entry + slippage_entry)
                fee_paid_total += fee_entry
                slippage_paid_total += slippage_entry
                entry_equity = equity
                entry_price = price
                position = target

        if position == 0:
            equity_curve.append(equity)
        elif position > 0:
            equity_curve.append(equity * (price / max(entry_price, 1e-9)))
        else:
            equity_curve.append(equity * (1.0 + (entry_price - price) / max(entry_price, 1e-9)))

    if position != 0:
        price = float(prices[steps - 1])
        if position > 0:
            gross = equity * (price / max(entry_price, 1e-9))
        else:
            gross = equity * (1.0 + (entry_price - price) / max(entry_price, 1e-9))
        fee_exit = gross * fee_rate
        slippage_exit = gross * slippage_rate
        equity = gross - fee_exit - slippage_exit
        fee_paid_total += fee_exit
        slippage_paid_total += slippage_exit
        pnl = equity - entry_equity
        trade_count += 1
        if pnl > 0:
            win_count += 1
            win_sum += pnl
        else:
            loss_sum += pnl
        equity_curve.append(equity)

    avg_win = win_sum / max(win_count, 1)
    loss_count = trade_count - win_count
    avg_loss = loss_sum / max(loss_count, 1)
    expectancy = (win_sum + loss_sum) / max(trade_count, 1)
    win_rate = win_count / max(trade_count, 1)
    total_return = (equity / initial_cash) - 1.0
    max_dd = _max_drawdown(equity_curve)

    return SelectorResult(
        total_return=total_return,
        trade_count=trade_count,
        win_rate=win_rate,
        avg_win_pnl=avg_win,
        avg_loss_pnl=avg_loss,
        expectancy=expectancy,
        max_drawdown=max_dd,
        total_fee_paid=fee_paid_total,
        total_slippage_paid=slippage_paid_total,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward strategy selector")
    parser.add_argument("--symbol", default=config.DEFAULT_SYMBOL)
    parser.add_argument("--timeframe", default=config.DEFAULT_TIMEFRAME)
    parser.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT)
    parser.add_argument("--history-candles", type=int, default=0)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--window", type=int, default=720)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feed = DataFeed(
        MarketConfig(
            symbol=args.symbol,
            timeframe=args.timeframe,
            limit=args.history_candles or args.limit,
            cache=args.cache,
            cache_only=args.cache_only,
            cache_dir=args.cache_dir,
        )
    )
    frame, _ = feed.fetch(include_indicators=True)
    if len(frame) < 2:
        raise SystemExit("Not enough data for selector.")
    steps = min(args.steps, len(frame) - 1)
    signals = baseline_eval._build_signals(frame)
    prices = frame["close"].to_numpy(dtype=float)
    result = _simulate_selector(
        prices,
        signals,
        steps=steps,
        window=args.window,
        fee_rate=config.FEE_RATE,
        slippage_rate=config.SLIPPAGE_RATE,
        initial_cash=config.INITIAL_CASH,
    )
    print(
        "selector,total_return,trade_count,win_rate,avg_win_pnl,avg_loss_pnl,"
        "expectancy,max_drawdown,total_fee_paid,total_slippage_paid"
    )
    print(
        f"selector,{result.total_return:.6f},{result.trade_count},{result.win_rate:.3f},"
        f"{result.avg_win_pnl:.6f},{result.avg_loss_pnl:.6f},{result.expectancy:.6f},"
        f"{result.max_drawdown:.4f},{result.total_fee_paid:.4f},{result.total_slippage_paid:.4f}"
    )


if __name__ == "__main__":
    main()
