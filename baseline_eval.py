from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data_feed import DataFeed, MarketConfig


@dataclass
class BaselineResult:
    name: str
    total_return: float
    trade_count: int
    win_rate: float
    avg_win_pnl: float
    avg_loss_pnl: float
    expectancy: float
    max_drawdown: float
    total_fee_paid: float
    total_slippage_paid: float


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


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


def _simulate(
    prices: np.ndarray,
    targets: np.ndarray,
    *,
    steps: int,
    cycle: bool,
    fee_rate: float,
    slippage_rate: float,
    initial_cash: float,
) -> BaselineResult:
    episode_len = len(prices) - 1
    if episode_len <= 0:
        raise ValueError("Not enough candles for baseline simulation.")
    if not cycle and steps > episode_len:
        steps = episode_len

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

    cost = float(fee_rate + slippage_rate)

    for step in range(steps):
        i = step % episode_len if cycle else step
        price = float(prices[i])
        target = int(targets[i])
        if target > 0:
            target = 1
        elif target < 0:
            target = -1
        else:
            target = 0

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
        price = float(prices[(steps - 1) % episode_len if cycle else steps - 1])
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
    max_dd = _max_drawdown(equity_curve)
    total_return = (equity / initial_cash) - 1.0

    return BaselineResult(
        name="",
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


def _build_signals(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    close = frame["close"]
    signals: dict[str, pd.Series] = {}

    signals["buy_hold"] = pd.Series(1, index=frame.index)

    sma_fast = close.rolling(20, min_periods=5).mean()
    sma_slow = close.rolling(60, min_periods=10).mean()
    signals["sma_20_60_ls"] = np.where(sma_fast > sma_slow, 1, np.where(sma_fast < sma_slow, -1, 0))

    trend = frame.get("trend_48", pd.Series(0.0, index=frame.index))
    signals["trend_48_ls"] = np.where(trend >= 0.01, 1, np.where(trend <= -0.01, -1, 0))
    signals["trend_48_long"] = np.where(trend >= 0.01, 1, 0)

    basis = frame.get("basis_pct", pd.Series(0.0, index=frame.index))
    signals["basis_meanrev_ls"] = np.where(basis <= -0.01, 1, np.where(basis >= 0.01, -1, 0))

    roll_high = close.rolling(20, min_periods=5).max()
    roll_low = close.rolling(20, min_periods=5).min()
    signals["breakout_20_ls"] = np.where(close >= roll_high, 1, np.where(close <= roll_low, -1, 0))

    mom = close.pct_change(6).fillna(0.0)
    signals["momentum_6_ls"] = np.where(mom >= 0.002, 1, np.where(mom <= -0.002, -1, 0))

    ret = frame.get("ret_1m", pd.Series(0.0, index=frame.index))
    rv = frame.get("rv_1m", pd.Series(0.0, index=frame.index))
    signals["ret_vol_ls"] = np.where(ret >= (1.5 * rv), 1, np.where(ret <= (-1.5 * rv), -1, 0))

    rsi = _compute_rsi(close, period=14)
    signals["rsi_14_meanrev_ls"] = np.where(rsi <= 30, 1, np.where(rsi >= 70, -1, 0))

    vol = frame.get("rv1m_pct_5m", pd.Series(0.0, index=frame.index))
    signals["trend_vol_ls"] = np.where(
        (trend >= 0.01) & (vol >= 0.35), 1, np.where((trend <= -0.01) & (vol >= 0.35), -1, 0)
    )

    return {name: np.asarray(series, dtype=int) for name, series in signals.items()}


def _load_frame(args: argparse.Namespace) -> pd.DataFrame:
    feed = DataFeed(
        MarketConfig(
            symbol=args.symbol,
            timeframe=args.timeframe,
            limit=args.history_candles or args.limit,
            offline=args.offline,
            cache=args.cache,
            cache_only=args.cache_only,
            cache_dir=args.cache_dir,
        )
    )
    frame, _ = feed.fetch(include_indicators=True)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline strategy evaluator")
    parser.add_argument("--symbol", default=config.DEFAULT_SYMBOL)
    parser.add_argument("--timeframe", default=config.DEFAULT_TIMEFRAME)
    parser.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT)
    parser.add_argument("--history-candles", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--cycle-window", action="store_true")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = _load_frame(args)
    if len(frame) < 2:
        raise SystemExit("Not enough data to evaluate baselines.")

    signals = _build_signals(frame)
    prices = frame["close"].to_numpy(dtype=float)

    results: list[BaselineResult] = []
    for name, target in signals.items():
        result = _simulate(
            prices,
            target,
            steps=args.steps,
            cycle=args.cycle_window,
            fee_rate=config.FEE_RATE,
            slippage_rate=config.SLIPPAGE_RATE,
            initial_cash=config.INITIAL_CASH,
        )
        result.name = name
        results.append(result)

    results.sort(key=lambda r: r.total_return, reverse=True)
    header = (
        "strategy,total_return,trade_count,win_rate,avg_win_pnl,avg_loss_pnl,"
        "expectancy,max_drawdown,total_fee_paid,total_slippage_paid"
    )
    print(header)
    for r in results[: max(1, args.top)]:
        print(
            f"{r.name},{r.total_return:.6f},{r.trade_count},{r.win_rate:.3f},"
            f"{r.avg_win_pnl:.6f},{r.avg_loss_pnl:.6f},{r.expectancy:.6f},"
            f"{r.max_drawdown:.4f},{r.total_fee_paid:.4f},{r.total_slippage_paid:.4f}"
        )

    if args.json_out:
        payload = [r.__dict__ for r in results]
        args.json_out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
