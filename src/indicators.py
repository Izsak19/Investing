from __future__ import annotations

import numpy as np
import pandas as pd

# Columns produced by compute_indicators
INDICATOR_COLUMNS = [
    "ma",
    "ema",
    "wma",
    "boll_mid",
    "boll_upper",
    "boll_lower",
    "vwap",
    "atr",
    "trix",
    "sar",
    "supertrend",
]


def _weighted_moving_average(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1)
    return ranges.max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.rolling(length).mean()


def _trix(close: pd.Series, length: int) -> pd.Series:
    ema1 = close.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()
    return ema3.pct_change() * 100


def _parabolic_sar(high: pd.Series, low: pd.Series, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    sar = pd.Series(index=high.index, dtype=float)
    up_trend = True
    af = step
    ep = low.iloc[0]
    sar.iloc[0] = low.iloc[0]

    # Note: this looped implementation is simple but slow on very large datasets.
    # Consider vectorization or numba for millions of rows.
    for i in range(1, len(high)):
        prev_sar = sar.iloc[i - 1]
        if up_trend:
            sar.iloc[i] = prev_sar + af * (ep - prev_sar)
            if low.iloc[i] < sar.iloc[i]:
                up_trend = False
                sar.iloc[i] = ep
                ep = low.iloc[i]
                af = step
            elif high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + step, max_step)
        else:
            sar.iloc[i] = prev_sar + af * (ep - prev_sar)
            if high.iloc[i] > sar.iloc[i]:
                up_trend = True
                sar.iloc[i] = ep
                ep = high.iloc[i]
                af = step
            elif low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + step, max_step)
    return sar


def _supertrend(df: pd.DataFrame, length: int, multiplier: float) -> pd.Series:
    atr = _atr(df["high"], df["low"], df["close"], length)
    hl2 = (df["high"] + df["low"]) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = True  # True for uptrend, False for downtrend

    # Pure-Python loop; optimize with vectorization/numba if scaling to very large windows.
    for i in range(len(df)):
        if i == 0:
            supertrend.iloc[i] = upperband.iloc[i]
            continue

        prev_supertrend = supertrend.iloc[i - 1]
        if direction:
            current_upper = min(upperband.iloc[i], prev_supertrend)
            current_lower = lowerband.iloc[i]
            if df["close"].iloc[i] <= current_upper:
                direction = False
                supertrend.iloc[i] = current_lower
            else:
                supertrend.iloc[i] = current_upper
        else:
            current_upper = upperband.iloc[i]
            current_lower = max(lowerband.iloc[i], prev_supertrend)
            if df["close"].iloc[i] >= current_lower:
                direction = True
                supertrend.iloc[i] = current_upper
            else:
                supertrend.iloc[i] = current_lower

    return supertrend


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy().reset_index(drop=True)

    enriched["ma"] = enriched["close"].rolling(14).mean()
    enriched["ema"] = enriched["close"].ewm(span=14, adjust=False).mean()
    enriched["wma"] = _weighted_moving_average(enriched["close"], 14)

    boll_mid = enriched["close"].rolling(20).mean()
    boll_std = enriched["close"].rolling(20).std()
    enriched["boll_mid"] = boll_mid
    enriched["boll_upper"] = boll_mid + 2 * boll_std
    enriched["boll_lower"] = boll_mid - 2 * boll_std

    typical_price = (enriched["high"] + enriched["low"] + enriched["close"]) / 3
    vwap_num = (typical_price * enriched["volume"]).cumsum()
    vwap_den = enriched["volume"].cumsum()
    enriched["vwap"] = vwap_num / vwap_den

    enriched["atr"] = _atr(enriched["high"], enriched["low"], enriched["close"], 14)
    enriched["trix"] = _trix(enriched["close"], 18)
    enriched["sar"] = _parabolic_sar(enriched["high"], enriched["low"], step=0.02, max_step=0.2)
    enriched["supertrend"] = _supertrend(enriched, length=10, multiplier=3.0)

    enriched = enriched.dropna().reset_index(drop=True)
    return enriched
