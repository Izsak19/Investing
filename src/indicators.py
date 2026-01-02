from __future__ import annotations

import numpy as np
import pandas as pd

# Columns produced by compute_indicators
INDICATOR_COLUMNS = [
    # Core microstructure
    "ret_1m",
    "rv_1m",
    "ofi_l1",
    "imb1",
    "micro_bias",
    "rel_spread",
    "aggr_imb",
    "dw_spread",
    # Flow & positioning
    "cvd_1m",
    "whale_net_rate_1m",
    "liq_net_rate_1m",
    "oi_delta_1m",
    "basis_pct",
    "funding_rate",
    "open_interest",
    "orderbook_imbalance",
    "trend_48",
    "trend_240",
    "trend_1000",
    "oi_delta_30",
    "oi_delta_120",
    "funding_8h",
    "funding_24h",
    "obi_proxy",
    "htf_trend_15m",
    "htf_trend_1h",
    "htf_rsi_15m",
    "htf_rsi_1h",
    "htf_ema_15m",
    "htf_ema_1h",
    "funding_x_time",
    # Market regime
    "rv1m_pct_5m",
    "spread_pct_5m",
    "tod_sin",
    "tod_cos",
    "atr_14",
]


def _safe_divide(numer: pd.Series, denom: pd.Series | float | int, *, fallback: float = 0.0) -> pd.Series:
    """Divide two series while gracefully handling zeros and scalars.

    When ``denom`` is a scalar, broadcast it to match the shape of ``numer`` so
    ``replace`` can be called safely. Any infinities or NaNs produced by the
    division are replaced with the provided ``fallback`` value.
    """

    numer_series = numer if isinstance(numer, pd.Series) else pd.Series(numer)
    if isinstance(denom, pd.Series):
        denom_series = denom
    else:
        denom_series = pd.Series(denom, index=numer_series.index)

    ratio = numer_series / denom_series.replace(0, np.nan)
    return ratio.replace([np.inf, -np.inf], np.nan).fillna(fallback)


def _rolling_percentile(feature: pd.Series, window: int) -> pd.Series:
    rolling_min = feature.rolling(window, min_periods=2).min()
    rolling_max = feature.rolling(window, min_periods=2).max()
    percentile = _safe_divide(feature - rolling_min, (rolling_max - rolling_min).abs(), fallback=0.0)
    return percentile.clip(0.0, 1.0)

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff().fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)

def _htf_series(enriched: pd.DataFrame, rule: str) -> pd.Series | None:
    if "timestamp" not in enriched.columns:
        return None
    ts = pd.to_datetime(enriched["timestamp"], errors="coerce")
    if ts.isna().all():
        return None
    base = enriched.copy()
    base["timestamp"] = ts
    base = base.set_index("timestamp")
    close = base["close"].resample(rule).last().dropna()
    if close.empty:
        return None
    aligned = close.reindex(base.index, method="ffill")
    aligned = aligned.reset_index(drop=True)
    return aligned


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy().reset_index(drop=True)

    close = enriched["close"]
    open_ = enriched["open"]
    high = enriched["high"]
    low = enriched["low"]
    volume = enriched["volume"]

    if "funding_rate" not in enriched.columns:
        enriched["funding_rate"] = 0.0
    else:
        enriched["funding_rate"] = pd.to_numeric(enriched["funding_rate"], errors="coerce").fillna(0.0)
    if "open_interest" not in enriched.columns:
        enriched["open_interest"] = 0.0
    else:
        enriched["open_interest"] = pd.to_numeric(enriched["open_interest"], errors="coerce").fillna(0.0)
    if "orderbook_imbalance" not in enriched.columns:
        enriched["orderbook_imbalance"] = 0.0
    else:
        enriched["orderbook_imbalance"] = pd.to_numeric(enriched["orderbook_imbalance"], errors="coerce").fillna(0.0)

    spread = (high - low).replace(0, np.nan)
    signed_move = close - open_
    ret_1m = close.pct_change().fillna(0.0)
    volume_roll = volume.rolling(30, min_periods=5).mean().replace(0, np.nan)
    volume_std = volume.rolling(60, min_periods=10).std().replace(0, np.nan)

    enriched["ret_1m"] = ret_1m
    enriched["rv_1m"] = ret_1m.rolling(5, min_periods=2).std().fillna(0.0)

    ofi_raw = _safe_divide(signed_move, spread)
    enriched["ofi_l1"] = _safe_divide(ofi_raw * volume, volume_roll.abs(), fallback=0.0).clip(-5, 5)

    price_position = _safe_divide((close - low) - (high - close), spread).clip(-1, 1)
    enriched["imb1"] = price_position
    enriched["micro_bias"] = _safe_divide(((high + low) / 2) - close, spread.abs()).clip(-5, 5)
    enriched["rel_spread"] = _safe_divide(spread, close.abs()).clip(0, 1)

    signed_volume = np.sign(signed_move.replace(0, 0.0)) * volume
    enriched["aggr_imb"] = _safe_divide(signed_volume, volume_roll.abs(), fallback=0.0).clip(-5, 5)
    dw_component = _safe_divide((close - open_).abs(), spread.abs()).clip(0.0, 2.0)
    enriched["dw_spread"] = (enriched["rel_spread"] * (1.0 + dw_component)).clip(0, 2)

    enriched["cvd_1m"] = _safe_divide(signed_volume, volume_roll.abs(), fallback=0.0).cumsum().clip(-50, 50)

    vol_z = _safe_divide(volume - volume_roll, volume_std, fallback=0.0)
    enriched["whale_net_rate_1m"] = (vol_z * np.sign(ret_1m)).clip(-5, 5)
    enriched["liq_net_rate_1m"] = (_safe_divide(np.minimum(ret_1m, 0.0) * volume, volume_roll, fallback=0.0)).clip(-5, 0)
    vol_short = volume.rolling(10, min_periods=3).mean().replace(0, np.nan)
    oi_series = enriched["open_interest"]
    if oi_series.abs().sum() > 0:
        enriched["oi_delta_1m"] = oi_series.pct_change().fillna(0.0).clip(-5, 5)
    else:
        enriched["oi_delta_1m"] = _safe_divide(vol_short - volume_roll, volume_roll.abs()).clip(-5, 5)

    basis_ref = close.rolling(30, min_periods=5).mean()
    enriched["basis_pct"] = _safe_divide(close - basis_ref, close.abs()).clip(-1, 1)
    enriched["trend_48"] = close.pct_change(48).fillna(0.0)
    enriched["trend_240"] = close.pct_change(240).fillna(0.0)
    enriched["trend_1000"] = close.pct_change(1000).fillna(0.0)
    enriched["oi_delta_30"] = enriched["oi_delta_1m"].rolling(30, min_periods=5).mean().fillna(0.0)
    enriched["oi_delta_120"] = enriched["oi_delta_1m"].rolling(120, min_periods=10).mean().fillna(0.0)
    funding_source = enriched["funding_rate"] if enriched["funding_rate"].abs().sum() > 0 else enriched["basis_pct"]
    enriched["funding_8h"] = funding_source.rolling(96, min_periods=12).mean().fillna(0.0)
    enriched["funding_24h"] = funding_source.rolling(288, min_periods=24).mean().fillna(0.0)
    enriched["obi_proxy"] = (
        enriched["aggr_imb"]
        + enriched["ofi_l1"]
        + enriched["imb1"]
        - (2.0 * enriched["rel_spread"])
        + (enriched["orderbook_imbalance"] * 5.0)
    ).clip(-5, 5)
    time_progress = np.linspace(0.0, 1.0, len(enriched)) if len(enriched) > 1 else np.array([0.0])
    enriched["funding_x_time"] = (funding_source * time_progress).astype(float).clip(-1, 1)

    enriched["rv1m_pct_5m"] = _rolling_percentile(enriched["rv_1m"], 5)
    enriched["spread_pct_5m"] = _rolling_percentile(enriched["rel_spread"], 5)

    prev_close = close.shift(1)
    tr_components = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    tr = tr_components.max(axis=1).fillna(0.0)
    enriched["atr_14"] = tr.rolling(14, min_periods=2).mean().fillna(0.0)

    htf_15m = _htf_series(enriched, "15min")
    htf_1h = _htf_series(enriched, "1h")
    if htf_15m is not None:
        enriched["htf_trend_15m"] = htf_15m.pct_change(1).fillna(0.0)
        enriched["htf_rsi_15m"] = _rsi(htf_15m, period=14)
        ema_15m = htf_15m.ewm(span=14, adjust=False).mean()
        enriched["htf_ema_15m"] = _safe_divide(htf_15m - ema_15m, ema_15m.abs(), fallback=0.0).clip(-1, 1)
    else:
        enriched["htf_trend_15m"] = 0.0
        enriched["htf_rsi_15m"] = 50.0
        enriched["htf_ema_15m"] = 0.0
    if htf_1h is not None:
        enriched["htf_trend_1h"] = htf_1h.pct_change(1).fillna(0.0)
        enriched["htf_rsi_1h"] = _rsi(htf_1h, period=14)
        ema_1h = htf_1h.ewm(span=14, adjust=False).mean()
        enriched["htf_ema_1h"] = _safe_divide(htf_1h - ema_1h, ema_1h.abs(), fallback=0.0).clip(-1, 1)
    else:
        enriched["htf_trend_1h"] = 0.0
        enriched["htf_rsi_1h"] = 50.0
        enriched["htf_ema_1h"] = 0.0

    if "timestamp" in enriched.columns:
        ts = pd.to_datetime(enriched["timestamp"], errors="coerce")
        seconds = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
        angle = 2 * np.pi * _safe_divide(seconds, 24 * 3600, fallback=0.0)
    else:
        angle = 2 * np.pi * _safe_divide(pd.Series(range(len(enriched))), max(len(enriched), 1), fallback=0.0)
    enriched["tod_sin"] = np.sin(angle)
    enriched["tod_cos"] = np.cos(angle)

    enriched = enriched.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return enriched
