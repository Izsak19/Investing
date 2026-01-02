from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

try:
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - environment without ccxt
    ccxt = None

import numpy as np
import pandas as pd

from src.timeframe import timeframe_to_minutes

@dataclass
class MarketConfig:
    symbol: str
    timeframe: str = "1m"
    limit: int = 200
    offline: bool = False
    cache: bool = False
    cache_only: bool = False
    cache_dir: Path | str = Path("data/cache")
    window_end: str | None = None
    derivatives: bool = False
    derivatives_exchange: str | None = "binanceusdm"
    include_orderbook: bool = True


class DataFeed:
    """Pulls OHLCV candles from Binance or generates synthetic data."""

    def __init__(self, config: MarketConfig):
        self.config = config
        self._exchange: Optional[object] = None
        self._deriv_exchange: Optional[object] = None
        self._deriv_symbol: str | None = None
        self._warned_synthetic = False
        self._warned_derivatives = False
        if not config.offline and ccxt is not None:
            try:
                self._exchange = ccxt.binance({"enableRateLimit": True})
            except Exception:
                self._exchange = None
            if config.derivatives:
                try:
                    deriv_name = config.derivatives_exchange or "binanceusdm"
                    exchange_cls = getattr(ccxt, deriv_name)
                    self._deriv_exchange = exchange_cls({"enableRateLimit": True})
                    self._deriv_symbol = self._resolve_deriv_symbol()
                except Exception:
                    self._deriv_exchange = None

        self._cache_dir = Path(config.cache_dir)

    @staticmethod
    def _sanitize_component(value: str) -> str:
        return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)

    def _cache_key(self, window_end: str | None) -> str:
        key_end = self._sanitize_component(window_end or "latest")
        safe_symbol = self._sanitize_component(self.config.symbol.replace("/", "_"))
        safe_timeframe = self._sanitize_component(self.config.timeframe)
        return f"{safe_symbol}__{safe_timeframe}__{self.config.limit}__{key_end}"

    def _cache_paths(self, window_end: str | None) -> tuple[Path, Path]:
        base = self._cache_dir / self._cache_key(window_end)
        return base.with_suffix(".csv"), base.with_suffix(".json")

    def _load_cached(self, window_end: str | None) -> tuple[pd.DataFrame, bool] | None:
        data_path, meta_path = self._cache_paths(window_end)
        if not data_path.exists():
            return None
        # pandas: infer_datetime_format is deprecated; strict parsing is default now.
        frame = pd.read_csv(data_path, parse_dates=["timestamp"])
        is_live = False
        if meta_path.exists():
            try:
                metadata = meta_path.read_text()
                is_live = "live" in metadata.lower()
            except Exception:
                pass
        return frame, is_live

    def _latest_cached_window(self) -> str | None:
        prefix = f"{self.config.symbol.replace('/', '_')}__{self.config.timeframe}__{self.config.limit}__"
        candidates = sorted(self._cache_dir.glob(f"{prefix}*.csv"))
        return candidates[-1].stem.replace(prefix, "") if candidates else None

    def _save_cache(self, frame: pd.DataFrame, window_end: str, is_live: bool) -> None:
        data_path, meta_path = self._cache_paths(window_end)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(data_path, index=False)
        meta_path.write_text("live" if is_live else "synthetic")

    def fetch(self, *, include_indicators: bool = False) -> tuple[pd.DataFrame, bool]:
        target_window = self.config.window_end or self._latest_cached_window()
        if (self.config.cache or self.config.cache_only) and target_window:
            cached = self._load_cached(target_window)
            if cached is not None:
                frame, is_live = cached
                if self.config.derivatives:
                    frame = self._attach_derivatives(frame)
                if include_indicators:
                    from src.indicators import compute_indicators, INDICATOR_COLUMNS

                    missing = [col for col in INDICATOR_COLUMNS if col not in frame.columns]
                    if missing:
                        raw_cols = ["timestamp", "open", "high", "low", "close", "volume"]
                        extra_cols = [col for col in ("funding_rate", "open_interest", "orderbook_imbalance") if col in frame.columns]
                        raw_frame = frame[raw_cols + extra_cols].copy()
                        frame = compute_indicators(raw_frame)
                return frame, is_live
            if self.config.cache_only:
                raise FileNotFoundError(f"Cached dataset not found for window {target_window}")

        raw_frame, is_live = self._fetch_live_or_synthetic()
        if self.config.derivatives:
            raw_frame = self._attach_derivatives(raw_frame)
        frame = raw_frame
        if include_indicators:
            from src.indicators import compute_indicators  # local import to avoid cycles

            frame = compute_indicators(raw_frame)

        last_ts = None
        if not frame.empty and "timestamp" in frame.columns:
            try:
                last_ts = pd.to_datetime(frame.iloc[-1]["timestamp"]).isoformat()
            except Exception:
                last_ts = None
        window_end = self.config.window_end or last_ts or datetime.now(timezone.utc).isoformat()
        if self.config.cache:
            self._save_cache(frame, window_end, is_live)
        return frame, is_live

    def _warn_derivatives(self, message: str) -> None:
        if not self._warned_derivatives:
            print(message)
            self._warned_derivatives = True

    def _attach_derivatives(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self._deriv_exchange is None or frame.empty or "timestamp" not in frame.columns:
            return frame
        enriched = frame.copy()
        ts = pd.to_datetime(enriched["timestamp"], errors="coerce")
        if ts.isna().all():
            return frame
        enriched["timestamp"] = ts
        start_ms = int(ts.min().timestamp() * 1000)
        end_ms = int(ts.max().timestamp() * 1000)

        funding = self._fetch_funding_history(start_ms, end_ms)
        if funding is not None and not funding.empty:
            enriched = pd.merge_asof(
                enriched.sort_values("timestamp"),
                funding.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )
        else:
            if "funding_rate" not in enriched.columns:
                enriched["funding_rate"] = np.nan

        open_interest = self._fetch_open_interest_history(start_ms, end_ms)
        if open_interest is not None and not open_interest.empty:
            enriched = pd.merge_asof(
                enriched.sort_values("timestamp"),
                open_interest.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )
        else:
            if "open_interest" not in enriched.columns:
                enriched["open_interest"] = np.nan

        if self.config.include_orderbook:
            imbalance = self._fetch_orderbook_imbalance()
            if imbalance is not None:
                if "orderbook_imbalance" not in enriched.columns:
                    enriched["orderbook_imbalance"] = np.nan
                enriched.loc[enriched.index[-1], "orderbook_imbalance"] = imbalance
            elif "orderbook_imbalance" not in enriched.columns:
                enriched["orderbook_imbalance"] = np.nan

        return enriched

    def _fetch_funding_history(self, start_ms: int, end_ms: int) -> pd.DataFrame | None:
        exchange = self._deriv_exchange
        if exchange is None or not hasattr(exchange, "fetchFundingRateHistory"):
            return None
        try:
            rows = self._fetch_history_series(
                exchange.fetchFundingRateHistory,
                start_ms,
                end_ms,
                value_key="fundingRate",
                symbol=self._deriv_symbol or self.config.symbol,
            )
        except Exception as exc:
            self._warn_derivatives(f"[data] Funding history unavailable: {exc}")
            rows = []
        if not rows:
            try:
                rows = self._fetch_history_series(
                    exchange.fetchFundingRateHistory,
                    start_ms,
                    end_ms,
                    value_key="fundingRate",
                    symbol=self._deriv_symbol or self.config.symbol,
                    use_since=False,
                )
            except Exception as exc:
                self._warn_derivatives(f"[data] Funding history unavailable: {exc}")
                return None
        if not rows:
            return None
        return pd.DataFrame(rows, columns=["timestamp", "funding_rate"])

    def _fetch_open_interest_history(self, start_ms: int, end_ms: int) -> pd.DataFrame | None:
        exchange = self._deriv_exchange
        if exchange is None or not hasattr(exchange, "fetchOpenInterestHistory"):
            return None
        tf_min = timeframe_to_minutes(self.config.timeframe)
        tf_ms = int(max(tf_min, 1e-9) * 60_000)
        max_lookback_ms = 30 * 24 * 60 * 60 * 1000
        if end_ms - start_ms > max_lookback_ms:
            start_ms = end_ms - max_lookback_ms
        if tf_ms > 0:
            start_ms = start_ms - (start_ms % tf_ms)
        params = {"period": self.config.timeframe}
        try:
            rows = self._fetch_history_series(
                exchange.fetchOpenInterestHistory,
                start_ms,
                end_ms,
                value_key="openInterest",
                symbol=self._deriv_symbol or self.config.symbol,
                params=params,
                timeframe=self.config.timeframe,
            )
        except Exception as exc:
            self._warn_derivatives(f"[data] Open interest history unavailable: {exc}")
            rows = []
        if not rows:
            try:
                rows = self._fetch_history_series(
                    exchange.fetchOpenInterestHistory,
                    start_ms,
                    end_ms,
                    value_key="openInterest",
                    symbol=self._deriv_symbol or self.config.symbol,
                    params=params,
                    timeframe=self.config.timeframe,
                    use_since=False,
                )
            except Exception as exc:
                self._warn_derivatives(f"[data] Open interest history unavailable: {exc}")
                return None
        if not rows:
            return None
        return pd.DataFrame(rows, columns=["timestamp", "open_interest"])

    def _fetch_orderbook_imbalance(self) -> float | None:
        exchange = self._deriv_exchange
        if exchange is None or not hasattr(exchange, "fetchOrderBook"):
            return None
        try:
            book = exchange.fetchOrderBook(self._deriv_symbol or self.config.symbol, limit=20)
            bids = book.get("bids") or []
            asks = book.get("asks") or []
            bid_vol = sum(float(b[1]) for b in bids[:10]) if bids else 0.0
            ask_vol = sum(float(a[1]) for a in asks[:10]) if asks else 0.0
            denom = bid_vol + ask_vol
            if denom <= 0:
                return None
            return (bid_vol - ask_vol) / denom
        except Exception as exc:
            self._warn_derivatives(f"[data] Order book snapshot unavailable: {exc}")
            return None

    def _resolve_deriv_symbol(self) -> str | None:
        exchange = self._deriv_exchange
        if exchange is None:
            return None
        try:
            markets = exchange.load_markets()
        except Exception as exc:
            self._warn_derivatives(f"[data] Derivatives markets unavailable: {exc}")
            return None
        if self.config.symbol in markets:
            return self.config.symbol
        if "/" not in self.config.symbol:
            return None
        base, quote = self.config.symbol.split("/", 1)
        candidates = []
        for market in markets.values():
            if market.get("base") != base or market.get("quote") != quote:
                continue
            candidates.append(market)
        if not candidates:
            return None
        for market in candidates:
            if market.get("swap") and market.get("active", True):
                return market.get("symbol")
        return candidates[0].get("symbol")

    def _fetch_history_series(
        self,
        fetcher,
        start_ms: int,
        end_ms: int,
        *,
        value_key: str,
        symbol: str,
        params: dict | None = None,
        timeframe: str | None = None,
        use_since: bool = True,
    ) -> list[tuple[pd.Timestamp, float]]:
        rows: list[tuple[pd.Timestamp, float]] = []
        since = int(start_ms)
        max_iters = 30
        params = params or {}
        for _ in range(max_iters):
            if use_since:
                if timeframe is None:
                    batch = fetcher(symbol, since=since, limit=1000, params=params)
                else:
                    batch = fetcher(symbol, timeframe, since, 1000, params)
            else:
                if timeframe is None:
                    batch = fetcher(symbol, None, 1000, params)
                else:
                    batch = fetcher(symbol, timeframe, None, 1000, params)
            if not batch:
                break
            last_ts = None
            for row in batch:
                if isinstance(row, dict):
                    ts = row.get("timestamp") or row.get("datetime")
                    value = row.get(value_key)
                    if value is None and value_key == "openInterest":
                        value = row.get("openInterestAmount") or row.get("openInterestValue")
                else:
                    continue
                if ts is None:
                    continue
                ts_dt = pd.to_datetime(ts, unit="ms", errors="coerce")
                if pd.isna(ts_dt):
                    ts_dt = pd.to_datetime(ts, errors="coerce")
                if pd.isna(ts_dt):
                    continue
                value_float = float(value) if value is not None else 0.0
                rows.append((ts_dt, value_float))
                last_ts = int(ts_dt.timestamp() * 1000)
            if last_ts is None:
                break
            if last_ts >= end_ms:
                break
            since = last_ts + 1
        return rows

    def _fetch_live_or_synthetic(self) -> tuple[pd.DataFrame, bool]:
        if self._exchange and not self.config.cache_only:
            try:
                target = max(1, int(self.config.limit))
                if target > self._max_ohlcv_limit():
                    ohlcv = self._fetch_live_history(target)
                else:
                    ohlcv = self._exchange.fetch_ohlcv(
                        self.config.symbol, timeframe=self.config.timeframe, limit=target
                    )
                return self._to_dataframe(ohlcv), True
            except Exception:
                pass
        frame = self._generate_synthetic()
        if not self._warned_synthetic:
            print("[data] Using synthetic candles (live feed unavailable or offline mode).")
            self._warned_synthetic = True
        return frame, False

    def _to_dataframe(self, ohlcv: list[list[float]]) -> pd.DataFrame:
        frame = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms")
        return frame

    def _max_ohlcv_limit(self) -> int:
        return 1000

    def _fetch_live_history(self, target_limit: int) -> list[list[float]]:
        max_limit = self._max_ohlcv_limit()
        tf_min = timeframe_to_minutes(self.config.timeframe)
        tf_ms = int(max(tf_min, 1e-9) * 60_000)
        now_ms = int(self._exchange.milliseconds())
        since = now_ms - (target_limit + 5) * tf_ms
        all_rows: list[list[float]] = []
        last_ts: int | None = None
        max_iters = max(1, int(math.ceil(target_limit / max_limit)) + 3)
        for _ in range(max_iters):
            remaining = target_limit - len(all_rows)
            if remaining <= 0:
                break
            batch = self._exchange.fetch_ohlcv(
                self.config.symbol,
                timeframe=self.config.timeframe,
                limit=min(max_limit, remaining),
                since=since,
            )
            if not batch:
                break
            if last_ts is not None:
                batch = [row for row in batch if row[0] > last_ts]
            if not batch:
                break
            all_rows.extend(batch)
            last_ts = int(batch[-1][0])
            since = last_ts + tf_ms
            if last_ts >= now_ms - tf_ms:
                break
        if len(all_rows) > target_limit:
            all_rows = all_rows[-target_limit:]
        return all_rows

    def _generate_synthetic(self) -> pd.DataFrame:
        now = datetime.utcnow()
        timestamps = [now - i * timedelta(minutes=1) for i in range(self.config.limit)][::-1]
        prices = [30000 + random.gauss(0, 40) for _ in timestamps]
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": prices,
            "high": [p + abs(random.gauss(0, 20)) for p in prices],
            "low": [p - abs(random.gauss(0, 20)) for p in prices],
            "close": [p + random.gauss(0, 10) for p in prices],
            "volume": np.abs(np.random.normal(100, 15, size=len(prices))),
        })
        return df
