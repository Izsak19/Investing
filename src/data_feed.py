from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

try:
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - environment without ccxt
    ccxt = None

import numpy as np
import pandas as pd


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


class DataFeed:
    """Pulls OHLCV candles from Binance or generates synthetic data."""

    def __init__(self, config: MarketConfig):
        self.config = config
        self._exchange: Optional[object] = None
        self._warned_synthetic = False
        if not config.offline and ccxt is not None:
            try:
                self._exchange = ccxt.binance({"enableRateLimit": True})
            except Exception:
                self._exchange = None

        self._cache_dir = Path(config.cache_dir)

    def _cache_key(self, window_end: str | None) -> str:
        key_end = window_end or "latest"
        safe_symbol = self.config.symbol.replace("/", "_")
        return f"{safe_symbol}__{self.config.timeframe}__{self.config.limit}__{key_end}"

    def _cache_paths(self, window_end: str | None) -> tuple[Path, Path]:
        base = self._cache_dir / self._cache_key(window_end)
        return base.with_suffix(".csv"), base.with_suffix(".json")

    def _load_cached(self, window_end: str | None) -> tuple[pd.DataFrame, bool] | None:
        data_path, meta_path = self._cache_paths(window_end)
        if not data_path.exists():
            return None
        frame = pd.read_csv(data_path, parse_dates=["timestamp"], infer_datetime_format=True)
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
                return cached
            if self.config.cache_only:
                raise FileNotFoundError(f"Cached dataset not found for window {target_window}")

        raw_frame, is_live = self._fetch_live_or_synthetic()
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
        window_end = self.config.window_end or last_ts or datetime.utcnow().isoformat()
        if self.config.cache:
            self._save_cache(frame, window_end, is_live)
        return frame, is_live

    def _fetch_live_or_synthetic(self) -> tuple[pd.DataFrame, bool]:
        if self._exchange and not self.config.cache_only:
            try:
                ohlcv = self._exchange.fetch_ohlcv(
                    self.config.symbol, timeframe=self.config.timeframe, limit=self.config.limit
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
