from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
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

    def fetch(self) -> tuple[pd.DataFrame, bool]:
        if self._exchange:
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
