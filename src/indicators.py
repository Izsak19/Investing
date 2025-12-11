from __future__ import annotations

import pandas as pd
import pandas_ta as ta


INDICATOR_COLUMNS = [
    "ma", "ema", "wma", "boll_mid", "boll_upper", "boll_lower",
    "vwap", "atr", "trix", "sar", "supertrend"  # supertrend provides upper/lower but we keep direction
]


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy().reset_index(drop=True)
    enriched["ma"] = ta.sma(enriched["close"], length=14)
    enriched["ema"] = ta.ema(enriched["close"], length=14)
    enriched["wma"] = ta.wma(enriched["close"], length=14)
    boll = ta.bbands(enriched["close"], length=20, std=2)
    if boll is not None:
        enriched["boll_mid"] = boll["BBM_20_2.0"]
        enriched["boll_upper"] = boll["BBU_20_2.0"]
        enriched["boll_lower"] = boll["BBL_20_2.0"]
    enriched["vwap"] = ta.vwap(enriched["high"], enriched["low"], enriched["close"], enriched["volume"], length=14)
    enriched["atr"] = ta.atr(enriched["high"], enriched["low"], enriched["close"], length=14)
    enriched["trix"] = ta.trix(enriched["close"], length=18)
    enriched["sar"] = ta.psar(enriched["high"], enriched["low"], enriched["close"], af=0.02, max_af=0.2)["PSARr_0.02_0.2"]
    supertrend = ta.supertrend(enriched["high"], enriched["low"], enriched["close"], length=10, multiplier=3.0)
    if supertrend is not None:
        enriched["supertrend"] = supertrend["SUPERTd_10_3.0"]
    enriched = enriched.dropna().reset_index(drop=True)
    return enriched
