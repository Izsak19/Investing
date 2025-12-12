from __future__ import annotations

import math
from typing import Iterable, Sequence

import math
import numpy as np


def compute_max_drawdown(equity_curve: Sequence[float]) -> float:
    peak = -float("inf")
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak <= 0:
            continue
        drawdown = (peak - value) / peak
        max_dd = max(max_dd, drawdown)
    return max_dd


def compute_sharpe_ratio(returns: Sequence[float], *, periods_per_year: float | None = None) -> float:
    if not returns:
        return 0.0
    arr = np.asarray(returns, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std <= 0:
        return 0.0
    scale = math.sqrt(periods_per_year) if periods_per_year and periods_per_year > 0 else 1.0
    return (mean / std) * scale


def total_return(initial_value: float, final_value: float) -> float:
    if initial_value <= 0:
        return 0.0
    return (final_value / initial_value) - 1.0


def rolling_volatility(returns: Iterable[float]) -> float:
    arr = np.fromiter(returns, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.std(arr))
