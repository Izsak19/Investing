from __future__ import annotations

MINUTES_PER_YEAR = 365 * 24 * 60


def timeframe_to_minutes(timeframe: str) -> float:
    """Convert timeframe strings like "1m", "5m", or "1h" to minutes."""

    if not timeframe:
        return 1.0

    unit = timeframe[-1].lower()
    try:
        quantity = float(timeframe[:-1])
    except ValueError:
        return 1.0

    multiplier = {"m": 1, "h": 60, "d": 24 * 60}.get(unit)
    return quantity * multiplier if multiplier else 1.0


def periods_per_year(timeframe: str) -> float:
    minutes = timeframe_to_minutes(timeframe)
    if minutes <= 0:
        return MINUTES_PER_YEAR
    return MINUTES_PER_YEAR / minutes
