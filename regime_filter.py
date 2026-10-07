"""Market-wide regime gate: no NEW positions while BTC is below its N-day average.

Backtests (449 days, 20 liquid coins) showed this one decision mattered far
more than per-coin entry signals: holding BTC only while it closes above its
50-day average turned -28% into +8% with less than half the drawdown.
Open positions are still managed normally; only new entries are blocked.
"""

from __future__ import annotations

import time
from typing import Callable

from app.config import settings

_CACHE_SECONDS = 1800
_cache: dict[str, object] = {"at": 0.0, "days": 0, "result": None}


def _default_fetch(days: int) -> list[float]:
    from app.pionex_client import fetch_klines_for

    rows = fetch_klines_for("BTC", "1D", min(500, days + 5))
    return [float(row["close"]) for row in rows if row.get("close")]


def regime_allows_entries(
    fetch: Callable[[int], list[float]] | None = None,
    now: float | None = None,
) -> tuple[bool, str]:
    """Return (entries_allowed, human_readable_detail). Off when days == 0."""
    days = int(getattr(settings, "regime_filter_sma_days", 0) or 0)
    if days <= 0:
        return True, "regime-filter uit"

    clock = time.time() if now is None else now
    cached = _cache["result"]
    if (
        fetch is None
        and cached is not None
        and _cache["days"] == days
        and clock - float(_cache["at"]) < _CACHE_SECONDS
    ):
        return cached  # type: ignore[return-value]

    try:
        closes = (fetch or _default_fetch)(days)
    except Exception as exc:  # fail closed: unknown market state blocks entries
        return False, f"regime-filter: BTC-data niet beschikbaar ({type(exc).__name__}); geen nieuwe posities"

    if len(closes) < days + 1:
        return False, f"regime-filter: te weinig BTC-dagdata ({len(closes)}/{days + 1}); geen nieuwe posities"

    # Use completed days only: the last candle is the still-open current day.
    last = closes[-2]
    average = sum(closes[-1 - days : -1]) / days
    allowed = last > average
    distance = (last / average - 1.0) * 100.0
    detail = (
        f"regime-filter: BTC {last:,.0f} {'boven' if allowed else 'onder'} "
        f"{days}-daags gemiddelde {average:,.0f} ({distance:+.1f}%)"
        + ("" if allowed else "; geen nieuwe posities")
    )
    result = (allowed, detail)
    if fetch is None:
        _cache.update(at=clock, days=days, result=result)
    return result
