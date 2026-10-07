"""Long-only mean-reversion gates shared by scoring and execution.

PAPER cannot short. Overbought RSI is therefore not an entry. Existing TDE
thresholds are reused: oversold RSI <= 34, trend <= 0.42, volatility floor
0.12, and the TDE vol-band decay start 0.70 as the hard ceiling. Bounce-complete
invalidation keeps the production RSI >= 55 + profitable-net rule.
"""

from __future__ import annotations

from typing import Any

OVERSOLD_RSI = 34.0
BOUNCE_COMPLETE_RSI = 55.0
FALLING_KNIFE_RSI = 30.0
FALLING_KNIFE_TREND = -0.05
FALLING_KNIFE_CONTINUATION_TREND = -0.02
MIN_VOLATILITY = 0.12
MAX_VOLATILITY = 0.70
MAX_TREND = 0.42
KNIFE_MOVE_PCT = -4.0
CONTINUATION_MOVE_PCT = -2.0
RANGE_REGIMES = frozenset({"sideways_range", "volatility_compression", "sideways"})
UNSUITABLE_REGIMES = frozenset(
    {
        "volatility_expansion",
        "bear_trend",
        "breakout",
        "volatile_range",
        "risk_off",
    }
)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_falling_knife(
    *,
    rsi: Any,
    trend: Any = 0.0,
    change_24h_pct: Any = 0.0,
) -> bool:
    """True when an oversold long would be buying a continuation dump."""
    rsi_value = _float(rsi, 50.0)
    trend_value = _float(trend)
    change = _float(change_24h_pct)
    if rsi_value <= FALLING_KNIFE_RSI and change <= CONTINUATION_MOVE_PCT:
        return True
    if rsi_value <= OVERSOLD_RSI and change <= KNIFE_MOVE_PCT:
        return True
    if rsi_value <= FALLING_KNIFE_RSI and trend_value <= FALLING_KNIFE_CONTINUATION_TREND:
        return True
    if rsi_value <= OVERSOLD_RSI and trend_value <= FALLING_KNIFE_TREND:
        return True
    return False


def entry_eligible(
    *,
    rsi: Any,
    trend: Any,
    vol: Any = None,
    change_24h_pct: Any = 0.0,
    regime: Any = None,
    require_vol: bool = True,
) -> bool:
    rsi_value = _float(rsi, 50.0)
    trend_value = _float(trend)
    regime_name = str(regime or "").strip().lower()
    if regime_name in UNSUITABLE_REGIMES:
        return False
    if require_vol:
        vol_value = _float(vol)
        if not (MIN_VOLATILITY <= vol_value <= MAX_VOLATILITY):
            return False
    if trend_value > MAX_TREND:
        return False
    if rsi_value > OVERSOLD_RSI:
        return False
    if is_falling_knife(
        rsi=rsi_value,
        trend=trend_value,
        change_24h_pct=change_24h_pct,
    ):
        return False
    return True


def thesis_invalidated(
    *,
    rsi: Any,
    trend: Any,
    net_pnl: Any,
    change_24h_pct: Any = 0.0,
) -> bool:
    """Bounce complete (existing) or thesis failed against a long mean-reversion."""
    rsi_value = _float(rsi, 50.0)
    trend_value = _float(trend)
    pnl = _float(net_pnl)
    bounce_complete = rsi_value >= BOUNCE_COMPLETE_RSI and pnl > 0.0
    failed = pnl < 0.0 and is_falling_knife(
        rsi=rsi_value,
        trend=trend_value,
        change_24h_pct=change_24h_pct,
    )
    return bool(bounce_complete or failed)
