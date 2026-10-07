"""Trend-hold sleeve (PAPER only): hold BTC+ETH while the BTC trend is up, else cash.

Backtests (8 years of daily data, 0.16% round-trip cost) showed that short-term
entry rules have no edge after costs, while a slow BTC trend switch mostly
*reduces drawdowns* (2022: -22% instead of -65%) and needs only a few trades a
year. This sleeve runs that switch on its own virtual capital, separate from the
slot-based portfolio, so the two never compete for slots and an experiment on
one cannot disturb the other.

Rules (completed daily candles only):
  enter when BTC close > SMA(days) * (1 + band)
  exit  when BTC close < SMA(days) * (1 - band)   (band = hysteresis, fewer whipsaws)

This module never talks to a broker: it only simulates fills with the shared
paper cost model and appends closed trades to data/closed_trades.jsonl, so they
show up in the Resultaten panel under the strategy name ``trend_hold``.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app import performance_log
from app.config import settings
from app.paper_costs import get_paper_cost_model, simulate_paper_fill

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATE_PATH = DATA_DIR / "trend_hold.json"
STRATEGY_NAME = "trend_hold"
_MIN_EVAL_SECONDS = 600
_last_eval_at = 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cfg() -> dict[str, Any]:
    coins = [
        part.strip().upper()
        for part in str(getattr(settings, "trend_hold_coins", "BTC,ETH") or "").split(",")
        if part.strip()
    ]
    return {
        "enabled": bool(getattr(settings, "trend_hold_enabled", False)),
        "days": int(getattr(settings, "trend_hold_sma_days", 100) or 100),
        "band": float(getattr(settings, "trend_hold_band_pct", 2.0) or 0.0) / 100.0,
        "coins": coins or ["BTC"],
        "capital": float(getattr(settings, "trend_hold_capital_usdt", 50.0) or 50.0),
    }


def decide(closes: list[float], days: int, band: float, currently_on: bool) -> tuple[bool, str]:
    """Return (hold_long, detail). Fails closed (cash) when data is insufficient."""
    if len(closes) < days + 2:
        return False, f"te weinig BTC-dagdata ({len(closes)}/{days + 2}); blijf in cash"
    last = closes[-2]  # last COMPLETED day; closes[-1] is the still-open day
    average = sum(closes[-1 - days : -1]) / days
    upper, lower = average * (1 + band), average * (1 - band)
    hold = last > lower if currently_on else last > upper
    distance = (last / average - 1.0) * 100.0
    detail = (
        f"BTC {last:,.0f} {distance:+.1f}% t.o.v. {days}-daags gemiddelde {average:,.0f} "
        f"(band ±{band * 100:.1f}%): {'long houden' if hold else 'cash'}"
    )
    return hold, detail


def _load(path: Path, capital: float) -> dict[str, Any]:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "created_at": _now_iso(),
        "start_capital": capital,
        "cash": capital,
        "on": False,
        "positions": [],
        "closed_count": 0,
        "realized_pnl": 0.0,
        "last_eval": None,
        "last_detail": "nog niet geëvalueerd",
    }


def _save(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _default_closes(days: int) -> list[float]:
    from app.pionex_client import fetch_klines_for

    rows = fetch_klines_for("BTC", "1D", min(500, days + 5))
    return [float(row["close"]) for row in rows if row.get("close")]


def _default_price(coin: str) -> float:
    from app.pionex_client import fetch_ticker_for

    return float(fetch_ticker_for(coin)["last"])


def _open(state: dict[str, Any], cfg: dict[str, Any], price_for: Callable[[str], float]) -> int:
    budget = float(state["cash"]) / len(cfg["coins"])
    model = get_paper_cost_model()
    opened = 0
    for coin in cfg["coins"]:
        price = price_for(coin)
        if price <= 0 or budget <= 0:
            continue
        gross_budget = budget / (1.0 + model.fee_rate("taker"))
        quantity = gross_budget / (price * (1.0 + model.slippage_for("taker")))
        fill = simulate_paper_fill(
            side="buy", reference_price=price, quantity=quantity, liquidity="taker", cost_model=model
        )
        state["cash"] -= -fill["net_value"]
        state["positions"].append(
            {
                "position_id": f"th-{uuid.uuid4().hex[:12]}",
                "coin": coin,
                "quantity": fill["quantity"],
                "entry_price": fill["price"],
                "size_usdt": round(-fill["net_value"], 6),
                "fees_paid": fill["fee"],
                "slippage_paid": fill["slippage"],
                "opened_at": _now_iso(),
            }
        )
        opened += 1
    return opened


def _close_all(
    state: dict[str, Any], price_for: Callable[[str], float], reason: str, log_path: Path
) -> int:
    model = get_paper_cost_model()
    closed_rows = []
    for position in state["positions"]:
        price = price_for(position["coin"])
        fill = simulate_paper_fill(
            side="sell",
            reference_price=price,
            quantity=position["quantity"],
            liquidity="taker",
            cost_model=model,
        )
        proceeds = fill["net_value"]
        pnl = proceeds - position["size_usdt"]
        state["cash"] += proceeds
        state["realized_pnl"] = state.get("realized_pnl", 0.0) + pnl
        state["closed_count"] = state.get("closed_count", 0) + 1
        closed_rows.append(
            {
                "position_id": position["position_id"],
                "status": "closed",
                "coin": position["coin"],
                "strategy": STRATEGY_NAME,
                "size_usdt": position["size_usdt"],
                "realized_pnl": pnl,
                "fees_paid": position["fees_paid"] + fill["fee"],
                "slippage_paid": position["slippage_paid"] + fill["slippage"],
                "exit_reason": reason,
                "opened_at": position["opened_at"],
                "closed_at": _now_iso(),
            }
        )
    state["positions"] = []
    performance_log.append_new_trades(closed_rows, path=log_path)
    return len(closed_rows)


def run_cycle(
    *,
    fetch_closes: Callable[[int], list[float]] | None = None,
    price_for: Callable[[str], float] | None = None,
    state_path: Path | None = None,
    log_path: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate the trend switch once. Safe to call every cycle; throttled."""
    global _last_eval_at
    cfg = _cfg()
    if not cfg["enabled"]:
        return {"status": "disabled"}
    clock = time.time()
    if not force and clock - _last_eval_at < _MIN_EVAL_SECONDS:
        return {"status": "throttled"}
    _last_eval_at = clock

    path = state_path or STATE_PATH
    log = log_path or performance_log.LOG_PATH
    state = _load(path, cfg["capital"])
    try:
        closes = (fetch_closes or _default_closes)(cfg["days"])
    except Exception as exc:  # unknown market state: change nothing
        state["last_detail"] = f"BTC-data niet beschikbaar ({type(exc).__name__}); niets gewijzigd"
        state["last_eval"] = _now_iso()
        _save(path, state)
        return {"status": "data_error", "detail": state["last_detail"]}

    price = price_for or _default_price
    hold, detail = decide(closes, cfg["days"], cfg["band"], bool(state.get("on")))
    action = "none"
    try:
        if hold and not state["positions"]:
            if _open(state, cfg, price):
                action = "opened"
        elif not hold and state["positions"]:
            _close_all(state, price, "TREND_OFF", log)
            action = "closed"
    except Exception as exc:  # price fetch failed mid-way: keep state consistent
        state["last_detail"] = f"prijs niet beschikbaar ({type(exc).__name__}); niets gewijzigd"
        state["last_eval"] = _now_iso()
        _save(path, state)
        return {"status": "price_error", "detail": state["last_detail"]}
    state["on"] = bool(state["positions"])
    state["last_eval"] = _now_iso()
    state["last_detail"] = detail
    _save(path, state)
    return {"status": "ok", "action": action, "hold": hold, "detail": detail}


def status(price_for: Callable[[str], float] | None = None, state_path: Path | None = None) -> dict[str, Any]:
    """Marked-to-market view for the dashboard."""
    cfg = _cfg()
    state = _load(state_path or STATE_PATH, cfg["capital"])
    price = price_for or _default_price
    value = float(state["cash"])
    for position in state["positions"]:
        try:
            value += position["quantity"] * price(position["coin"])
        except Exception:
            value += position["size_usdt"]
    start = float(state.get("start_capital") or cfg["capital"])
    return {
        "enabled": cfg["enabled"],
        "holding": bool(state["positions"]),
        "coins": [p["coin"] for p in state["positions"]],
        "start_capital": start,
        "equity": round(value, 4),
        "return_pct": round(100.0 * (value / start - 1.0), 3) if start else 0.0,
        "closed_trades": state.get("closed_count", 0),
        "last_eval": state.get("last_eval"),
        "detail": state.get("last_detail"),
        "paper_only": True,
    }
