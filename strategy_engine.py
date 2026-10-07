"""
APEXION v1.0 — Strategy Engine
============================
Translates a CoinAnalysis (from Market Intelligence) into a concrete
ExecutionPlan, then validates it through RiskEngine.assess().

Rule: This module does NO market analysis. It only consumes CoinAnalysis
output and produces executable plans.
"""

from __future__ import annotations

import uuid
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

from app.capital_stops import (
    base_net_stop_loss_pct,
    base_net_take_profit_pct,
    tp_sl_by_risk,
)
from app.persisted_intelligence import compact_market_intelligence, compact_trade_plan


# ---------------------------------------------------------------------------
# Strategy-specific parameter dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GridParams:
    """Parameters for a Grid Trading bot on Pionex."""

    grid_count: int = 10
    upper_price: float = 0.0
    lower_price: float = 0.0
    take_profit_pct: float = 2.0
    stop_loss_pct: float = 3.0
    trailing_tp: bool = False
    grid_type: str = "arithmetic"
    rebuild_buffer_pct: float = 1.5


@dataclass
class DCAParams:
    """Parameters for a Dollar-Cost-Averaging bot."""

    dca_amount_per_order: float = 10.0
    dca_interval_hours: float = 4.0
    max_rounds: int = 5
    take_profit_pct: float = 5.0
    stop_loss_pct: float = 5.0



@dataclass
class FlywheelParams:
    """Cyclische dip-buy / recovery-sell strategie zonder leverage."""

    buy_deviation_pct: float = 1.2
    sell_recovery_pct: float = 0.9
    max_cycles: int = 20
    order_size_usdt: float = 10.0
    take_profit_pct: float = 6.0
    stop_loss_pct: float = 7.0


@dataclass
class RebalanceParams:
    """Parameters for a Rebalancing bot."""

    target_allocation_pct: float = 10.0
    rebalance_threshold_pct: float = 2.0
    max_slippage_pct: float = 0.5


@dataclass
class HoldParams:
    """Parameters for a Hold-with-exit strategy."""

    take_profit_pct: float = 10.0
    stop_loss_pct: float = 8.0
    trailing_tp: bool = True


# ---------------------------------------------------------------------------
# ExecutionPlan
# ---------------------------------------------------------------------------


@dataclass
class ExecutionPlan:
    """Concrete executable plan produced by the Strategy Engine.

    This is the single artifact that flows into RiskEngine.assess() and,
    if approved, into the Execution Engine → Pionex.
    """

    plan_id: str = ""
    coin: str = ""
    strategy: str = ""  # grid | dca | rebalance | hold
    size_usdt: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    risk_assessment: Optional[Dict[str, Any]] = None  # populated by RiskEngine
    status: str = "pending"  # pending | approved | rejected | executed | failed
    created_at: str = ""
    executed_at: Optional[str] = None
    result_summary: Optional[str] = None
    paper_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.plan_id:
            self.plan_id = f"plan-{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dict (JSON-safe)."""
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Supported strategies (descriptive metadata for plan building)
#
# NOT runtime capability authority. Paper/live support, execution types and
# safety gates are defined exclusively in app.runtime_capabilities.
# ---------------------------------------------------------------------------

STRATEGIES: Dict[str, Dict[str, Any]] = {
    "grid": {
        "name": "Grid Trading",
        "description": (
            "Plaatst een grid van koop/verkoop-orders binnen een prijsrange. "
            "Geschikt voor zijwaartse markten met voldoende volatiliteit."
        ),
        "pionex_bot_type": "grid",
        "min_volatility": 0.01,
        "max_volatility": 0.15,
        "ideal_trend_range": (-0.02, 0.02),
    },
    "spot_grid": {
        "name": "Spot Grid",
        "description": "Spot Grid zonder leverage voor liquide zijwaartse markten.",
        "pionex_bot_type": "spot_grid",
        "min_volatility": 0.0,
        "max_volatility": 2.0,
        "ideal_trend_range": (0.0, 0.65),
    },
    "dca": {
        "name": "Dollar-Cost Averaging",
        "description": (
            "Spreidt aankopen over tijd om gemiddelde instapprijs te verlagen. "
            "Geschikt voor dalende of onzekere markten."
        ),
        "pionex_bot_type": "dca",
        "min_volatility": 0.005,
        "max_volatility": 0.10,
        "ideal_trend_range": (-0.10, 0.01),
    },

"flywheel": {
    "name": "APEXION Flywheel",
    "description": (
        "Koopt gecontroleerde dips en verkoopt elke herstelbeweging, "
        "waarna dezelfde winstcyclus opnieuw wordt gestart."
    ),
    "pionex_bot_type": "managed_spot",
    "min_volatility": 0.01,
    "max_volatility": 0.20,
    "ideal_trend_range": (-0.08, 0.20),
},
    "rebalance": {
        "name": "Portfolio Rebalancing",
        "description": (
            "Houdt portefeuille-allocatie op koers door periodiek bij te stellen. "
            "Geschikt voor meerdere posities in een range-bound markt."
        ),
        "pionex_bot_type": "rebalance",
        "min_volatility": 0.005,
        "max_volatility": 0.08,
        "ideal_trend_range": (-0.03, 0.03),
    },
    "hold": {
        "name": "Hold met Exit-regels",
        "description": "Spotpositie met take-profit, stop-loss en trailing bescherming.",
        "pionex_bot_type": "manual",
        "min_volatility": 0.005,
        "max_volatility": 2.0,
        "ideal_trend_range": (0.01, 1.0),
    },
    "spot": {
        "name": "Actieve Spot Trade",
        "description": "Gewone spotpositie met dynamische winstbescherming.",
        "pionex_bot_type": "manual_spot",
        "min_volatility": 0.0,
        "max_volatility": 2.0,
        "ideal_trend_range": (0.0, 1.0),
    },
    "momentum": {
        "name": "Momentum Spot",
        "description": "Actieve Spot-trade op sterke recente koersbeweging.",
        "pionex_bot_type": "manual_spot",
        "min_volatility": 0.0,
        "max_volatility": 2.0,
        "ideal_trend_range": (0.15, 1.0),
    },
    "breakout": {
        "name": "Breakout",
        "description": "Volgt een bevestigde opwaartse uitbraak met snelle invalidatie.",
        "pionex_bot_type": "manual_spot",
        "min_volatility": 0.0,
        "max_volatility": 2.0,
        "ideal_trend_range": (0.25, 1.0),
    },
    "trend_follow": {
        "name": "Trend Follow",
        "description": "Blijft in een sterke trend zolang de marktstructuur geldig blijft.",
        "pionex_bot_type": "manual_spot",
        "min_volatility": 0.0,
        "max_volatility": 2.0,
        "ideal_trend_range": (0.35, 1.0),
    },
    "mean_reversion": {
        "name": "Mean Reversion",
        "description": "Handelt een tijdelijke afwijking terug richting het gemiddelde.",
        "pionex_bot_type": "manual_spot",
        "min_volatility": 0.0,
        "max_volatility": 2.0,
        "ideal_trend_range": (0.0, 0.55),
    },
}


# ---------------------------------------------------------------------------
# Parameter computation helpers (NO market analysis — only arithmetic)
# ---------------------------------------------------------------------------


def _tp_sl_by_risk(risk_level: str, base_tp: float, base_sl: float) -> Tuple[float, float]:
    """Scale TP/SL. HIGH risk never widens the capital-protection stop."""
    return tp_sl_by_risk(risk_level, base_tp, base_sl)


def _compute_grid_params(
    price_usdt: float,
    volatility_14d: float,
    trend_strength: float,
    risk_level: str,
) -> GridParams:
    """Bepaal een fee-aware dynamische range, spacing en grid count."""
    from app.paper_costs import get_paper_cost_model, round_trip_break_even_move

    volatility = max(0.01, min(0.40, float(volatility_14d or 0.0)))
    trend = max(-0.20, min(0.20, float(trend_strength or 0.0)))
    half_width_pct = max(0.025, min(0.18, volatility * 0.45))
    center_bias = max(-0.04, min(0.04, trend * 0.25))
    center = price_usdt * (1.0 + center_bias)
    upper_price = round(center * (1.0 + half_width_pct), 8)
    lower_price = round(max(center * (1.0 - half_width_pct), price_usdt * 0.5), 8)

    grid_type = "geometric" if half_width_pct * 2.0 >= 0.10 else "arithmetic"
    model = get_paper_cost_model()
    minimum_move = (
        round_trip_break_even_move(liquidity="maker", cost_model=model)
        + model.min_grid_net_profit_pct / 100.0
    )
    target_move = max(minimum_move * 1.25, min(0.02, max(0.003, volatility * 0.035)))
    if grid_type == "geometric":
        total_move = math.log(upper_price / lower_price)
    else:
        total_move = (upper_price - lower_price) / max(center, 1e-12)
    grid_count = max(5, min(30, int(total_move / target_move)))
    tp, sl = _tp_sl_by_risk(
        risk_level,
        base_net_take_profit_pct("grid"),
        base_net_stop_loss_pct("grid"),
    )
    trailing_tp = abs(trend_strength) < 0.03

    return GridParams(
        grid_count=grid_count,
        upper_price=upper_price,
        lower_price=lower_price,
        take_profit_pct=round(tp, 2),
        stop_loss_pct=round(sl, 2),
        trailing_tp=trailing_tp,
        grid_type=grid_type,
        rebuild_buffer_pct=round(max(0.75, min(3.0, half_width_pct * 20.0)), 2),
    )


def _compute_dca_params(
    size_usdt: float,
    volatility_14d: float,
    trend_strength: float,
    risk_level: str,
) -> DCAParams:
    """Derive DCA parameters."""
    if volatility_14d > 0.08:
        interval_hours = 2.0
    elif volatility_14d > 0.04:
        interval_hours = 4.0
    else:
        interval_hours = 6.0

    if trend_strength < -0.03:
        max_rounds = 8
    elif trend_strength < 0.0:
        max_rounds = 5
    else:
        max_rounds = 3

    amount_per_order = round(size_usdt / max_rounds, 2)
    tp, sl = _tp_sl_by_risk(
        risk_level,
        base_net_take_profit_pct("dca"),
        base_net_stop_loss_pct("dca"),
    )

    return DCAParams(
        dca_amount_per_order=amount_per_order,
        dca_interval_hours=interval_hours,
        max_rounds=max_rounds,
        take_profit_pct=round(tp, 2),
        stop_loss_pct=round(sl, 2),
    )



def _compute_flywheel_params(
    size_usdt: float,
    volatility_14d: float,
    risk_level: str,
) -> FlywheelParams:
    deviation = max(0.6, min(3.0, volatility_14d * 35.0))
    recovery = max(0.45, min(2.0, deviation * 0.72))
    cycles = 12 if risk_level == "low" else 8 if risk_level == "medium" else 5
    order_size = max(1.0, round(size_usdt / 4.0, 2))
    tp, sl = _tp_sl_by_risk(
        risk_level,
        base_net_take_profit_pct("flywheel"),
        base_net_stop_loss_pct("flywheel"),
    )
    return FlywheelParams(
        buy_deviation_pct=round(deviation, 2),
        sell_recovery_pct=round(recovery, 2),
        max_cycles=cycles,
        order_size_usdt=order_size,
        take_profit_pct=round(tp, 2),
        stop_loss_pct=round(sl, 2),
    )


FLYWHEEL_UNFILLED_TTL_SECONDS = 45 * 60
FLYWHEEL_DIP_RANGE_COVER = 0.6


def _flywheel_range_pct(change_24h_pct: float = 0.0, volatility_14d: float = 0.0) -> float:
    try:
        range_pct = abs(float(change_24h_pct or 0.0))
    except (TypeError, ValueError):
        range_pct = 0.0
    if range_pct > 0.0:
        return range_pct
    try:
        return abs(float(volatility_14d or 0.0)) * 100.0
    except (TypeError, ValueError):
        return 0.0


def flywheel_dip_is_realistic(
    buy_deviation_pct: float,
    change_24h_pct: float = 0.0,
    volatility_14d: float = 0.0,
) -> bool:
    """True when the configured dip is covered by recent range/vol."""
    try:
        deviation = float(buy_deviation_pct or 0.0)
    except (TypeError, ValueError):
        return False
    range_pct = _flywheel_range_pct(change_24h_pct, volatility_14d)
    if range_pct <= 0.0 or deviation <= 0.0:
        return False
    return deviation <= FLYWHEEL_DIP_RANGE_COVER * range_pct


def _parse_iso(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def flywheel_is_unfilled(position: Mapping[str, Any] | None) -> bool:
    payload = dict(position or {})
    holding = dict(dict(payload.get("strategy_state") or {}).get("holding") or {})
    try:
        quantity = float(payload.get("quantity") or 0.0)
    except (TypeError, ValueError):
        quantity = 0.0
    try:
        held = float(holding.get("quantity") or 0.0)
    except (TypeError, ValueError):
        held = 0.0
    return quantity <= 0.0 and held <= 0.0


def flywheel_unfilled_age_seconds(
    position: Mapping[str, Any] | None,
    *,
    now: datetime | None = None,
) -> float | None:
    payload = dict(position or {})
    state = dict(payload.get("strategy_state") or {})
    parsed = _parse_iso(state.get("armed_at") or payload.get("opened_at"))
    if parsed is None:
        return None
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return max(0.0, (current - parsed).total_seconds())


def flywheel_unfilled_within_ttl(
    position: Mapping[str, Any] | None,
    *,
    now: datetime | None = None,
) -> bool:
    if not flywheel_is_unfilled(position):
        return False
    age = flywheel_unfilled_age_seconds(position, now=now)
    if age is None:
        return True
    return age < FLYWHEEL_UNFILLED_TTL_SECONDS


def flywheel_unfilled_ttl_expired(
    position: Mapping[str, Any] | None,
    *,
    now: datetime | None = None,
) -> bool:
    if not flywheel_is_unfilled(position):
        return False
    age = flywheel_unfilled_age_seconds(position, now=now)
    if age is None:
        return False
    return age >= FLYWHEEL_UNFILLED_TTL_SECONDS


def _compute_rebalance_params(
    volatility_14d: float,
    composite_score: float,
) -> RebalanceParams:
    """Derive rebalancing parameters."""
    target_pct = round(max(5.0, min(25.0, composite_score * 25)), 1)
    threshold_pct = round(max(1.0, volatility_14d * 30), 1)

    return RebalanceParams(
        target_allocation_pct=target_pct,
        rebalance_threshold_pct=threshold_pct,
        max_slippage_pct=0.5,
    )


def _compute_hold_params(
    trend_strength: float,
    risk_level: str,
) -> HoldParams:
    """Derive hold-with-exit parameters."""
    tp, sl = _tp_sl_by_risk(
        risk_level,
        base_net_take_profit_pct("hold"),
        base_net_stop_loss_pct("hold"),
    )
    trailing_tp = trend_strength > 0.02

    return HoldParams(
        take_profit_pct=round(tp, 2),
        stop_loss_pct=round(sl, 2),
        trailing_tp=trailing_tp,
    )


# ---------------------------------------------------------------------------
# Store helpers
# ---------------------------------------------------------------------------


def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a dataclass instance to a plain dict (shallow)."""
    import dataclasses

    if dataclasses.is_dataclass(obj):
        return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    return dict(obj) if isinstance(obj, dict) else {}


def _compact_plan_metadata(value: Any) -> Dict[str, Any]:
    metadata = dict(value or {})
    metadata["market_intelligence"] = compact_market_intelligence(
        metadata.get("market_intelligence")
    )
    metadata["trade_plan"] = compact_trade_plan(metadata.get("trade_plan"))
    return metadata


def _persist_plan(plan: ExecutionPlan) -> None:
    """Save plan to the shared store so the dashboard can display it."""
    try:
        from app.store import load, save, audit

        state = load()
        plans: List[Dict[str, Any]] = state.get("execution_plans", [])

        plan_dict = {
            "plan_id": plan.plan_id,
            "coin": plan.coin,
            "strategy": plan.strategy,
            "size_usdt": plan.size_usdt,
            "params": plan.params,
            "risk_score": plan.risk_score,
            "risk_assessment": plan.risk_assessment,
            "status": plan.status,
            "created_at": plan.created_at,
            "executed_at": plan.executed_at,
            "result_summary": plan.result_summary,
            "paper_metadata": _compact_plan_metadata(plan.paper_metadata),
        }

        replaced = False
        for i, p in enumerate(plans):
            if p.get("plan_id") == plan.plan_id:
                plans[i] = plan_dict
                replaced = True
                break
        if not replaced:
            plans.append(plan_dict)

        state["execution_plans"] = plans[-500:]
        save(state)
        audit("plan_persisted", f"{plan.plan_id}:{plan.coin}:{plan.strategy}")
    except ImportError:
        pass
    except Exception:
        pass


def _update_plan_in_store(plan: ExecutionPlan) -> None:
    """Update status/result of an existing plan in the store."""
    try:
        from app.store import load, save

        state = load()
        plans: List[Dict[str, Any]] = state.get("execution_plans", [])
        for i, p in enumerate(plans):
            if p.get("plan_id") == plan.plan_id:
                plans[i]["status"] = plan.status
                plans[i]["executed_at"] = plan.executed_at
                plans[i]["result_summary"] = plan.result_summary
                break
        state["execution_plans"] = plans
        save(state)
    except ImportError:
        pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main: build_execution_plan
# ---------------------------------------------------------------------------


def build_execution_plan(
    analysis: Any,  # CoinAnalysis from app.market_intel
    size_usdt: float | None = None,
    total_balance_usdt: float | None = None,
    active_positions: int = 0,
    risk_max_per_coin_usdt: float | None = None,
) -> ExecutionPlan | None:
    """Convert a CoinAnalysis into a validated ExecutionPlan.

    Uses only the real field names from app.market_intel.CoinAnalysis
    and app.market_intel.CoinMetrics:

        analysis.recommended_strategy   (not analysis.strategy)
        analysis.composite_score        (not analysis.score)
        analysis.risk_level
        analysis.strategy_confidence

        metrics.price_usdt              (not metrics.price)
        metrics.volatility_14d          (not metrics.volatility)
        metrics.trend_strength
        metrics.momentum_rsi
        metrics.liquidity_score
        metrics.data_quality

    Args:
        analysis: CoinAnalysis from Market Intelligence.
        size_usdt: Override investment size. If None, derived from config.
        total_balance_usdt: Total paper/live balance for risk context.
        active_positions: Current active position count for risk context.

    Returns:
        ExecutionPlan if strategy is actionable, None if 'none' or invalid.
    """
    # ── Guard: only act on actionable strategies ────────────────────────
    recommended_strategy = getattr(analysis, "recommended_strategy", "none")
    if not recommended_strategy or recommended_strategy == "none":
        return None

    if recommended_strategy not in STRATEGIES:
        return None

    # ── Extract metrics (already computed by Market Intel) ──────────────
    metrics = getattr(analysis, "metrics", None)
    if metrics is None:
        return None
    required_metric_fields = ("price_usdt", "volatility_14d", "trend_strength")
    if any(not hasattr(metrics, name) for name in required_metric_fields):
        return None
    if float(getattr(metrics, "price_usdt", 0.0) or 0.0) <= 0:
        return None

    # Use ONLY real field names from CoinMetrics
    price_usdt = getattr(metrics, "price_usdt", 0.0)
    volatility_14d = getattr(metrics, "volatility_14d", 0.0)
    trend_strength = getattr(metrics, "trend_strength", 0.0)
    # momentum_rsi and liquidity_score available but not needed for all
    # strategies; pulled only where used below.

    # Use ONLY real field names from CoinAnalysis
    composite_score = getattr(analysis, "composite_score", 0.0)
    risk_level = getattr(analysis, "risk_level", "medium")
    coin = getattr(analysis, "coin", "UNKNOWN")

    # ── Determine size ──────────────────────────────────────────────────
    if size_usdt is None:
        from app.config import settings

        size_usdt = settings.default_investment_usdt

    # ── Build strategy-specific params ──────────────────────────────────
    if recommended_strategy in {"grid", "spot_grid"}:
        params_obj = _compute_grid_params(
            price_usdt, volatility_14d, trend_strength, risk_level
        )
    elif recommended_strategy == "dca":
        params_obj = _compute_dca_params(
            size_usdt, volatility_14d, trend_strength, risk_level
        )
    elif recommended_strategy == "flywheel":
        params_obj = _compute_flywheel_params(
            size_usdt, volatility_14d, risk_level
        )
    elif recommended_strategy == "rebalance":
        params_obj = _compute_rebalance_params(volatility_14d, composite_score)
    elif recommended_strategy == "hold":
        params_obj = _compute_hold_params(trend_strength, risk_level)
    elif recommended_strategy in {"spot", "momentum", "breakout", "trend_follow", "mean_reversion"}:
        # Deze strategieën worden in paper mode als actieve spottrade uitgevoerd.
        # De parameters zijn expliciet zodat portfolio intelligence ze later kan beheren.
        tp, sl = _tp_sl_by_risk(
            risk_level,
            base_net_take_profit_pct(recommended_strategy),
            base_net_stop_loss_pct(recommended_strategy),
        )
        params_obj = {
            "entry_type": "market",
            "take_profit_pct": round(tp, 2),
            "stop_loss_pct": round(sl, 2),
            "trailing_tp": recommended_strategy in {"momentum", "breakout", "trend_follow"},
            "reference_price": round(float(price_usdt), 8),
            "strategy_confidence": round(float(getattr(analysis, "strategy_confidence", 0.0)), 4),
        }
    else:
        return None

    # ── Build ExecutionPlan ─────────────────────────────────────────────
    params_payload = _dataclass_to_dict(params_obj)
    intelligence = dict(getattr(analysis, "intelligence", {}) or {})
    intelligence_trade_plan = dict(
        getattr(analysis, "trade_plan", {})
        or intelligence.get("trade_plan", {})
        or {}
    )
    if recommended_strategy in {"grid", "spot_grid"} and intelligence_trade_plan:
        params_payload.update(
            {
                "lower_price": intelligence_trade_plan.get(
                    "grid_lower",
                    params_payload.get("lower_price"),
                ),
                "upper_price": intelligence_trade_plan.get(
                    "grid_upper",
                    params_payload.get("upper_price"),
                ),
                "grid_count": intelligence_trade_plan.get(
                    "grid_count",
                    params_payload.get("grid_count"),
                ),
                "intelligence_grid_spacing_pct": intelligence_trade_plan.get(
                    "grid_spacing_pct"
                ),
            }
        )
    elif intelligence_trade_plan:
        params_payload.update(
            {
                "protective_stop_level": intelligence_trade_plan.get(
                    "protective_stop"
                ),
                "take_profit_levels": intelligence_trade_plan.get(
                    "take_profit_levels", []
                ),
                "trailing_logic": intelligence_trade_plan.get(
                    "trailing_logic", ""
                ),
                "expected_holding_period": intelligence_trade_plan.get(
                    "expected_holding_period", ""
                ),
            }
        )
    plan = ExecutionPlan(
        plan_id=f"plan-{uuid.uuid4().hex[:12]}",
        coin=coin,
        strategy=recommended_strategy,
        size_usdt=size_usdt,
        params=params_payload,
        status="pending",
        paper_metadata={
            "market_intelligence": compact_market_intelligence(intelligence),
            "trade_plan": compact_trade_plan(intelligence_trade_plan),
            "decision_id": str(getattr(analysis, "decision_id", "") or ""),
            "decision_status": str(getattr(analysis, "decision_status", "") or ""),
            "preferred_strategy": str(getattr(analysis, "preferred_strategy", "") or ""),
            "confidence": float(
                getattr(analysis, "strategy_confidence", 0.0) or 0.0
            ),
            "risk_level": str(getattr(analysis, "risk_level", "medium")),
            "reason": str(
                getattr(analysis, "decision_rationale", "")
                or getattr(analysis, "rationale", "")
            ),
        },
    )

    # ── Validate through Risk Engine ────────────────────────────────────
    try:
        from app.risk_engine import RiskLimits, assess

        limits = RiskLimits.from_config(
            total_balance_usdt=total_balance_usdt,
            active_positions=active_positions,
            max_per_coin_usdt=risk_max_per_coin_usdt,
        )
        risk_result = assess(
            coin=coin,
            size_usdt=size_usdt,
            score=composite_score,
            risk_level=risk_level,
            limits=limits,
            plan_id=plan.plan_id,
        )
        risk_payload = risk_result.to_dict()
        plan.risk_score = risk_payload.get("risk_score", 0.0)
        plan.risk_assessment = risk_payload

        if not risk_payload.get("approved", False):
            plan.status = "rejected"
            plan.result_summary = "; ".join(
                risk_payload.get("reasons", ["Risk check failed"])
            )
    except ImportError:
        plan.risk_assessment = {"warning": "risk_engine not importable"}
    except Exception as exc:
        plan.risk_assessment = {"error": str(exc)}
        plan.status = "rejected"
        plan.result_summary = f"Risk assessment error: {exc}"

    # ── Persist to store ────────────────────────────────────────────────
    _persist_plan(plan)

    return plan


# ---------------------------------------------------------------------------
# Strategy metadata
# ---------------------------------------------------------------------------


def get_strategy_info(strategy: str) -> Dict[str, Any] | None:
    """Return metadata about a strategy (name, description, constraints)."""
    return STRATEGIES.get(strategy)


def list_strategies() -> List[Dict[str, Any]]:
    """Return all supported strategies with their metadata."""
    return [{"id": k, **v} for k, v in STRATEGIES.items()]


# ---------------------------------------------------------------------------
# Legacy enrichment
# ---------------------------------------------------------------------------


def enrich_with_legacy(plan: ExecutionPlan) -> ExecutionPlan:
    """Optionally enrich a plan with insights from legacy_v085 strategy modules."""
    try:
        from app.legacy_adapter import enrich_strategy_plan

        enriched = enrich_strategy_plan(
            coin=plan.coin,
            strategy=plan.strategy,
            params=plan.params,
        )
        if enriched:
            plan.params.update(enriched)
            plan.result_summary = (plan.result_summary or "") + " [legacy enriched]"
    except ImportError:
        pass
    except Exception:
        pass
    return plan


# ---------------------------------------------------------------------------
# Execution logging
# ---------------------------------------------------------------------------


def log_execution(plan: ExecutionPlan, result: str, detail: str = "") -> None:
    """Log the result of executing a plan."""
    try:
        from app.store import audit

        plan.status = "executed" if result == "success" else "failed"
        plan.executed_at = datetime.now(timezone.utc).isoformat()
        plan.result_summary = detail or result

        audit("strategy_executed", f"{plan.plan_id}:{plan.coin}:{plan.strategy} → {result}")
        _update_plan_in_store(plan)
    except ImportError:
        pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Plan retrieval
# ---------------------------------------------------------------------------


def get_plans(status: str | None = None) -> List[Dict[str, Any]]:
    """Retrieve execution plans from the store, optionally filtered by status."""
    try:
        from app.store import load

        state = load()
        plans: List[Dict[str, Any]] = state.get("execution_plans", [])
        if status:
            plans = [p for p in plans if p.get("status") == status]
        return plans
    except ImportError:
        return []
    except Exception:
        return []
