"""
APEXION Trading AI — Core Engine v1.4
====================================
Centrale orchestrator voor:

Market Scan
    -> Score Engine
    -> Confidence Engine
    -> Proposal Engine
    -> Portfolio Intelligence
    -> Strategy/Risk Plan
    -> Paper Portfolio

Veiligheid:
- paper mode blijft standaard;
- high-risk voorstellen worden geblokkeerd;
- confidence en AI-score moeten voldoende hoog zijn;
- Portfolio Intelligence geeft adviserend hold/open/replace/close;
- vervanging of opening gebeurt alleen na expliciete goedkeuring;
- live DIRECT_SPOT-uitvoering gaat via execution_bridge.execute_live_intent;
- live uitvoering voor grid/DCA/trend_follow blijft geblokkeerd tot een latere release.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from copy import deepcopy
from types import SimpleNamespace
import hashlib
import uuid

from app.ai.confidence_engine import ConfidenceEngine
from app.ai.portfolio_intelligence import PortfolioIntelligence
from app.ai.proposal_engine import ProposalEngine, TradeProposal
from app.ai.score_engine import ScoreEngine
# Re-exported for test patches / legacy callers. Canonical PAPER learning writer is
# portfolio_manager.record_position_experience → record_closed_trade(..., state=).
# Engine/rotation/active_execution paths must not invoke this a second time.
from app.ai.market_memory import record_closed_trade  # noqa: F401
from app.ai.decision_logger import DecisionLogger
from app.ai.research_intelligence import build_shadow_score
from app.ai.research_intelligence_v2 import build_research_v2
from app.ai.decision_replay import DecisionReplayStore
from app.ai.opportunity_ranker import OpportunityRanker
from app.ai.lifetime_manager import LifetimeManager
from app.ai.trade_manager import AITradeManager
from app.ai.proposal_history_manager import ProposalHistoryManager
from app.audit_logger import log_event
from app.config import is_micro_live_execution_mode, is_paper_execution_mode, settings
from app.capital_allocator import (
    CapitalAllocationDecision,
    CapitalAllocationSnapshot,
    allocate_dynamic_capital,
    correlated_exposure_usdt,
    policy_from_settings,
)
from app.legacy_adapter import enrich_strategy_plan, status as legacy_status
from app.market_intel import CoinAnalysis, MarketScan, analysis_by_coin_from_scan, scan_watchlist
from app.portfolio_manager import (
    book_live_exchange_exit,
    close_position,
    capital_ledger,
    get_available_balance,
    get_open_positions,
    is_live_reservation,
    portfolio_summary,
    release_live_reservation,
    update_position,
    simulate_slot_state,
    can_reserve_simulated_slot,
    reserve_simulated_slot,
)
from app.risk_engine import (
    clear_emergency_stop,
    is_emergency_stopped,
    risk_summary,
    trigger_emergency_stop,
)
from app.store import MAX_ARCHIVED_PROPOSALS, MAX_PROPOSALS, load, save
from app.persisted_intelligence import (
    compact_market_intelligence,
    compact_replay_snapshot,
    compact_trade_plan,
)
from app.strategy_engine import ExecutionPlan, build_execution_plan
from app.execution_bridge import (
    ExecutionIntent,
    execute_live_intent_sync,
    execute_paper_intent_sync,
    cancel_live_spot_order_sync,
    place_live_spot_sell_sync,
)
from app.active_execution import (
    get_trade_events,
    persist_scan_intelligence_on_position,
)
from app.paper_execution import get_paper_execution_status
from app.paper_grid import preview_grid
from app.network_guard import (
    action_gate,
    actions_allowed,
    observe_portfolio_scan,
    snapshot as network_guard_snapshot,
)
from app.restart_recovery import (
    ensure_runtime_state,
    heartbeat_runtime_session,
    observe_fresh_recovery_scan,
    start_runtime_session,
    stop_runtime_session,
)
from app.runtime_capabilities import (
    is_live_strategy_supported,
    is_micro_live_direct_spot_strategy,
    strategy_execution_status,
)
from app.runtime_guard import enforce_bundled_runtime
from app.gex_overlay import effective_size, scale_allocation, strategy_policy
from app.gex_provider import refresh_if_due


ENGINE_VERSION = "APEXION AI v0.1.7 ULTIMATE SPOT PAPER"
MAX_STORED_PROPOSALS = MAX_PROPOSALS
MAX_STORED_ROTATION_REPORTS = 100
MAX_DASHBOARD_PROPOSALS = 24

# Compacte productie-set.  Oude grid/DCA/flywheel-proposals mogen niet meer
# door nieuwe scans worden aangemaakt; bestaande posities blijven alleen
# leesbaar/beheerbaar voor veilige afbouw.
PROPOSAL_STRATEGIES = {
    "grid", "spot_grid", "dca", "flywheel",
    "spot", "momentum", "breakout", "trend_follow", "mean_reversion",
}
# Slot-reserving statuses. Terminal/executed proposals must never reserve.
from app.proposal_lifecycle import (
    ACTIVE_RESERVATION_STATUSES as ACTIVE_PROPOSAL_STATUSES,
    archive_proposal_for_closed_position,
    claim_proposal_for_execution,
    finalize_proposal_executed,
    finalize_proposal_failed,
    is_active_reservation_status,
    reconcile_proposal_reservations,
)

_engine_started_at: datetime | None = None


def _observe_scan_and_save(state: dict[str, Any], scan: MarketScan) -> dict[str, Any]:
    """Update network + restart recovery atomically for one portfolio scan."""

    guard = observe_portfolio_scan(state, scan)
    healthy = str(guard.get("mode", "BLOCKED")).upper() != "BLOCKED"
    recovery = observe_fresh_recovery_scan(state, healthy=healthy)
    heartbeat_runtime_session(state)
    if bool(recovery.get("required", False)) or not bool(recovery.get("actions_allowed", True)):
        guard["last_reason"] = str(recovery.get("reason") or guard.get("last_reason"))
        guard["recovery_mode"] = True
    else:
        guard["recovery_mode"] = False
    save(state)
    return guard


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _object_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, Mapping):
        return dict(value)

    if is_dataclass(value):
        return asdict(value)

    if hasattr(value, "to_dict"):
        try:
            result = value.to_dict()
            if isinstance(result, Mapping):
                return dict(result)
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        return dict(vars(value))

    return {}


def _safe_audit(*args: Any, **kwargs: Any) -> Any:
    try:
        return log_event(*args, **kwargs)
    except Exception:
        return None


def _normalise_strategy(value: Any) -> str:
    strategy = str(value or "").strip().lower()

    aliases = {
        "spotgrid": "spot_grid",
        "spot-grid": "spot_grid",
        "grid_bot": "grid",
        "rebalancing": "rebalance",
        "wait": "hold",
    }

    return aliases.get(strategy, strategy)


def _proposal_created_at(
    proposal: Mapping[str, Any],
) -> str:
    return str(
        proposal.get("created_at")
        or proposal.get("generated_at")
        or proposal.get("timestamp")
        or proposal.get("approved_at")
        or proposal.get("executed_at")
        or ""
    )


def _proposal_is_test_record(
    proposal: Mapping[str, Any],
) -> bool:
    rationale = str(
        proposal.get("rationale")
        or proposal.get("reason")
        or ""
    ).strip().lower()

    if rationale not in {
        "test",
        "test rationale",
        "test rationale.",
    } and not bool(proposal.get("test_record")):
        return False

    return (
        _safe_float(proposal.get("ai_score")) <= 0.0
        and _proposal_confidence(proposal) <= 0.0
        and _safe_float(
            proposal.get("confidence_pct")
        ) <= 0.0
    )


def _active_duplicate(
    *,
    coin: str,
    strategy: str,
    ignore_id: str = "",
) -> dict[str, Any] | None:
    normalized_coin = str(coin or "").upper()
    normalized_strategy = _normalise_strategy(strategy)

    state = load()
    for existing in reversed(
        state.get("proposals", [])
    ):
        if existing.get("id") == ignore_id:
            continue

        if (
            str(existing.get("coin", "")).upper()
            != normalized_coin
        ):
            continue

        if (
            _normalise_strategy(
                existing.get("strategy")
            )
            != normalized_strategy
        ):
            continue

        if (
            str(existing.get("status", "")).lower()
            in ACTIVE_PROPOSAL_STATUSES
        ):
            return existing

    return None


def _coin_has_open_position(coin: str) -> bool:
    normalized_coin = str(coin or "").upper()

    try:
        return any(
            str(position.get("coin", "")).upper()
            == normalized_coin
            for position in get_open_positions()
        )
    except Exception:
        return False


def cleanup_old_test_proposals() -> int:
    """
    Archiveer uitsluitend duidelijke oude testvoorstellen.

    Deze functie verwijdert geen echte AI-voorstellen.
    """
    state = load()
    proposals = list(state.get("proposals", []))
    archive = list(
        state.get("proposal_archive", [])
    )

    kept: list[dict[str, Any]] = []
    removed = 0

    for proposal in proposals:
        if not _proposal_is_test_record(proposal):
            kept.append(proposal)
            continue

        archived = dict(proposal)
        archived["archived_at"] = _utc_now()
        archived["archive_reason"] = (
            "oude testdata automatisch opgeschoond"
        )
        archive.append(archived)
        removed += 1

    if removed:
        state["proposals"] = kept[
            -MAX_STORED_PROPOSALS:
        ]
        state["proposal_archive"] = archive[-MAX_ARCHIVED_PROPOSALS:]
        save(state)

    return removed



def cleanup_duplicate_proposals() -> int:
    """
    Ruim dubbele dashboardkaarten op.

    Regels:
    - per proposal-ID blijft alleen de nieuwste versie bestaan;
    - per coin/strategie blijft maar één pending of approved voorstel bestaan;
    - per coin/strategie blijft maar één recente blocked kaart bestaan;
    - oude duplicaten gaan naar proposal_archive.
    """
    state = load()
    proposals = list(state.get("proposals", []) or [])
    archive = list(state.get("proposal_archive", []) or [])

    # Nieuwste records eerst verwerken.
    ordered = sorted(
        proposals,
        key=_proposal_created_at,
        reverse=True,
    )

    kept: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_active: set[tuple[str, str]] = set()
    seen_blocked: set[tuple[str, str]] = set()
    removed = 0

    for proposal in ordered:
        item = dict(proposal)
        proposal_id = str(item.get("id", "")).strip()
        key = _proposal_key(
            item.get("coin", ""),
            item.get("strategy", ""),
        )
        status = str(item.get("status", "")).lower()

        duplicate_reason = ""

        if proposal_id and proposal_id in seen_ids:
            duplicate_reason = "duplicate_proposal_id"
        elif status in ACTIVE_PROPOSAL_STATUSES and key in seen_active:
            duplicate_reason = "duplicate_active_coin_strategy"
        elif status == "blocked" and key in seen_blocked:
            duplicate_reason = "duplicate_blocked_coin_strategy"

        if duplicate_reason:
            item["archived_at"] = _utc_now()
            item["archive_reason"] = duplicate_reason
            archive.append(item)
            removed += 1
            continue

        if proposal_id:
            seen_ids.add(proposal_id)
        if status in ACTIVE_PROPOSAL_STATUSES:
            seen_active.add(key)
        if status == "blocked":
            seen_blocked.add(key)

        kept.append(item)

    kept.sort(key=_proposal_created_at)
    if removed:
        state["proposals"] = kept[-MAX_STORED_PROPOSALS:]
        state["proposal_archive"] = archive[-MAX_ARCHIVED_PROPOSALS:]
        save(state)

    return removed


def _cleanup_stale_autonomous_paper_proposals() -> int:
    """Archive stale slot reservations from a previous autonomous PAPER session.

    PAPER and LIVE_DRY_RUN execute qualified proposals automatically. Therefore an
    active proposal that already existed before the current engine start must never
    reserve a slot forever: it belongs to an interrupted/legacy session and must be
    re-evaluated against fresh market data instead of being executed blindly.

    MICRO_LIVE is intentionally excluded because pending proposals there are real
    manual-approval reservations.
    """
    if not is_paper_execution_mode(str(getattr(settings, "mode", "paper"))):
        return 0
    if not bool(getattr(settings, "auto_execute_paper", True)):
        return 0
    if bool(getattr(settings, "effective_approval_required", True)):
        return 0

    cutoff = _engine_started_at or datetime.now(timezone.utc)
    state = load()
    kept: list[dict[str, Any]] = []
    archive = list(state.get("proposal_archive", []) or [])
    moved = 0

    for raw in state.get("proposals", []) or []:
        proposal = dict(raw)
        status = str(proposal.get("status", "")).lower()
        if status not in ACTIVE_PROPOSAL_STATUSES:
            kept.append(proposal)
            continue

        coin = str(proposal.get("coin", "")).upper()
        if coin and _coin_has_open_position(coin):
            kept.append(proposal)
            continue

        created_text = _proposal_created_at(proposal).strip()
        created_at: datetime | None = None
        if created_text:
            try:
                created_at = datetime.fromisoformat(created_text.replace("Z", "+00:00"))
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
            except ValueError:
                created_at = None

        # Missing/invalid timestamps are treated as legacy state. Any proposal
        # created before this process started must be re-scored, not reserved.
        if created_at is None or created_at < cutoff:
            proposal["archived_at"] = _utc_now()
            proposal["archive_reason"] = "stale_autonomous_paper_reservation_restart"
            proposal["previous_status"] = status
            archive.append(proposal)
            moved += 1
            continue

        kept.append(proposal)

    if moved:
        state["proposals"] = kept[-MAX_STORED_PROPOSALS:]
        state["proposal_archive"] = archive[-MAX_ARCHIVED_PROPOSALS:]
        save(state)

    return moved


def _pending_slot_reservations() -> list[dict[str, Any]]:
    """
    Count only genuinely executable active reservations.

    Terminal statuses (executed/failed/closed/archived) never reserve a slot.
    Proposals already linked to an open position also never reserve an extra slot.
    """
    state = load()
    reservations: list[dict[str, Any]] = []
    seen_coins: set[str] = set()

    for proposal in state.get("proposals", []) or []:
        if not is_active_reservation_status(proposal.get("status")):
            continue

        coin = str(proposal.get("coin", "")).upper()
        if not coin or coin in seen_coins or _coin_has_open_position(coin):
            continue

        # Executed-but-mislabelled or already-linked proposals must not reserve.
        if str(proposal.get("position_id") or "").strip() and _coin_has_open_position(coin):
            continue

        seen_coins.add(coin)
        reservations.append(
            {
                "coin": coin,
                "strategy": _normalise_strategy(proposal.get("strategy", "")),
                "status": "reserved",
                "proposal_id": str(proposal.get("id") or ""),
                "proposal_status": str(proposal.get("status") or ""),
            }
        )

    return reservations


def _candidate_value(candidate: Mapping[str, Any]) -> float:
    score = _safe_float(candidate.get("score"))
    confidence = _safe_float(candidate.get("confidence"), score)
    return score * 0.65 + confidence * 0.35


def _authoritative_trade_snapshot(
    analysis: CoinAnalysis,
) -> tuple[dict[str, Any] | None, str]:
    """Validate the exact final intelligence snapshot used for PAPER execution."""
    report = dict(getattr(analysis, "intelligence", {}) or {})
    gate = dict(report.get("decision_gate") or {})
    competition = dict(report.get("competition") or {})
    if not gate or not competition:
        return None, ""

    strategy = _normalise_strategy(
        getattr(analysis, "recommended_strategy", "")
    )
    preferred = _normalise_strategy(
        gate.get(
            "preferred_strategy",
            getattr(analysis, "preferred_strategy", strategy),
        )
    )
    winner = _normalise_strategy(competition.get("winner", ""))
    evaluations = [
        dict(item)
        for item in list(competition.get("evaluations") or [])
        if isinstance(item, Mapping)
    ]
    winner_row = next(
        (
            item for item in evaluations
            if _normalise_strategy(item.get("strategy")) == winner
        ),
        {},
    )
    quality = dict(report.get("data_quality") or {})
    grid = dict(report.get("grid_suitability") or {})
    market_context = dict(report.get("market_context") or {})
    gex_overlay = dict(report.get("gex_overlay") or {})
    gex_policy = dict(report.get("gex_policy") or {})
    if not gex_policy:
        gex_policy = strategy_policy(
            gex_overlay,
            block_near_wall=bool(
                getattr(settings, "gex_block_near_wall_entry", True)
            ),
            wall_entry_block_pct=float(
                getattr(settings, "gex_wall_entry_block_pct", 0.30)
            ),
            block_gamma_transition=bool(
                getattr(settings, "gex_block_gamma_transition_entry", True)
            ),
        )
    timeframes = dict(report.get("timeframes") or {})
    timeframe_config = dict(timeframes.get("config") or {})
    frames = dict(timeframes.get("frames") or {})
    entry_timeframe = str(timeframe_config.get("entry") or "15m")
    entry_frame = dict(frames.get(entry_timeframe) or {})
    evidence_timestamp = str(gate.get("evidence_timestamp") or "")
    entry_timestamp = str(entry_frame.get("timestamp") or "")
    snapshot_consistent = bool(
        evidence_timestamp
        and entry_timestamp
        and evidence_timestamp == entry_timestamp
    )
    safety_blockers = list(
        dict.fromkeys(
            [
                *[
                    str(item)
                    for item in list(getattr(analysis, "risk_flags", []) or [])
                    if str(item).strip()
                ],
                *[
                    str(item)
                    for item in list(gate.get("blockers") or [])
                    if str(item).strip()
                ],
            ]
        )
    )
    snapshot = {
        "decision_id": str(
            getattr(analysis, "decision_id", "")
            or report.get("decision_id", "")
        ),
        "coin": str(getattr(analysis, "coin", "")).upper(),
        "final_decision": str(
            getattr(analysis, "decision_status", "WAIT")
        ).upper(),
        "winner": winner,
        "preferred_strategy": preferred,
        "candidate_eligible": bool(winner_row.get("eligible")),
        "candidate": {
            **winner_row,
            "strategy": winner,
            "score": _safe_float(winner_row.get("risk_adjusted_score")),
        },
        "calibrated_confidence": _safe_float(
            gate.get("calibrated_confidence")
        ),
        "net_edge_pct": _safe_float(
            gate.get(
                "net_edge_pct",
                winner_row.get("expected_net_edge_pct"),
            )
        ),
        "reward_risk": _safe_float(
            gate.get("reward_risk", winner_row.get("reward_risk"))
        ),
        "data_quality_status": str(quality.get("status") or "UNKNOWN"),
        "data_quality_trade_allowed": bool(quality.get("trade_allowed")),
        "grid_suitable": bool(grid.get("suitable")),
        "grid_suitability_score": _safe_float(grid.get("score")),
        "grid_breakout_risk": max(
            0.0,
            min(1.0, _safe_float(grid.get("breakout_risk"))),
        ),
        "market_risk_proxy": str(
            market_context.get("risk_proxy") or "neutral"
        ).lower(),
        "higher_timeframe_bias": str(
            timeframes.get("higher_timeframe_bias") or "neutral"
        ).lower(),
        "gex_overlay": deepcopy(gex_overlay),
        "gex_policy": deepcopy(gex_policy),
        "evidence_timestamp": evidence_timestamp,
        "entry_timestamp": entry_timestamp,
        "snapshot_consistent": snapshot_consistent,
        "consecutive_confirmations": int(
            gate.get("consecutive_confirmations", 0) or 0
        ),
        "required_confirmations": int(
            gate.get("required_confirmations", 0) or 0
        ),
        "safety_blockers": safety_blockers,
        "reason": str(
            getattr(analysis, "main_reason", "")
            or gate.get("reason", "")
        ),
        "qualification_source": "final_intelligence_trade",
    }
    snapshot_identity = "|".join((
        snapshot["decision_id"],
        snapshot["coin"],
        snapshot["winner"],
        snapshot["evidence_timestamp"],
        str(snapshot["gex_overlay"].get("regime") or ""),
        str(snapshot["gex_overlay"].get("as_of") or ""),
        f"{snapshot['calibrated_confidence']:.12f}",
        f"{snapshot['net_edge_pct']:.12f}",
    ))
    snapshot["snapshot_id"] = hashlib.sha256(
        snapshot_identity.encode("utf-8")
    ).hexdigest()

    # ChatGPT Research Layer v1: shadow-only evidence. It deliberately cannot
    # change TRADE/WATCH/WAIT or execution eligibility in this build.
    snapshot["research_v1"] = build_shadow_score(
        confidence=snapshot["calibrated_confidence"],
        net_edge_pct=snapshot["net_edge_pct"],
        grid_suitability_score=snapshot["grid_suitability_score"],
        liquidity_score=_safe_float(report.get("liquidity_score"), 0.5),
        correlation_risk=_safe_float(report.get("correlation_risk"), 0.0),
        drawdown_risk=_safe_float(report.get("drawdown_risk"), 0.0),
        macro_risk=_safe_float(report.get("macro_risk"), 0.0),
        relative_strength=_safe_float(report.get("relative_strength"), 0.0),
        analog_consensus=None,
    )
    snapshot["research_v2"] = build_research_v2(
        confidence=snapshot["calibrated_confidence"],
        net_edge_pct=snapshot["net_edge_pct"],
        grid_suitability_score=snapshot["grid_suitability_score"],
        liquidity_score=_safe_float(report.get("liquidity_score"), 0.5),
        correlation_risk=_safe_float(report.get("correlation_risk"), 0.0),
        drawdown_risk=_safe_float(report.get("drawdown_risk"), 0.0),
        macro_risk=_safe_float(report.get("macro_risk"), 0.0),
        relative_strength=_safe_float(report.get("relative_strength"), 0.0),
        evidence_timestamp=snapshot["evidence_timestamp"],
        catalyst_timestamp=(
            report.get("next_catalyst_timestamp")
            or report.get("catalyst_timestamp")
            or report.get("next_event_timestamp")
        ),
        fee_pct=_safe_float(report.get("fee_pct"), 0.0),
        slippage_pct=_safe_float(report.get("slippage_pct"), 0.0),
        spread_pct=_safe_float(report.get("spread_pct"), 0.0),
    )
    try:
        DecisionReplayStore.record(snapshot)
    except Exception:
        # Replay telemetry may never break the production PAPER decision path.
        pass

    blockers: list[str] = []
    if snapshot["final_decision"] != "TRADE":
        blockers.append("final intelligence decision is geen TRADE")
    if not strategy or strategy != winner or strategy != preferred:
        blockers.append("executionstrategie wijkt af van de competition-winnaar")
    if not snapshot["candidate_eligible"]:
        blockers.append("competition-winnaar is niet execution-eligible")
    if snapshot["net_edge_pct"] <= 0.0:
        blockers.append("autoritatieve netto edge is niet positief")
    if not snapshot["data_quality_trade_allowed"]:
        blockers.append("data-quality gate blokkeert execution")
    if not snapshot_consistent:
        blockers.append("entry-candle en decision evidence verschillen")
    if strategy in {"grid", "spot_grid"} and not snapshot["grid_suitable"]:
        blockers.append("grid suitability gate blokkeert execution")
    if strategy in {"grid", "spot_grid"}:
        maximum_breakout_risk = _safe_float(
            getattr(settings, "paper_grid_max_entry_breakout_risk", 0.55),
            0.55,
        )
        if snapshot["grid_breakout_risk"] > maximum_breakout_risk:
            blockers.append(
                "grid entry geblokkeerd: breakout-risico boven limiet"
            )
        if (
            bool(
                getattr(
                    settings,
                    "paper_grid_block_risk_off_bearish_entry",
                    True,
                )
            )
            and snapshot["market_risk_proxy"] == "risk_off"
            and snapshot["higher_timeframe_bias"] == "bearish"
        ):
            blockers.append(
                "grid entry geblokkeerd: risk-off markt en bearish hogere timeframe"
            )
    if (
        bool(getattr(settings, "gex_strategy_router_enabled", True))
        and bool(gex_overlay.get("valid", False))
    ):
        allowed_gex_strategies = {
            str(item).strip().lower()
            for item in list(gex_policy.get("allowed_strategies") or [])
        }
        if not bool(gex_policy.get("entry_allowed", False)):
            blockers.append(
                str(gex_policy.get("reason") or "GEX entry policy blokkeert execution")
            )
        elif strategy not in allowed_gex_strategies:
            blockers.append(
                "GEX-route vereist strategie: "
                + ", ".join(sorted(allowed_gex_strategies))
            )
    blockers.extend(safety_blockers)
    return snapshot, "; ".join(dict.fromkeys(blockers))


def _set_execution_state(
    scan: MarketScan,
    analysis: CoinAnalysis,
    *,
    status: str,
    blocker: str = "",
    snapshot: Mapping[str, Any] | None = None,
) -> None:
    analysis.execution_status = str(status or "NOT_ATTEMPTED").upper()
    analysis.execution_blocker = str(blocker or "")
    if snapshot is not None:
        analysis.execution_snapshot = dict(snapshot)
    report = dict(getattr(analysis, "intelligence", {}) or {})
    report["execution"] = {
        "status": analysis.execution_status,
        "blocker": analysis.execution_blocker,
        "snapshot": dict(analysis.execution_snapshot or {}),
        "pionex_write_calls": 0,
    }
    analysis.intelligence = report
    for item in list(getattr(scan, "intelligence", []) or []):
        if str(item.get("coin", "")).upper() == analysis.coin.upper():
            item["execution"] = dict(report["execution"])
    for item in list(dict(getattr(scan, "market_data", {}) or {}).get("all_markets") or []):
        if str(item.get("coin", "")).upper() == analysis.coin.upper():
            item["execution_status"] = analysis.execution_status
            item["execution_blocker"] = analysis.execution_blocker


WAIT_REASON_TEXT = {
    "low_confidence": "Geen kandidaat haalt de vereiste confidence-drempel.",
    "high_risk": "De beschikbare kandidaat heeft een te hoog neerwaarts risico.",
    "insufficient_reward_risk": "De verwachte reward/risk-verhouding is onvoldoende.",
    "insufficient_net_edge": "De verwachte gridwinst is na fees en slippage onvoldoende.",
    "strategy_mismatch": "Geen geldige kandidaat past bij dit slottype en marktregime.",
    "duplicate_exposure": "De kandidaat zou dubbele coin-exposure veroorzaken.",
    "insufficient_virtual_capital": "Onvoldoende vrij virtueel paperkapitaal binnen de limieten.",
    "emergency_stop": "Emergency stop blokkeert alle nieuwe paperacties.",
    "approval_gate": "De paper execution- of risicogate heeft de kandidaat geweigerd.",
    "market_data_error": "Publieke marktdata is niet beschikbaar; er wordt geen paperbeslissing genomen.",
    "network_guard": "Netwerk- of marktdataherstel wordt eerst bevestigd; alle paperacties blijven geblokkeerd.",
    "no_valid_candidate": "De publieke marktscan bevat geen geldige kandidaat voor dit slot.",
    "other": "Geen paperpositie geopend; zie de exacte detailreden.",
}


def _wait_reason(category: str, detail: str = "") -> dict[str, str]:
    resolved = category if category in WAIT_REASON_TEXT else "other"
    return {
        "category": resolved,
        "reason": WAIT_REASON_TEXT[resolved],
        "detail": str(detail or WAIT_REASON_TEXT[resolved]),
    }


def _categorize_wait_detail(detail: str) -> str:
    value = str(detail or "").casefold()
    if "netto grid-edge" in value or "net grid-edge" in value:
        return "insufficient_net_edge"
    if "confidence" in value:
        return "low_confidence"
    if "high risk" in value or "hoog risico" in value:
        return "high_risk"
    if "edge" in value or "reward" in value or "r/r" in value:
        return "insufficient_reward_risk"
    if "al actief" in value or "duplicate" in value or "open positie" in value:
        return "duplicate_exposure"
    if "kapitaal" in value or "balance" in value or "bedrag" in value:
        return "insufficient_virtual_capital"
    if "emergency" in value or "noodstop" in value:
        return "emergency_stop"
    if "network" in value or "netwerk" in value or "recovery" in value or "verbinding" in value:
        return "network_guard"
    if "slot" in value or "strategie" in value or "strategy" in value:
        return "strategy_mismatch"
    if "risk" in value or "approval" in value or "gate" in value or "score" in value:
        return "approval_gate"
    return "other"


def _analysis_group_rejection(
    analysis: CoinAnalysis,
    *,
    long_running: bool,
) -> dict[str, str]:
    risk = str(getattr(analysis, "risk_level", "medium")).lower()
    if risk == "high":
        return _wait_reason("high_risk", f"{analysis.coin}: high risk")

    candidates = [
        item
        for item in list(getattr(analysis, "strategy_candidates", []) or [])
        if (
            _normalise_strategy(item.get("strategy"))
            in {"grid", "spot_grid", "dca", "flywheel"}
        ) == long_running
        and _normalise_strategy(item.get("strategy")) not in {"none", "hold", ""}
    ]
    if not candidates:
        return _wait_reason(
            "strategy_mismatch",
            f"{analysis.coin}: geen passende strategie",
        )

    best = max(candidates, key=_candidate_value)
    confidence = _safe_float(best.get("confidence"))
    if confidence < 0.55:
        return _wait_reason(
            "low_confidence",
            f"{analysis.coin}: confidence {confidence:.2f} lager dan 0.55",
        )
    reward_risk = _safe_float(getattr(analysis, "reward_risk_ratio", 0.0))
    if reward_risk < 1.0:
        return _wait_reason(
            "insufficient_reward_risk",
            f"{analysis.coin}: reward/risk {reward_risk:.2f}",
        )
    return _wait_reason(
        "strategy_mismatch",
        f"{analysis.coin}: strategie-fit/score haalt de bestaande kwalificatie niet",
    )


def _best_candidate_for_group(
    analysis: CoinAnalysis,
    *,
    long_running: bool,
    max_gap_from_winner: float = 0.16,
) -> dict[str, Any] | None:
    """
    Zoek de beste strategie uit de gevraagde groep.

    Een alternatieve strategie mag alleen gekozen worden wanneer zij dicht
    genoeg bij de winnaar ligt. Zo krijgt diversificatie ruimte zonder een
    duidelijk slechtere trade te forceren.
    """
    candidates = list(getattr(analysis, "strategy_candidates", []) or [])
    if not candidates:
        return None

    winner_score = max(
        _safe_float(candidate.get("score"))
        for candidate in candidates
    )

    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        strategy = _normalise_strategy(candidate.get("strategy", ""))
        is_long = strategy in {"grid", "spot_grid", "dca", "flywheel"}

        if is_long != long_running:
            continue

        score = _safe_float(candidate.get("score"))
        confidence = _safe_float(candidate.get("confidence"), score)

        if score < 0.50 or confidence < 0.55:
            continue
        if winner_score - score > max_gap_from_winner:
            continue

        selected.append(dict(candidate))

    if not selected:
        return None

    selected.sort(key=_candidate_value, reverse=True)
    return selected[0]


def _analysis_with_strategy(
    analysis: CoinAnalysis,
    candidate: Mapping[str, Any],
) -> CoinAnalysis:
    """
    Maak een veilige kopie van CoinAnalysis met de gekozen tournamentstrategie.
    """
    adapted = deepcopy(analysis)
    adapted.recommended_strategy = _normalise_strategy(
        candidate.get("strategy", "")
    )
    adapted.strategy_confidence = _safe_float(
        candidate.get("confidence"),
        _safe_float(candidate.get("score")),
    )

    reason = str(candidate.get("reason", "")).strip()
    if reason:
        adapted.rationale = (
            f"Portfolio-diversificatie koos {adapted.recommended_strategy}: "
            f"{reason} {getattr(adapted, 'rationale', '')}"
        ).strip()

    return adapted


def _store_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
    state = load()
    proposals = list(state.get("proposals", []))

    proposal = dict(proposal)
    proposal.setdefault("created_at", _utc_now())
    proposal["strategy"] = _normalise_strategy(
        proposal.get("strategy")
    )

    for index, item in enumerate(proposals):
        if item.get("id") == proposal.get("id"):
            proposals[index] = proposal
            break
    else:
        proposals.append(proposal)

    if str(proposal.get("status") or "").lower() == "blocked":
        reasons = list(
            (proposal.get("metadata") or {}).get("rejection_reasons") or []
        )
        if not str(proposal.get("rejection_reason") or "").strip() and reasons:
            proposal["rejection_reason"] = str(reasons[0])

    state["proposals"] = proposals[-MAX_STORED_PROPOSALS:]
    save(state)
    from app.acceptance_trace import record_proposal_decision

    record_proposal_decision(proposal)
    return proposal


def _store_rotation_report(report: dict[str, Any]) -> dict[str, Any]:
    state = load()
    reports = list(state.get("portfolio_intelligence_reports", []))
    reports.append(report)
    state["portfolio_intelligence_reports"] = reports[
        -MAX_STORED_ROTATION_REPORTS:
    ]
    save(state)
    return report


def _find_proposal(
    proposal_id: str,
    *,
    pending_only: bool = False,
) -> dict[str, Any] | None:
    state = load()

    for proposal in state.get("proposals", []):
        if proposal.get("id") != proposal_id:
            continue

        if pending_only and proposal.get("status") != "pending":
            return None

        return proposal

    return None


def _update_stored_proposal(
    proposal_id: str,
    updates: Mapping[str, Any],
) -> dict[str, Any] | None:
    state = load()

    for proposal in state.get("proposals", []):
        if proposal.get("id") == proposal_id:
            proposal.update(dict(updates))
            save(state)
            from app.acceptance_trace import record_proposal_decision

            record_proposal_decision(proposal)
            return proposal

    return None


def _proposal_size(proposal: Mapping[str, Any]) -> float:
    return _safe_float(
        proposal.get(
            "investment_usdt",
            proposal.get("size_usdt", 0.0),
        )
    )


def _proposal_confidence(proposal: Mapping[str, Any]) -> float:
    return _safe_float(
        proposal.get(
            "confidence",
            proposal.get("strategy_confidence", 0.0),
        )
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def engine_start() -> dict[str, Any]:
    global _engine_started_at
    runtime_guard = enforce_bundled_runtime()
    _engine_started_at = datetime.now(timezone.utc)
    runtime_state = load()
    ensure_runtime_state(runtime_state)
    start_runtime_session(
        runtime_state,
        mode=str(getattr(settings, "mode", "paper")),
        required_fresh_scans=3,
    )
    save(runtime_state)
    # GEX is refreshed asynchronously and can never block the normal scan or
    # change network/restart safety state.
    refresh_if_due()
    cleaned_test_records = cleanup_old_test_proposals()
    cleaned_duplicates = cleanup_duplicate_proposals()
    archived_old_proposals = ProposalHistoryManager.cleanup()
    archived_stale_paper_proposals = _cleanup_stale_autonomous_paper_proposals()
    reconcile_summary = reconcile_proposal_reservations()

    _safe_audit(
        "INFO",
        "system",
        f"APEXION Engine {ENGINE_VERSION} started",
        details={
            "mode": getattr(settings, "mode", "paper"),
            "version": ENGINE_VERSION,
            "approval_required": getattr(
                settings,
                "effective_approval_required",
                True,
            ),
            "live_execution_enabled": getattr(
                settings,
                "live_execution_enabled",
                False,
            ),
            "sys_executable": runtime_guard.get("executable"),
            "bundled_runtime": runtime_guard.get("bundled"),
            "proposal_reconcile": reconcile_summary,
        },
    )

    return {
        "status": "started",
        "version": ENGINE_VERSION,
        "started_at": _engine_started_at.isoformat(),
        "mode": getattr(settings, "mode", "paper"),
        "sys_executable": runtime_guard.get("executable"),
        "bundled_runtime": runtime_guard.get("bundled"),
        "cleaned_test_records": cleaned_test_records,
        "cleaned_duplicate_records": cleaned_duplicates,
        "archived_old_proposals": archived_old_proposals,
        "archived_stale_paper_proposals": archived_stale_paper_proposals,
        "proposal_reconcile": reconcile_summary,
        "startup_recovery": dict(runtime_state.get("startup_recovery") or {}),
    }


def engine_stop() -> dict[str, Any]:
    state = load()
    stop_runtime_session(state)
    save(state)
    _safe_audit(
        "INFO",
        "system",
        f"APEXION Engine {ENGINE_VERSION} stopping",
    )

    return {
        "status": "stopped",
        "version": ENGINE_VERSION,
        "stopped_at": _utc_now(),
        "clean_shutdown": True,
    }


def engine_status() -> dict[str, Any]:
    proposals = get_all_proposals()
    state = load()

    return {
        "engine": {
            "version": ENGINE_VERSION,
            "started_at": (
                _engine_started_at.isoformat()
                if _engine_started_at
                else None
            ),
            "mode": getattr(settings, "mode", "paper"),
            "approval_required": getattr(
                settings,
                "effective_approval_required",
                True,
            ),
            "live_execution_enabled": getattr(
                settings,
                "live_execution_enabled",
                False,
            ),
            "emergency_stop": is_emergency_stopped(),
        },
        "risk": risk_summary(),
        "runtime_session": dict(state.get("runtime_session") or {}),
        "startup_recovery": dict(state.get("startup_recovery") or {}),
        "live_broker_reconciliation": dict(state.get("live_broker_reconciliation") or {}),
        "portfolio": portfolio_summary(),
        "portfolio_intelligence": {
            "reports": len(
                state.get("portfolio_intelligence_reports", [])
            ),
            "last_report": (
                state.get("portfolio_intelligence_reports", [])[-1]
                if state.get("portfolio_intelligence_reports")
                else None
            ),
        },
        "proposals": {
            "pending": sum(
                1
                for proposal in proposals
                if proposal.get("status") == "pending"
            ),
            "blocked": sum(
                1
                for proposal in proposals
                if proposal.get("status") == "blocked"
            ),
            "approved": sum(
                1
                for proposal in proposals
                if proposal.get("status") == "approved"
            ),
            "executed": sum(
                1
                for proposal in proposals
                if proposal.get("status") == "executed"
            ),
            "total": len(proposals),
        },
        "ai": {
            "minimum_ai_score": ProposalEngine.MIN_AI_SCORE,
            "minimum_confidence_pct": (
                ProposalEngine.MIN_CONFIDENCE * 100.0
            ),
            "minimum_rotation_gap": (
                PortfolioIntelligence.MIN_ROTATION_GAP
            ),
        },
        "legacy": legacy_status(),
    }


# ---------------------------------------------------------------------------
# Proposal reads
# ---------------------------------------------------------------------------


def get_pending_proposals() -> list[dict[str, Any]]:
    cleanup_old_test_proposals()
    cleanup_duplicate_proposals()
    state = load()

    pending = [
        dict(proposal)
        for proposal in state.get("proposals", [])
        if proposal.get("status") == "pending"
        and not _proposal_is_test_record(proposal)
    ]

    return list(
        reversed(
            sorted(
                pending,
                key=_proposal_created_at,
            )
        )
    )


def get_all_proposals() -> list[dict[str, Any]]:
    """
    Dashboardweergave:

    - alle pending voorstellen eerst;
    - daarna uitsluitend de meest recente afgehandelde records;
    - oude testdata en dubbele IDs worden verborgen.
    """
    cleanup_old_test_proposals()
    cleanup_duplicate_proposals()
    state = load()

    unique: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []

    for proposal in state.get("proposals", []):
        if _proposal_is_test_record(proposal):
            continue

        item = dict(proposal)
        proposal_id = str(item.get("id", "")).strip()

        if proposal_id:
            unique[proposal_id] = item
        else:
            anonymous.append(item)

    proposals = [
        *unique.values(),
        *anonymous,
    ]

    proposals.sort(
        key=_proposal_created_at,
        reverse=True,
    )

    pending = [
        item
        for item in proposals
        if item.get("status") == "pending"
    ]
    recent = [
        item
        for item in proposals
        if item.get("status") != "pending"
    ]

    room = max(
        0,
        MAX_DASHBOARD_PROPOSALS - len(pending),
    )

    return [
        *pending,
        *recent[:room],
    ]


def get_portfolio_intelligence_reports() -> list[dict[str, Any]]:
    state = load()
    return list(
        reversed(
            state.get("portfolio_intelligence_reports", [])
        )
    )


# ---------------------------------------------------------------------------
# AI layers
# ---------------------------------------------------------------------------


def _calculate_ai_layers(
    analysis: CoinAnalysis,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = _object_to_dict(analysis.metrics)

    score = ScoreEngine.calculate(metrics)
    score_dict = score.to_dict()

    confidence = ConfidenceEngine.calculate(
        metrics=metrics,
        score_breakdown=score_dict,
    )
    confidence_dict = confidence.to_dict()

    return score_dict, confidence_dict


def _build_safe_execution_plan(
    analysis: CoinAnalysis,
    requested_size_usdt: float,
    *,
    risk_max_per_coin_usdt: float | None = None,
) -> ExecutionPlan | None:
    try:
        return build_execution_plan(
            analysis,
            size_usdt=requested_size_usdt,
            risk_max_per_coin_usdt=risk_max_per_coin_usdt,
        )
    except Exception as exc:
        _safe_audit(
            "ERROR",
            "strategy",
            f"Execution plan build failed for {analysis.coin}: {exc}",
            coin=analysis.coin,
            details={"error": str(exc)},
        )
        return None


def _attach_plan_data(
    proposal: dict[str, Any],
    plan: ExecutionPlan | None,
    analysis: CoinAnalysis,
) -> dict[str, Any]:
    if plan is not None:
        proposal["plan_id"] = getattr(plan, "plan_id", "")
        proposal["plan_status"] = getattr(plan, "status", "")
        proposal["plan_params"] = _object_to_dict(
            getattr(plan, "params", {})
        )
        proposal["risk_assessment"] = _object_to_dict(
            getattr(plan, "risk_assessment", {})
        )

    proposal.setdefault("risk_assessment", {})
    proposal["risk_assessment"].setdefault(
        "current_price",
        _safe_float(
            getattr(analysis.metrics, "price_usdt", 0.0)
        ),
    )

    return proposal


def _attach_legacy_enrichment(
    proposal: dict[str, Any],
) -> dict[str, Any]:
    try:
        extra = enrich_strategy_plan(
            proposal.get("coin", ""),
            proposal.get("strategy", ""),
            proposal.get("plan_params", {}),
        )
    except Exception:
        extra = None

    if extra:
        proposal.setdefault("legacy", {}).update(extra)

    return proposal


# ---------------------------------------------------------------------------
# Proposal creation
# ---------------------------------------------------------------------------


def _surface_proposal_block(proposal: dict[str, Any], reason: str) -> dict[str, Any]:
    """Keep top-level rejection_reason in sync with metadata reasons."""
    text = str(reason or "").strip()
    if not text:
        return proposal
    proposal["status"] = "blocked"
    proposal["qualifies"] = False
    metadata = dict(proposal.get("metadata") or {})
    reasons = [
        str(item)
        for item in list(metadata.get("rejection_reasons") or [])
        if str(item).strip()
    ]
    if text not in reasons:
        reasons.append(text)
    metadata["rejection_reasons"] = reasons
    proposal["metadata"] = metadata
    existing = str(proposal.get("rejection_reason") or "").strip()
    if not existing:
        proposal["rejection_reason"] = text
    elif text not in existing:
        proposal["rejection_reason"] = f"{existing}; {text}"
    return proposal


def create_proposal_from_analysis(
    analysis: CoinAnalysis,
    size_usdt: float | None = None,
    *,
    authoritative_trade: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    strategy = _normalise_strategy(
        getattr(
            analysis,
            "recommended_strategy",
            "",
        )
    )

    if strategy not in PROPOSAL_STRATEGIES:
        return {
            "id": f"blocked_{uuid.uuid4().hex[:12]}",
            "coin": str(
                getattr(analysis, "coin", "")
            ).upper(),
            "strategy": strategy or "unsupported",
            "status": "blocked",
            "qualifies": False,
            "created_at": _utc_now(),
            "rejection_reason": (
                "strategie wordt niet ondersteund voor voorstellen"
            ),
            "metadata": {
                "rejection_reasons": [
                    "strategie wordt niet ondersteund "
                    "voor voorstellen"
                ]
            },
        }

    duplicate = _active_duplicate(
        coin=getattr(analysis, "coin", ""),
        strategy=strategy,
    )
    if duplicate is not None:
        result = dict(duplicate)
        result.setdefault("metadata", {})
        result["metadata"]["duplicate_reused"] = True
        return result

    if _coin_has_open_position(
        getattr(analysis, "coin", "")
    ):
        return {
            "id": f"blocked_{uuid.uuid4().hex[:12]}",
            "coin": str(
                getattr(analysis, "coin", "")
            ).upper(),
            "strategy": strategy,
            "status": "blocked",
            "qualifies": False,
            "created_at": _utc_now(),
            "rejection_reason": "er staat al een positie open voor deze coin",
            "metadata": {
                "rejection_reasons": [
                    "er staat al een positie open voor deze coin"
                ]
            },
        }

    requested_size = _safe_float(
        size_usdt,
        _safe_float(
            getattr(settings, "default_investment_usdt", 10.0),
            10.0,
        ),
    )
    gex_overlay = dict(
        (getattr(analysis, "intelligence", {}) or {}).get("gex_overlay") or {}
    )
    allocation_decision = _build_capital_allocation(
        analysis,
        strategy=strategy,
        authoritative_trade=authoritative_trade,
    )
    if allocation_decision is not None and allocation_decision.allowed:
        requested_size = allocation_decision.approved_allocation_usdt
    else:
        # Do not feed approved_allocation_usdt=0 into ProposalEngine.create.
        # That made the primary rejection "investeringsbedrag moet positief
        # zijn" even when the allocator had a real cash/slot/correlation
        # blocker. Keep the default size for evaluation; the allocator
        # blocker is applied as the primary reason below.
        requested_size = effective_size(requested_size, gex_overlay)

    score_dict, confidence_dict = _calculate_ai_layers(analysis)
    # Canonical AI score is ScoreEngine's bounded 0–100 value. Never overwrite
    # it with unbounded strategy_competition.risk_adjusted_score.
    canonical_ai_score = max(
        0.0,
        min(100.0, _safe_float(score_dict.get("ai_score"))),
    )
    score_dict["ai_score"] = round(canonical_ai_score, 2)
    if "total" in score_dict:
        score_dict["total"] = round(canonical_ai_score / 100.0, 4)
        score_dict["ai_score_total"] = round(canonical_ai_score, 2)
        score_dict["ai_score_max"] = 100.0
    if authoritative_trade:
        authoritative = dict(authoritative_trade)
        authoritative_confidence = _safe_float(
            authoritative.get("calibrated_confidence")
        )
        candidate_score = _safe_float(
            dict(authoritative.get("candidate") or {}).get(
                "risk_adjusted_score"
            )
        )
        confidence_dict["confidence"] = authoritative_confidence
        confidence_dict["confidence_pct"] = authoritative_confidence * 100.0
        confidence_dict["confidence_label"] = "final intelligence TRADE"
        # Competition score lives under its own explicit field — not ai_score.
        score_dict["risk_adjusted_competition_score"] = candidate_score
        score_dict["strategy_competition_score"] = candidate_score * 100.0

    plan = _build_safe_execution_plan(
        analysis,
        requested_size,
        risk_max_per_coin_usdt=(
            allocation_decision.single_position_cap_usdt
            if allocation_decision is not None and allocation_decision.allowed
            else None
        ),
    )
    if plan is not None and allocation_decision is not None and allocation_decision.allowed:
        _sync_plan_allocation(plan, allocation_decision)

    plan_size = requested_size
    if plan is not None:
        candidate_size = _safe_float(
            getattr(plan, "size_usdt", requested_size),
            requested_size,
        )
        if candidate_size > 0:
            plan_size = candidate_size

    max_per_coin = _safe_float(
        getattr(settings, "max_per_coin_usdt", 0.0),
        0.0,
    )
    if max_per_coin <= 0:
        max_per_coin = None

    try:
        available_usdt = get_available_balance()
    except Exception:
        available_usdt = None

    proposal_obj: TradeProposal = ProposalEngine.create(
        analysis=analysis,
        score_breakdown=score_dict,
        confidence_breakdown=confidence_dict,
        authoritative_trade=authoritative_trade,
        requested_investment_usdt=plan_size,
        approved_allocation_usdt=(
            allocation_decision.approved_allocation_usdt
            if allocation_decision is not None and allocation_decision.allowed
            else None
        ),
        max_per_coin_usdt=max_per_coin,
        available_usdt=available_usdt,
        reserve_ratio=_safe_float(
            getattr(settings, "reserve_ratio", 0.20),
            0.20,
        ),
    )

    proposal = proposal_obj.to_dict()
    proposal["strategy"] = strategy
    proposal["size_usdt"] = proposal["investment_usdt"]
    execution_supported, paper_only_flag = strategy_execution_status(strategy)
    proposal["paper_only"] = paper_only_flag
    proposal["execution_supported"] = execution_supported
    proposal.setdefault("metadata", {}).update({
        "decision_id": str(getattr(analysis, "decision_id", "") or ""),
        "decision_status": str(
            getattr(analysis, "decision_status", "") or ""
        ),
        "preferred_strategy": str(
            getattr(analysis, "preferred_strategy", "") or ""
        ),
        "market_intelligence": compact_market_intelligence(
            dict(getattr(analysis, "intelligence", {}) or {})
        ),
        "trade_plan": compact_trade_plan(
            dict(getattr(analysis, "trade_plan", {}) or {})
        ),
        "net_expected_edge_pct": _safe_float(
            dict(authoritative_trade or {}).get("net_edge_pct")
        ),
        "gex_overlay": deepcopy(gex_overlay),
    })
    if allocation_decision is not None:
        proposal["metadata"]["capital_allocation"] = allocation_decision.to_dict()
        if not allocation_decision.allowed:
            blocker = (
                allocation_decision.blocker
                or "Harde PAPER capital-allocation constraint blokkeert execution."
            )
            reasons = list(
                proposal.get("metadata", {}).get("rejection_reasons") or []
            )
            reasons = [
                item
                for item in reasons
                if item != "investeringsbedrag moet positief zijn"
            ]
            proposal.setdefault("metadata", {})["rejection_reasons"] = reasons
            existing = str(proposal.get("rejection_reason") or "")
            if "investeringsbedrag moet positief zijn" in existing:
                proposal["rejection_reason"] = ""
            _surface_proposal_block(proposal, blocker)

    proposal = _attach_plan_data(
        proposal,
        plan,
        analysis,
    )
    if strategy == "flywheel" and plan is not None:
        from app.strategy_engine import flywheel_dip_is_realistic

        plan_params = dict(proposal.get("plan_params") or {})
        metrics = getattr(analysis, "metrics", None)
        if not flywheel_dip_is_realistic(
            plan_params.get("buy_deviation_pct", 1.2),
            getattr(metrics, "change_24h_pct", 0.0) if metrics is not None else 0.0,
            getattr(metrics, "volatility_14d", 0.0) if metrics is not None else 0.0,
        ):
            _surface_proposal_block(
                proposal,
                "flywheel-dip is onrealistisch t.o.v. recente range",
            )

    if plan is not None and strategy in {"grid", "spot_grid"}:
        grid_params = dict(proposal.get("plan_params", {}) or {})
        if allocation_decision is not None:
            grid_params.setdefault(
                "single_position_cap_usdt",
                allocation_decision.single_position_cap_usdt,
            )
            grid_params.setdefault(
                "total_equity_usdt",
                allocation_decision.snapshot.total_equity_usdt,
            )
        grid_preview = preview_grid(
            entry_price=_safe_float(
                proposal.get("risk_assessment", {}).get("current_price")
            ),
            size_usdt=_safe_float(proposal.get("size_usdt"), plan_size),
            params=grid_params,
        )
        proposal["grid_preview"] = grid_preview
        if not grid_preview.get("net_edge_sufficient"):
            _surface_proposal_block(
                proposal,
                str(
                    grid_preview.get("rejection_reason")
                    or "onvoldoende netto grid-edge na fees en slippage"
                ),
            )

    if plan is None:
        _surface_proposal_block(
            proposal,
            "execution plan kon niet worden gebouwd",
        )

    elif getattr(plan, "status", "") == "rejected":
        _surface_proposal_block(
            proposal,
            "strategy/risk execution plan werd geweigerd",
        )

    proposal = _attach_legacy_enrichment(proposal)
    _store_proposal(proposal)

    level = "INFO" if proposal.get("qualifies") else "WARNING"
    _safe_audit(
        level,
        "execution",
        (
            f"AI proposal {proposal.get('status')} for "
            f"{proposal.get('coin')}: "
            f"score={proposal.get('ai_score')}, "
            f"confidence={proposal.get('confidence_pct')}%"
        ),
        coin=proposal.get("coin"),
        plan_id=proposal.get("plan_id", ""),
        details={"proposal": proposal},
    )

    return proposal


def create_proposal_for_coin(
    coin: str,
    size_usdt: float | None = None,
) -> dict[str, Any] | None:
    normalized_coin = str(coin or "").strip().upper()

    if not normalized_coin:
        return None

    if is_emergency_stopped():
        _safe_audit(
            "WARNING",
            "execution",
            f"Proposal blocked for {normalized_coin}: emergency stop",
            coin=normalized_coin,
        )
        return None

    try:
        scan: MarketScan = scan_watchlist([normalized_coin])
        state = load()
        guard = _observe_scan_and_save(state, scan)
        if not bool(guard.get("mode") == "ARMED" and guard.get("resume_ready", False)) or not actions_allowed(state):
            return None

        if not scan.analyses:
            return None

        return create_proposal_from_analysis(
            scan.analyses[0],
            size_usdt=size_usdt,
        )

    except Exception as exc:
        _safe_audit(
            "ERROR",
            "execution",
            f"Proposal creation failed for {normalized_coin}: {exc}",
            coin=normalized_coin,
            details={"error": str(exc)},
        )
        return None


def _market_data_failure_result(scan: MarketScan) -> dict[str, Any]:
    diagnostics = dict(getattr(scan, "market_data", {}) or {})
    exact_error = str(
        diagnostics.get("error")
        or diagnostics.get("last_http_status_error")
        or "Publieke marktdata leverde geen bruikbare ticker- en candledata."
    )
    market_waits = {
        "bot": [
            _wait_reason("market_data_error", exact_error),
            _wait_reason("market_data_error", exact_error),
        ],
        "spot": [_wait_reason("market_data_error", exact_error)],
    }
    slot_states = _slot_state_snapshot(market_waits)
    for item in slot_states:
        if item.get("wait"):
            item["status"] = "MARKET_DATA_ERROR"
    return {
        "status": "market_data_error",
        "reason": exact_error,
        "scan": {
            "scanned_at": scan.scanned_at,
            "coins_analyzed": scan.coins_analyzed,
            "coins_skipped": scan.coins_skipped,
            "universe_count": getattr(scan, "universe_count", 0),
            "valid_market_count": getattr(scan, "valid_market_count", 0),
        },
        "market_scan": asdict(scan),
        "position_updates": [],
        "active_execution": {},
        "auto_opened_positions": [],
        "position_decisions": [],
        "trade_manager_decisions": [],
        "report": {
            "decision": {
                "action": "market_data_error",
                "reason": exact_error,
                "should_close": False,
                "should_open": False,
                "requires_approval": False,
            }
        },
        "proposals": [],
        "proposal": None,
        "ranked_opportunities": [],
        "slot_state": {"open_count": len(get_open_positions())},
        "slot_states": slot_states,
        "selected_candidates": [],
        "last_paper_action": {
            "action": "WAIT",
            "timestamp": _utc_now(),
            **_wait_reason("market_data_error", exact_error),
        },
        "last_error": exact_error,
        "skipped_candidates": diagnostics.get("symbols_rejected", []),
    }


def _network_guard_failure_result(
    scan: MarketScan,
    guard: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a visible WAIT result without touching positions or proposals."""
    guard_copy = dict(guard or {})
    guard_reason = str(
        guard_copy.get("last_reason")
        or "Netwerkbeveiliging is niet gewapend."
    )
    status = (
        "network_guard_recovery"
        if str(guard_copy.get("mode", "")).upper() == "RECOVERY"
        or bool(guard_copy.get("recovery_mode", False))
        else "network_guard_blocked"
    )
    wait = _wait_reason("network_guard", guard_reason)
    market_waits = {"bot": [wait, wait], "spot": [wait]}
    slot_states = _slot_state_snapshot(market_waits)
    for item in slot_states:
        if item.get("wait"):
            item["status"] = "NETWORK_GUARD"
    diagnostics = dict(getattr(scan, "market_data", {}) or {})
    return {
        "status": status,
        "reason": guard_reason,
        "scan": {
            "scanned_at": scan.scanned_at,
            "coins_analyzed": scan.coins_analyzed,
            "coins_skipped": scan.coins_skipped,
            "universe_count": getattr(scan, "universe_count", 0),
            "valid_market_count": getattr(scan, "valid_market_count", 0),
        },
        "market_scan": asdict(scan),
        "network_guard": guard_copy,
        "position_updates": [],
        "active_execution": {},
        "auto_opened_positions": [],
        "position_decisions": [],
        "trade_manager_decisions": [],
        "hard_exit_closed_positions": [],
        "report": {
            "decision": {
                "action": "network_guard_wait",
                "reason": guard_reason,
                "should_close": False,
                "should_open": False,
                "requires_approval": False,
            }
        },
        "proposals": [],
        "proposal": None,
        "ranked_opportunities": [],
        "slot_state": {"open_count": len(get_open_positions())},
        "slot_states": slot_states,
        "selected_candidates": [],
        "last_paper_action": {
            "action": "WAIT",
            "timestamp": _utc_now(),
            **wait,
            "network_guard": guard_copy,
        },
        "last_error": None,
        "skipped_candidates": diagnostics.get("symbols_rejected", []),
    }


def scan_and_create_top_proposal(
    watchlist: list[str] | None = None,
    size_usdt: float | None = None,
) -> dict[str, Any]:
    refresh_if_due()
    scan = scan_watchlist(watchlist)
    if getattr(scan, "market_data_status", "") == "error":
        state = load()
        guard = _observe_scan_and_save(state, scan)
        return _network_guard_failure_result(scan, guard)
    state = load()
    guard = _observe_scan_and_save(state, scan)
    if not bool(guard.get("mode") == "ARMED" and guard.get("resume_ready", False)) or not actions_allowed(state):
        return _network_guard_failure_result(scan, guard)
    position_updates = update_open_positions_from_scan(scan)

    result: dict[str, Any] = {
        "scanned_at": scan.scanned_at,
        "coins_analyzed": scan.coins_analyzed,
        "coins_skipped": scan.coins_skipped,
        "proposal": None,
        "position_updates": position_updates,
        "active_execution": active_execution,
        "auto_opened_positions": auto_opened,
    }

    for analysis in scan.analyses:
        strategy = _normalise_strategy(
            getattr(
                analysis,
                "recommended_strategy",
                "",
            )
        )

        if strategy not in PROPOSAL_STRATEGIES:
            continue

        if _coin_has_open_position(
            getattr(analysis, "coin", "")
        ):
            continue

        if _active_duplicate(
            coin=getattr(analysis, "coin", ""),
            strategy=strategy,
        ) is not None:
            continue

        score_dict, confidence_dict = _calculate_ai_layers(analysis)

        if (
            _safe_float(score_dict.get("ai_score"))
            < ProposalEngine.MIN_AI_SCORE
        ):
            continue

        if (
            _safe_float(confidence_dict.get("confidence"))
            < ProposalEngine.MIN_CONFIDENCE
        ):
            continue

        if str(analysis.risk_level).lower() == "high":
            continue

        result["proposal"] = create_proposal_from_analysis(
            analysis,
            size_usdt=size_usdt,
        )
        break

    return result


# ---------------------------------------------------------------------------
# Paper position market-price synchronisation
# ---------------------------------------------------------------------------


def _analysis_market_price(
    analysis: CoinAnalysis | None,
) -> float:
    if analysis is None:
        return 0.0

    metrics = getattr(analysis, "metrics", None)
    return _safe_float(
        getattr(metrics, "price_usdt", 0.0),
        0.0,
    )


def update_open_positions_from_scan(
    scan: MarketScan,
) -> dict[str, Any]:
    """
    Synchroniseer alle open paperposities met de nieuwste prijzen uit een scan.

    Deze functie:
    - koppelt positiecoin aan CoinAnalysis;
    - schrijft current_price terug naar de store;
    - laat portfolio_manager de unrealized PnL herberekenen;
    - verandert niets wanneer voor een coin geen geldige prijs beschikbaar is.
    """
    analyses_by_coin = analysis_by_coin_from_scan(scan)

    updated: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for position in get_open_positions():
        coin = str(position.get("coin", "")).upper()
        position_id = str(position.get("position_id", ""))

        analysis = analyses_by_coin.get(coin)
        market_price = _analysis_market_price(analysis)

        if not position_id or market_price <= 0.0:
            skipped.append(
                {
                    "position_id": position_id,
                    "coin": coin,
                    "reason": "geen geldige actuele marktprijs",
                }
            )
            continue

        refreshed = update_position(
            position_id,
            market_price,
        )

        if refreshed is None:
            skipped.append(
                {
                    "position_id": position_id,
                    "coin": coin,
                    "reason": "positie kon niet worden bijgewerkt",
                }
            )
            continue

        persist_scan_intelligence_on_position(position_id, analysis)
        updated.append(refreshed)

    if updated:
        _safe_audit(
            "INFO",
            "portfolio",
            (
                f"{len(updated)} open paperpositie(s) bijgewerkt "
                "met actuele scanprijzen."
            ),
            details={
                "updated_positions": updated,
                "skipped_positions": skipped,
            },
        )

    return {
        "updated": len(updated),
        "skipped": len(skipped),
        "positions": updated,
        "skipped_positions": skipped,
    }


def refresh_open_position_price(
    position_id: str,
) -> dict[str, Any] | None:
    """
    Haal vlak vóór handmatig sluiten nog één actuele prijs op.

    Zo wordt een positie niet afgesloten tegen een oude opgeslagen prijs wanneer
    de gebruiker sinds de laatste scan niet opnieuw heeft gescand.
    """
    target = next(
        (
            position
            for position in get_open_positions()
            if str(position.get("position_id", "")) == str(position_id)
        ),
        None,
    )

    if target is None:
        return None

    coin = str(target.get("coin", "")).upper()
    if not coin:
        return target

    try:
        scan = scan_watchlist([coin])
        state = load()
        guard = _observe_scan_and_save(state, scan)
        if not bool(guard.get("mode") == "ARMED" and guard.get("resume_ready", False)) or not actions_allowed(state):
            _safe_audit(
                "WARNING",
                "portfolio",
                f"Close-prijs niet vrijgegeven door network guard: {guard.get('last_reason')}",
                coin=coin,
                details={"network_guard": network_guard_snapshot(state)},
            )
            return target
        update_open_positions_from_scan(scan)
    except Exception as exc:
        try:
            state = load()
            guard = _observe_scan_and_save(
                state,
                MarketScan(
                    scanned_at=_utc_now(),
                    market_data_status="error",
                    market_data_connected=False,
                    market_data={
                        "symbols_requested": [coin],
                        "symbols_successfully_loaded": [],
                        "freshness_status": "unavailable",
                    },
                ),
            )
            save(state)
        except Exception:
            guard = None
        _safe_audit(
            "WARNING",
            "portfolio",
            (
                f"Actuele prijs voor {coin} kon vóór sluiten "
                f"niet worden vernieuwd: {exc}"
            ),
            coin=coin,
            details={
                "error": str(exc),
                "network_guard": network_guard_snapshot(state) if guard is not None else {},
            },
        )

    return next(
        (
            position
            for position in get_open_positions()
            if str(position.get("position_id", "")) == str(position_id)
        ),
        target,
    )




# ---------------------------------------------------------------------------
# V3 multi-opportunity portfolio helpers
# ---------------------------------------------------------------------------

def _proposal_key(coin: str, strategy: str) -> tuple[str, str]:
    return str(coin or "").upper(), _normalise_strategy(strategy)


def _active_proposal_keys() -> set[tuple[str, str]]:
    state = load()
    keys: set[tuple[str, str]] = set()
    for item in state.get("proposals", []) or []:
        if str(item.get("status", "")).lower() not in ACTIVE_PROPOSAL_STATUSES:
            continue
        keys.add(_proposal_key(item.get("coin", ""), item.get("strategy", "")))
    return keys


def _position_management_snapshot(
    positions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Beoordeel elke bestaande positie zonder tijdgestuurde sluiting.

    Grid/DCA blijven lopen. Alleen een harde strategiegrens wordt als
    close_candidate gemarkeerd; uitvoering blijft handmatig/approval-based.
    """
    decisions: list[dict[str, Any]] = []

    for position in positions:
        assessment = LifetimeManager.assess(position)
        action = "hold"
        reason = "Strategie blijft actief; geen harde exit."

        if assessment.get("exit_signal") == "max_loss":
            action = "close_candidate"
            reason = "Maximale verliesgrens van de strategie is bereikt."
        elif assessment.get("exit_signal") == "target_profit":
            action = "review_profit"
            reason = "Winstdoel bereikt; positie mag worden herbeoordeeld."

        item = {
            "position_id": position.get("position_id"),
            "coin": str(position.get("coin", "")).upper(),
            "strategy": _normalise_strategy(position.get("strategy", "")),
            "action": action,
            "reason": reason,
            "lifetime": assessment,
        }
        decisions.append(item)

        DecisionLogger.log(
            action=action,
            coin=item["coin"],
            strategy=item["strategy"],
            reason=reason,
            details={"position_id": item["position_id"], "lifetime": assessment},
        )

    return decisions


def _create_ranked_portfolio_proposals_legacy(
    scan: MarketScan,
    *,
    size_usdt: float | None = None,
    max_new_proposals: int | None = None,
) -> dict[str, Any]:
    """
    Maak een gebalanceerde reeks voorstellen voor alle resterende slots.

    Prioriteit:
    1. tel open posities én pending/approved voorstellen als bezette slots;
    2. vul eerst ontbrekende actieve Spot-slots;
    3. vul daarna resterende Grid/DCA- of Spot-slots;
    4. maximaal één actieve reservering per coin;
    5. geen slechte alternatieve strategie forceren.
    """
    cleanup_duplicate_proposals()

    open_positions = get_open_positions()
    reservations = _pending_slot_reservations()
    simulated_positions = [
        *open_positions,
        *reservations,
    ]
    simulated = simulate_slot_state(simulated_positions)

    ranked = OpportunityRanker.rank(scan.analyses)
    active_keys = _active_proposal_keys()

    free_slots = max(0, simulated["max_active"] - simulated["open_count"])
    proposal_limit = free_slots if max_new_proposals is None else min(
        free_slots,
        max(0, int(max_new_proposals)),
    )

    created: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    selected_coins: set[str] = set(simulated["coins"])

    # Bouw per coin een actieve en een langlopende kandidaat.
    active_pool: list[tuple[float, CoinAnalysis, dict[str, Any]]] = []
    long_pool: list[tuple[float, CoinAnalysis, dict[str, Any]]] = []

    for analysis in scan.analyses:
        coin = str(getattr(analysis, "coin", "")).upper()
        if not coin or coin in selected_coins:
            continue

        active_candidate = _best_candidate_for_group(
            analysis,
            long_running=False,
        )
        if active_candidate is not None:
            active_pool.append(
                (_candidate_value(active_candidate), analysis, active_candidate)
            )

        long_candidate = _best_candidate_for_group(
            analysis,
            long_running=True,
        )
        if long_candidate is not None:
            long_pool.append(
                (_candidate_value(long_candidate), analysis, long_candidate)
            )

    active_pool.sort(key=lambda item: item[0], reverse=True)
    long_pool.sort(key=lambda item: item[0], reverse=True)

    def try_create(
        analysis: CoinAnalysis,
        candidate: dict[str, Any],
    ) -> bool:
        if len(created) >= proposal_limit:
            return False

        coin = str(getattr(analysis, "coin", "")).upper()
        strategy = _normalise_strategy(candidate.get("strategy", ""))
        key = _proposal_key(coin, strategy)

        if not coin or coin in selected_coins:
            return False
        if strategy not in PROPOSAL_STRATEGIES:
            skipped.append({"coin": coin, "strategy": strategy, "reason": "unsupported"})
            return False
        if key in active_keys:
            skipped.append({"coin": coin, "strategy": strategy, "reason": "active proposal exists"})
            return False

        allowed, reason = can_reserve_simulated_slot(
            simulated,
            coin,
            strategy,
        )
        if not allowed:
            skipped.append({"coin": coin, "strategy": strategy, "reason": reason})
            return False

        adapted = _analysis_with_strategy(analysis, candidate)
        score_dict, confidence_dict = _calculate_ai_layers(adapted)

        if _safe_float(score_dict.get("ai_score")) < ProposalEngine.MIN_AI_SCORE:
            skipped.append({"coin": coin, "strategy": strategy, "reason": "AI score too low"})
            return False
        if _safe_float(confidence_dict.get("confidence")) < ProposalEngine.MIN_CONFIDENCE:
            skipped.append({"coin": coin, "strategy": strategy, "reason": "confidence too low"})
            return False
        if str(getattr(adapted, "risk_level", "")).lower() == "high":
            skipped.append({"coin": coin, "strategy": strategy, "reason": "high risk"})
            return False

        proposal = create_proposal_from_analysis(
            adapted,
            size_usdt=size_usdt,
        )

        if proposal.get("status") not in {"pending", "approved", "executed"}:
            skipped.append({
                "coin": coin,
                "strategy": strategy,
                "reason": proposal.get("status", "blocked"),
            })
            return False

        reserve_simulated_slot(simulated, coin, strategy)
        active_keys.add(key)
        selected_coins.add(coin)
        created.append(proposal)

        DecisionLogger.log(
            action="proposal",
            coin=coin,
            strategy=strategy,
            reason="Gebalanceerde tournamentkans voor beschikbaar portfolioslot.",
            details={
                "candidate": candidate,
                "slot_state": {
                    **simulated,
                    "coins": sorted(simulated["coins"]),
                },
            },
        )
        return True

    # Eerst actief Spot-slot aanvullen.
    missing_active = max(
        0,
        simulated["reserved_spot"] - simulated["active_spot_count"],
    )
    for _, analysis, candidate in active_pool:
        if missing_active <= 0 or len(created) >= proposal_limit:
            break
        if try_create(analysis, candidate):
            missing_active -= 1

    # Daarna alle resterende kansen op kwaliteit rangschikken.
    combined = [
        *[(value, analysis, candidate) for value, analysis, candidate in active_pool],
        *[(value, analysis, candidate) for value, analysis, candidate in long_pool],
    ]
    combined.sort(key=lambda item: item[0], reverse=True)

    for _, analysis, candidate in combined:
        if len(created) >= proposal_limit:
            break
        try_create(analysis, candidate)

    cleanup_duplicate_proposals()

    return {
        "created": created,
        "created_count": len(created),
        "skipped": skipped,
        "ranked_opportunities": [item.to_dict() for item in ranked],
        "simulated_slot_state": {
            **simulated,
            "coins": sorted(simulated["coins"]),
        },
        "reserved_existing_proposals": reservations,
    }


def _disabled_strategy_names() -> set[str]:
    raw = str(getattr(settings, "disabled_strategies", "") or "")
    return {
        _normalise_strategy(part)
        for part in raw.split(",")
        if part.strip()
    }


def create_ranked_portfolio_proposals(
    scan: MarketScan,
    *,
    size_usdt: float | None = None,
    max_new_proposals: int | None = None,
) -> dict[str, Any]:
    """Vul flexibele PAPER-slots met de beste geldige spot-only kansen."""
    cleanup_duplicate_proposals()
    open_positions = get_open_positions()
    reservations = _pending_slot_reservations()
    simulated = simulate_slot_state([*open_positions, *reservations])
    ranked = OpportunityRanker.rank(scan.analyses)
    active_keys = _active_proposal_keys()
    free_slots = max(0, simulated["max_active"] - simulated["open_count"])
    proposal_limit = free_slots if max_new_proposals is None else min(
        free_slots,
        max(0, int(max_new_proposals)),
    )

    created: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failures: dict[str, list[dict[str, str]]] = {"bot": [], "spot": []}
    selected_coins: set[str] = set(simulated["coins"])
    active_pool: list[tuple[float, CoinAnalysis, dict[str, Any]]] = []
    bot_pool: list[tuple[float, CoinAnalysis, dict[str, Any]]] = []
    authoritative_snapshots: dict[str, dict[str, Any]] = {}
    from app.regime_filter import regime_allows_entries

    regime_ok, regime_detail = regime_allows_entries()

    for analysis in scan.analyses:
        coin = str(getattr(analysis, "coin", "")).upper()
        if not coin:
            continue
        authoritative, authoritative_blocker = _authoritative_trade_snapshot(
            analysis
        )
        if coin in selected_coins:
            duplicate = _wait_reason(
                "duplicate_exposure",
                f"{coin} is al actief of gereserveerd.",
            )
            failures["bot"].append(duplicate)
            failures["spot"].append(duplicate)
            if authoritative is not None:
                _set_execution_state(
                    scan,
                    analysis,
                    status="BLOCKED",
                    blocker=duplicate["detail"],
                    snapshot=authoritative,
                )
            continue
        if authoritative is not None:
            strategy = _normalise_strategy(
                authoritative.get("preferred_strategy", "")
            )
            group = (
                "bot"
                if strategy in {"grid", "spot_grid", "dca", "flywheel"}
                else "spot"
            )
            if strategy in _disabled_strategy_names():
                authoritative_blocker = (
                    f"strategie '{strategy}' is uitgeschakeld "
                    "(APEXION_DISABLED_STRATEGIES)"
                )
            if not regime_ok:
                authoritative_blocker = regime_detail
            if authoritative_blocker:
                wait = _wait_reason(
                    "approval_gate",
                    f"{coin}: {authoritative_blocker}",
                )
                failures[group].append(wait)
                skipped.append({
                    "coin": coin,
                    "strategy": strategy,
                    **wait,
                })
                _set_execution_state(
                    scan,
                    analysis,
                    status="BLOCKED",
                    blocker=wait["detail"],
                    snapshot=authoritative,
                )
                continue
            authoritative_snapshots[coin] = authoritative
            _set_execution_state(
                scan,
                analysis,
                status="ELIGIBLE",
                snapshot=authoritative,
            )
            candidate = dict(authoritative.get("candidate") or {})
            ranking_value = _candidate_value(candidate)
            if group == "bot":
                bot_pool.append((ranking_value, analysis, candidate))
            else:
                active_pool.append((ranking_value, analysis, candidate))
            continue

        if not regime_ok:
            continue

        opportunity = _safe_float(getattr(analysis, "opportunity_score", 0.0))
        active_candidate = _best_candidate_for_group(analysis, long_running=False)
        if active_candidate is not None:
            active_pool.append((
                opportunity * 0.70 + _candidate_value(active_candidate) * 0.30,
                analysis,
                active_candidate,
            ))
        bot_candidate = _best_candidate_for_group(analysis, long_running=True)
        if bot_candidate is not None:
            bot_pool.append((
                opportunity * 0.70 + _candidate_value(bot_candidate) * 0.30,
                analysis,
                bot_candidate,
            ))

    active_pool.sort(key=lambda item: item[0], reverse=True)
    bot_pool.sort(key=lambda item: item[0], reverse=True)

    def reject(
        group: str,
        coin: str,
        strategy: str,
        category: str,
        detail: str,
    ) -> bool:
        wait = _wait_reason(category, detail)
        failures[group].append(wait)
        skipped.append({"coin": coin, "strategy": strategy, **wait})
        return False

    def try_create(
        analysis: CoinAnalysis,
        candidate: dict[str, Any],
        group: str,
    ) -> bool:
        if len(created) >= proposal_limit:
            return False

        coin = str(getattr(analysis, "coin", "")).upper()
        strategy = _normalise_strategy(candidate.get("strategy", ""))
        opportunity = _safe_float(
            getattr(analysis, "opportunity_score", candidate.get("score", 0.0))
        )
        key = _proposal_key(coin, strategy)
        if not coin:
            return False
        if coin in selected_coins:
            return reject(group, coin, strategy, "duplicate_exposure", f"{coin} is al geselecteerd.")
        if strategy not in PROPOSAL_STRATEGIES or strategy == "hold":
            return reject(group, coin, strategy, "strategy_mismatch", f"Niet-uitvoerbare strategie: {strategy}.")
        if key in active_keys:
            return reject(group, coin, strategy, "duplicate_exposure", "Er bestaat al een actief voorstel.")

        allowed, slot_detail = can_reserve_simulated_slot(simulated, coin, strategy)
        if not allowed:
            return reject(
                group,
                coin,
                strategy,
                _categorize_wait_detail(slot_detail),
                slot_detail,
            )

        adapted = _analysis_with_strategy(analysis, candidate)
        authoritative = authoritative_snapshots.get(coin)
        if authoritative is None:
            score_dict, confidence_dict = _calculate_ai_layers(adapted)
            ai_score = _safe_float(score_dict.get("ai_score"))
            confidence = _safe_float(confidence_dict.get("confidence"))
            if ai_score < ProposalEngine.MIN_AI_SCORE:
                return reject(
                    group,
                    coin,
                    strategy,
                    "approval_gate",
                    f"AI-score {ai_score:.2f} lager dan {ProposalEngine.MIN_AI_SCORE:.2f}.",
                )
            if confidence < ProposalEngine.MIN_CONFIDENCE:
                return reject(
                    group,
                    coin,
                    strategy,
                    "low_confidence",
                    f"Confidence {confidence:.2f} lager dan {ProposalEngine.MIN_CONFIDENCE:.2f}.",
                )
        if str(getattr(adapted, "risk_level", "")).lower() == "high":
            return reject(group, coin, strategy, "high_risk", f"{coin} is high risk.")

        proposal = create_proposal_from_analysis(
            adapted,
            size_usdt=size_usdt,
            authoritative_trade=authoritative,
        )
        if proposal.get("status") not in {"pending", "approved", "executed"}:
            detail = "; ".join(
                str(item)
                for item in (proposal.get("metadata", {}).get("rejection_reasons", []) or [])
            ) or str(proposal.get("status", "blocked"))
            if authoritative is not None:
                _set_execution_state(
                    scan,
                    analysis,
                    status="BLOCKED",
                    blocker=detail,
                    snapshot=authoritative,
                )
            return reject(
                group,
                coin,
                strategy,
                _categorize_wait_detail(detail),
                detail,
            )

        _set_execution_state(
            scan,
            analysis,
            status="PENDING_APPROVAL",
            snapshot=authoritative,
        )

        reserve_simulated_slot(simulated, coin, strategy)
        active_keys.add(key)
        selected_coins.add(coin)
        created.append(proposal)
        DecisionLogger.log(
            action="proposal",
            coin=coin,
            strategy=strategy,
            reason="Risk-adjusted kandidaat kwalificeert voor het vaste paper-slot.",
            details={"candidate": candidate, "opportunity_score": opportunity},
        )
        return True

    # Eerst de minimale direct-spot capaciteit invullen wanneer die nog ontbreekt.
    missing_spot_minimum = max(
        0,
        simulated["reserved_spot"] - simulated["active_spot_count"],
    )
    for _, analysis, candidate in active_pool:
        if missing_spot_minimum <= 0 or len(created) >= proposal_limit:
            break
        if try_create(analysis, candidate, "spot"):
            missing_spot_minimum -= 1

    # Daarna concurreren alle resterende strategieen op kwaliteit. Zo blijven
    # vrije slots niet kunstmatig BOT- of DIRECT-SPOT-only wanneer de andere
    # groep duidelijk betere kansen heeft.
    combined_pool = [
        *[(value, analysis, candidate, "spot") for value, analysis, candidate in active_pool],
        *[(value, analysis, candidate, "bot") for value, analysis, candidate in bot_pool],
    ]
    combined_pool.sort(key=lambda item: item[0], reverse=True)
    for _, analysis, candidate, group in combined_pool:
        if len(created) >= proposal_limit:
            break
        coin = str(getattr(analysis, "coin", "")).upper()
        if coin in selected_coins:
            continue
        try_create(analysis, candidate, group)

    missing_total = max(0, simulated["max_active"] - simulated["open_count"])
    if missing_total and not failures["bot"] and not failures["spot"]:
        failures["spot"] = [
            _wait_reason("no_valid_candidate", "Geen extra kwalificerende kans voor een vrij flexibel slot.")
        ]

    legacy_ranked_by_coin = {
        item.coin: item.to_dict() for item in ranked
    }
    ranked_opportunities: list[dict[str, Any]] = []
    for analysis in scan.analyses:
        coin = str(getattr(analysis, "coin", "")).upper()
        authoritative = authoritative_snapshots.get(coin)
        if authoritative is None:
            if coin in legacy_ranked_by_coin:
                ranked_opportunities.append(legacy_ranked_by_coin[coin])
            continue
        candidate = dict(authoritative.get("candidate") or {})
        ranked_opportunities.append({
            "coin": coin,
            "strategy": str(authoritative.get("winner") or ""),
            "opportunity_score": _safe_float(candidate.get("risk_adjusted_score")),
            "confidence": _safe_float(authoritative.get("calibrated_confidence")),
            "risk_level": str(getattr(analysis, "risk_level", "medium")),
            "long_running": str(authoritative.get("winner") or "")
            in {"grid", "spot_grid", "dca", "flywheel"},
            "expected_edge": _safe_float(authoritative.get("net_edge_pct")),
            "expected_reward": _safe_float(
                candidate.get("expected_gross_edge_pct")
            ),
            "downside_risk": _safe_float(candidate.get("downside_risk_pct")),
            "reward_risk_ratio": _safe_float(authoritative.get("reward_risk")),
            "trend_score": 0.0,
            "momentum_score": 0.0,
            "regime_fit": _safe_float(candidate.get("market_regime_fit")),
            "liquidity_score": _safe_float(
                getattr(getattr(analysis, "metrics", None), "liquidity_score", 0.0)
            ),
            "learning_factor": _safe_float(
                getattr(analysis, "learning_factor", 1.0), 1.0
            ),
            "reason": str(authoritative.get("reason") or ""),
            "qualification_source": "final_intelligence_trade",
            "decision_id": str(authoritative.get("decision_id") or ""),
            "evidence_timestamp": str(
                authoritative.get("evidence_timestamp") or ""
            ),
        })
    ranked_opportunities.sort(
        key=lambda item: (
            _safe_float(item.get("opportunity_score")),
            _safe_float(item.get("confidence")),
            _safe_float(item.get("expected_edge")),
        ),
        reverse=True,
    )

    cleanup_duplicate_proposals()
    return {
        "created": created,
        "created_count": len(created),
        "skipped": skipped,
        "ranked_opportunities": ranked_opportunities,
        "simulated_slot_state": {
            **simulated,
            "coins": sorted(simulated["coins"]),
        },
        "reserved_existing_proposals": reservations,
        "slot_wait_reasons": {
            "flex": [
                dict(
                    ((failures["spot"] or failures["bot"]) or [_wait_reason("no_valid_candidate")])[0]
                )
                for _ in range(max(0, simulated["max_active"] - simulated["open_count"]))
            ],
            "bot": [],
            "spot": [],
        },
    }


def _slot_state_snapshot(
    slot_wait_reasons: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    waits = dict(slot_wait_reasons or {})
    bot_waits = list(waits.get("bot", []) or [])
    spot_waits = list(waits.get("spot", []) or [])
    flex_waits = list(waits.get("flex", []) or [])
    positions = get_open_positions()
    max_slots = int(getattr(settings, "max_active_bots", 4) or 4)
    by_slot = {
        int(item.get("slot_id", 0) or 0): item
        for item in positions
        if 1 <= int(item.get("slot_id", 0) or 0) <= max_slots
    }

    result: list[dict[str, Any]] = []
    for slot_id in range(1, max_slots + 1):
        position = by_slot.get(slot_id)
        if position is not None:
            strategy = _normalise_strategy(position.get("strategy", ""))
            slot_type = (
                "BOT"
                if strategy in {"grid", "spot_grid", "dca", "flywheel"}
                else "DIRECT_SPOT"
            )
            result.append({
                "slot_id": slot_id,
                "slot_type": slot_type,
                "status": "ACTIVE",
                "position_id": position.get("position_id"),
                "coin": position.get("coin"),
                "strategy": position.get("strategy"),
                "wait": None,
            })
            continue

        wait = None
        for source in (flex_waits, spot_waits, bot_waits):
            if source:
                wait = dict(source.pop(0))
                break
        if wait is None:
            wait = _wait_reason("approval_gate", "Geen uitgevoerde positie voor dit flexibele slot.")
        result.append({
            "slot_id": slot_id,
            "slot_type": "FLEX",
            "status": "CASH_WAIT",
            "position_id": None,
            "coin": None,
            "strategy": None,
            "wait": wait,
        })
    return result


# ---------------------------------------------------------------------------
# Portfolio Intelligence
# ---------------------------------------------------------------------------


def run_portfolio_intelligence(
    watchlist: list[str] | None = None,
    *,
    scan: MarketScan | None = None,
) -> dict[str, Any]:
    """
    Scan markten, heranalyseer open posities en beslis:

    - hold
    - open
    - replace
    - close

    Deze functie voert de beslissing niet automatisch uit.
    """
    refresh_if_due()
    if is_emergency_stopped():
        report = {
            "generated_at": _utc_now(),
            "decision": {
                "action": "hold",
                "reason": "Emergency stop is actief.",
                "should_close": False,
                "should_open": False,
                "requires_approval": False,
            },
        }
        return _store_rotation_report(report)

    positions = get_open_positions()
    open_coins = [
        str(position.get("coin", "")).upper()
        for position in positions
        if position.get("coin")
    ]

    requested_watchlist = list(
        dict.fromkeys(
            [
                *(watchlist or []),
                *open_coins,
            ]
        )
    )

    if scan is None:
        scan = scan_watchlist(requested_watchlist or None)
    position_updates = update_open_positions_from_scan(scan)
    positions = get_open_positions()

    analyses_by_coin = analysis_by_coin_from_scan(scan)

    confidence_by_coin: dict[str, dict[str, Any]] = {}

    for coin, analysis in analyses_by_coin.items():
        _, confidence_dict = _calculate_ai_layers(analysis)
        confidence_by_coin[coin] = confidence_dict

    best_analysis: CoinAnalysis | None = None
    best_confidence: dict[str, Any] | None = None
    best_proposal_preview: dict[str, Any] | None = None

    for analysis in scan.analyses:
        strategy = _normalise_strategy(
            getattr(
                analysis,
                "recommended_strategy",
                "",
            )
        )

        if strategy not in PROPOSAL_STRATEGIES:
            continue

        score_dict, confidence_dict = _calculate_ai_layers(analysis)

        if (
            _safe_float(score_dict.get("ai_score"))
            < ProposalEngine.MIN_AI_SCORE
        ):
            continue

        if (
            _safe_float(confidence_dict.get("confidence"))
            < ProposalEngine.MIN_CONFIDENCE
        ):
            continue

        if str(analysis.risk_level).lower() == "high":
            continue

        best_analysis = analysis
        best_confidence = confidence_dict

        exec_supported, paper_only_preview = strategy_execution_status(strategy)
        best_proposal_preview = {
            "ai_score": score_dict.get("ai_score", 0.0),
            "confidence_pct": confidence_dict.get(
                "confidence_pct",
                0.0,
            ),
            "risk_level": analysis.risk_level,
            "execution_supported": exec_supported,
            "paper_only": paper_only_preview,
        }
        break

    report_obj = PortfolioIntelligence.build_report(
        positions=positions,
        analyses_by_coin=analyses_by_coin,
        confidence_by_coin=confidence_by_coin,
        best_analysis=best_analysis,
        best_confidence=best_confidence,
        best_proposal=best_proposal_preview,
        max_positions=int(
            getattr(settings, "max_active_bots", 1)
        ),
    )

    report = report_obj.to_dict()
    report["scan"] = {
        "scanned_at": scan.scanned_at,
        "coins_analyzed": scan.coins_analyzed,
        "coins_skipped": scan.coins_skipped,
    }

    _store_rotation_report(report)

    decision = report.get("decision", {})
    _safe_audit(
        "INFO",
        "portfolio",
        (
            f"Portfolio Intelligence: "
            f"{decision.get('action')} — "
            f"{decision.get('reason')}"
        ),
        details={"report": report},
    )

    return report


def create_proposal_from_portfolio_decision(
    report: Mapping[str, Any],
    size_usdt: float | None = None,
) -> dict[str, Any] | None:
    """
    Maak uitsluitend een nieuw voorstel voor een open/replace-beslissing.

    Er wordt niets gesloten en niets geopend zonder aparte goedkeuring.
    """
    decision = dict(report.get("decision", {}))
    action = str(decision.get("action", "")).lower()
    candidate_coin = str(
        decision.get("candidate_coin", "")
    ).upper()

    if action not in {"open", "replace"}:
        return None

    if not candidate_coin:
        return None

    proposal = create_proposal_for_coin(
        candidate_coin,
        size_usdt=size_usdt,
    )

    if proposal is None:
        return None

    proposal.setdefault("metadata", {})
    proposal["metadata"]["portfolio_decision"] = decision

    _store_proposal(proposal)
    return proposal



def run_ai_trade_manager(
    scan: MarketScan,
) -> list[dict[str, Any]]:
    """
    Beoordeel alle open posities tegen de nieuwste opportunity ranking.
    Geen automatische uitvoering.
    """
    ranked = OpportunityRanker.rank(scan.analyses)
    positions = get_open_positions()
    scan_analyses = list(analysis_by_coin_from_scan(scan).values())
    decisions = AITradeManager.assess_positions(
        positions,
        scan_analyses,
        ranked,
    )

    payload: list[dict[str, Any]] = []
    for decision in decisions:
        item = decision.to_dict()
        payload.append(item)

        DecisionLogger.log(
            action=item["action"],
            coin=item["coin"],
            strategy=item["strategy"],
            reason=item["reason"],
            details={
                "position_id": item["position_id"],
                "pnl_pct": item["pnl_pct"],
                "replacement_coin": item["replacement_coin"],
                "replacement_strategy": item["replacement_strategy"],
                "replacement_score": item["replacement_score"],
                "suggested_strategy": item["suggested_strategy"],
                "current_confidence": item["current_confidence"],
                "market_score": item["market_score"],
                "risk_level": item["risk_level"],
                "trend_strength": item["trend_strength"],
                "momentum_rsi": item["momentum_rsi"],
                "breakout_pressure": item["breakout_pressure"],
                "breakdown_pressure": item["breakdown_pressure"],
            },
        )

    state = load()
    history = list(state.get("trade_manager_reports", []) or [])
    history.append(
        {
            "generated_at": _utc_now(),
            "decisions": payload,
        }
    )
    state["trade_manager_reports"] = history[-200:]
    save(state)

    return payload


def get_trade_manager_reports() -> list[dict[str, Any]]:
    state = load()
    return list(reversed(state.get("trade_manager_reports", []) or []))


def _ai_hard_exits_permitted() -> bool:
    if bool(getattr(settings, "simple_exits", False)):
        return False
    mode = str(getattr(settings, "mode", "paper"))
    live_enabled = bool(getattr(settings, "live_execution_enabled", False))
    return is_paper_execution_mode(mode) or is_micro_live_execution_mode(
        mode, live_enabled
    )


def _analyses_from_open_positions(
    positions: Sequence[Mapping[str, Any]],
) -> list[Any]:
    """Build TM analysis rows from persisted live intelligence (fast cycle)."""
    rows: list[Any] = []
    for position in positions:
        intel = dict(position.get("current_intelligence") or {})
        if not intel:
            continue
        rows.append(
            SimpleNamespace(
                coin=str(position.get("coin", "")).upper(),
                decision_status=intel.get("decision_status"),
                recommended_strategy=intel.get("recommended_strategy"),
                strategy_confidence=intel.get("confidence"),
                risk_level=intel.get("risk_level"),
                composite_score=intel.get("opportunity_score")
                or intel.get("composite_score")
                or 0.0,
                breakout_pressure=intel.get("breakout_pressure") or 0.0,
                breakdown_pressure=intel.get("breakdown_pressure") or 0.0,
                metrics=SimpleNamespace(
                    trend_strength=intel.get("trend") or 0.0,
                    momentum_rsi=intel.get("rsi") or 50.0,
                ),
                intelligence={"competition": intel.get("competition") or {}},
            )
        )
    return rows


def execute_ai_hard_exits(
    decisions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Close CLOSE_CANDIDATE positions in PAPER and MICRO_LIVE.

    LIVE sells on the broker first via ``engine_close_position``. Reserved
    (unfilled) live slots are skipped. When ``decisions`` is omitted the
    Trade Manager re-assesses current open positions, including Bug A
    wait-scan health handling.
    """
    if not _ai_hard_exits_permitted():
        return []

    payload = decisions
    if payload is None:
        positions = [
            item
            for item in get_open_positions()
            if not is_live_reservation(item)
        ]
        assessed = AITradeManager.assess_positions(
            positions,
            _analyses_from_open_positions(positions),
            [],
        )
        payload = [item.to_dict() for item in assessed]

    closed: list[dict[str, Any]] = []
    for decision in payload:
        if str(decision.get("action", "")).upper() != "CLOSE_CANDIDATE":
            continue
        position_id = str(decision.get("position_id", "") or "")
        if not position_id:
            continue
        target = next(
            (
                item
                for item in get_open_positions()
                if str(item.get("position_id") or "") == position_id
            ),
            None,
        )
        if is_live_reservation(target):
            continue
        reason = str(decision.get("reason", "") or "Harde AI-exit bereikt.")
        result = engine_close_position(
            position_id,
            exit_reason=f"AI_HARD_EXIT: {reason}",
        )
        if not isinstance(result, dict):
            continue
        if str(result.get("status") or "") == "blocked":
            continue
        if str(result.get("status") or "open") == "open":
            continue
        closed.append(result)
    return closed


def _execute_paper_hard_exits(
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Back-compat wrapper; LIVE and PAPER both use execute_ai_hard_exits."""
    return execute_ai_hard_exits(decisions)


def _compact_portfolio_audit_details(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Build bounded cycle telemetry without embedding the complete scan.

    The full result remains available to the caller and in the dedicated
    portfolio report. Audit rotation should retain hours of decisions, not a
    handful of multi-megabyte candle snapshots.
    """

    def brief_rows(
        value: Any,
        fields: tuple[str, ...],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        return [
            {field: item.get(field) for field in fields if field in item}
            for item in list(value or [])[:limit]
            if isinstance(item, Mapping)
        ]

    report = dict(result.get("report") or {})
    return {
        "status": result.get("status"),
        "scan": dict(result.get("scan") or {}),
        "counts": {
            "position_updates": len(list(result.get("position_updates") or [])),
            "auto_opened_positions": len(
                list(result.get("auto_opened_positions") or [])
            ),
            "position_decisions": len(
                list(result.get("position_decisions") or [])
            ),
            "trade_manager_decisions": len(
                list(result.get("trade_manager_decisions") or [])
            ),
            "hard_exit_closed_positions": len(
                list(result.get("hard_exit_closed_positions") or [])
            ),
            "proposals": len(list(result.get("proposals") or [])),
            "skipped_candidates": len(
                list(result.get("skipped_candidates") or [])
            ),
        },
        "decision": dict(report.get("decision") or {}),
        "proposals": brief_rows(
            result.get("proposals"),
            ("id", "coin", "strategy", "status", "size_usdt"),
            limit=15,
        ),
        "selected_candidates": brief_rows(
            result.get("selected_candidates"),
            ("rank", "coin", "strategy", "opportunity_score"),
            limit=15,
        ),
        "last_paper_action": dict(result.get("last_paper_action") or {}),
        "network_guard": dict(result.get("network_guard") or {}),
        "last_error": result.get("last_error"),
    }


def arbitrate_portfolio_cycle_decision(
    *,
    intelligence_decision: Mapping[str, Any],
    hard_exit_closed: Sequence[Mapping[str, Any]],
    trade_manager_decisions: Sequence[Mapping[str, Any]],
    rotation_result: Mapping[str, Any] | None,
    proposal_batch: Mapping[str, Any] | None,
    approval_required: bool,
) -> dict[str, Any]:
    """
    Single canonical portfolio decision per cycle.

    Priority:
    1. hard safety exit (already executed)
    2. autonomous safe PAPER rotation
    3. new open proposals (free slots)
    4. trade-manager review signals
    5. hold / wait
    """
    intel = dict(intelligence_decision or {})
    rotation = dict(rotation_result or {})
    rotation_status = str(rotation.get("status", "")).lower()
    proposals = list((proposal_batch or {}).get("created") or [])

    if hard_exit_closed:
        closed_coins = ", ".join(str(item.get("coin", "")) for item in hard_exit_closed)
        return {
            "action": "close",
            "reason": f"Harde PAPER-exit uitgevoerd voor {closed_coins}.",
            "should_close": True,
            "should_open": bool(proposals),
            "requires_approval": False,
            "closed_position_ids": [
                item.get("position_id") for item in hard_exit_closed
            ],
            "pionex_write_calls": 0,
            "decision_source": "hard_safety_exit",
        }

    if rotation_status == "executed":
        closed = dict(rotation.get("closed_position") or {})
        opened = dict(rotation.get("new_position") or {})
        return {
            "action": "replace",
            "reason": (
                f"Autonome PAPER-rotatie: {closed.get('coin')} → "
                f"{opened.get('coin')} ({opened.get('strategy')})."
            ),
            "current_position_id": closed.get("position_id"),
            "current_coin": closed.get("coin"),
            "candidate_coin": opened.get("coin"),
            "should_close": True,
            "should_open": True,
            "requires_approval": False,
            "autonomous": True,
            "execution_id": rotation.get("execution_id"),
            "decision_source": "autonomous_paper_rotation",
            "pionex_write_calls": 0,
        }

    if rotation_status == "aborted_after_close":
        return {
            "action": "hold",
            "reason": (
                "Rotatie-close uitgevoerd; open na close geblokkeerd. "
                f"Cash vrij ({rotation.get('reason', '')}). "
                "Volgende scan zoekt nieuwe kansen — geen handmatige actie vereist."
            ),
            "should_close": False,
            "should_open": False,
            "requires_approval": False,
            "rotation_status": "ROTATION_ABORTED_AFTER_CLOSE",
            "decision_source": "rotation_abort_after_close",
            "pionex_write_calls": 0,
        }

    if rotation_status == "rejected":
        return {
            "action": "hold",
            "reason": f"Rotatie geblokkeerd: {rotation.get('reason', '')}",
            "should_close": False,
            "should_open": False,
            "requires_approval": False,
            "candidate_coin": intel.get("candidate_coin"),
            "decision_source": "rotation_rejected",
            "pionex_write_calls": 0,
        }

    if rotation_status == "duplicate_skipped":
        return {
            **intel,
            "action": "hold",
            "reason": "Rotatie al verwerkt in deze of eerdere cyclus.",
            "should_close": False,
            "should_open": False,
            "requires_approval": False,
            "decision_source": "rotation_duplicate_skipped",
            "pionex_write_calls": 0,
        }

    intel_action = str(intel.get("action", "hold")).lower()
    if intel_action == "replace":
        return {
            **intel,
            "requires_approval": True,
            "decision_source": "portfolio_intelligence",
            "pionex_write_calls": 0,
        }

    if proposals:
        best = proposals[0]
        return {
            "action": "open",
            "reason": (
                f"{len(proposals)} nieuwe gebalanceerde kans(en). "
                f"Beste: {best.get('coin')} {best.get('strategy')}."
            ),
            "candidate_coin": best.get("coin"),
            "candidate_strategy": best.get("strategy"),
            "should_close": False,
            "should_open": True,
            "requires_approval": approval_required,
            "decision_source": "ranked_proposals",
            "pionex_write_calls": 0,
        }

    rotate = next(
        (item for item in trade_manager_decisions if item.get("action") == "ROTATE_CANDIDATE"),
        None,
    )
    close_candidate = next(
        (item for item in trade_manager_decisions if item.get("action") == "CLOSE_CANDIDATE"),
        None,
    )
    if rotate:
        return {
            "action": "review",
            "reason": (
                f"Rotatiekandidaat: {rotate.get('coin')} → "
                f"{rotate.get('replacement_coin')} {rotate.get('replacement_strategy')}."
            ),
            "should_close": False,
            "should_open": False,
            "requires_approval": True,
            "decision_source": "trade_manager",
            "pionex_write_calls": 0,
        }
    if close_candidate:
        return {
            "action": "review",
            "reason": f"{close_candidate.get('coin')}: harde sluitkandidaat.",
            "should_close": False,
            "should_open": False,
            "requires_approval": True,
            "decision_source": "trade_manager",
            "pionex_write_calls": 0,
        }

    if (proposal_batch or {}).get("reserved_existing_proposals"):
        return {
            "action": "review",
            "reason": "Slots gereserveerd door pending voorstellen.",
            "should_close": False,
            "should_open": False,
            "requires_approval": True,
            "decision_source": "pending_proposals",
            "pionex_write_calls": 0,
        }

    if trade_manager_decisions:
        open_summary = ", ".join(
            f"{item.get('coin')} {item.get('strategy')}"
            for item in trade_manager_decisions
        )
        return {
            "action": "hold",
            "reason": f"Actieve portefeuille: {open_summary}. Geen harde exit of betere rotatie.",
            "should_close": False,
            "should_open": False,
            "requires_approval": False,
            "decision_source": "trade_manager_hold",
            "pionex_write_calls": 0,
        }

    return {
        "action": str(intel.get("action", "hold")),
        "reason": str(intel.get("reason", "Geen actie in deze cyclus.")),
        "should_close": bool(intel.get("should_close", False)),
        "should_open": bool(intel.get("should_open", False)),
        "requires_approval": bool(intel.get("requires_approval", False)),
        "decision_source": "portfolio_intelligence",
        "pionex_write_calls": 0,
    }


def run_validated_paper_rotation(
    portfolio_decision: Mapping[str, Any],
    scan: MarketScan,
    execution_id: str,
    *,
    proposal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
  Execute a FASE 4 gated PAPER rotation (autonomous or approval-backed).

    Does not bypass approval_required for non-rotation flows.
    """
    from app.paper_rotation import (
        evaluate_paper_rotation,
        execute_rotation_close,
        finalize_rotation_execution,
        validate_rotation_open,
    )

    if not is_paper_execution_mode(str(getattr(settings, "mode", "paper")).lower()):
        return {"status": "rejected", "reason": "paper_mode_required", "execution_id": execution_id}

    if bool(getattr(settings, "live_execution_enabled", False)):
        return {"status": "rejected", "reason": "live_execution_blocked", "execution_id": execution_id}

    state = load()
    if execution_id in {
        str(item) for item in list(state.get("rotation_execution_ids", []) or [])
    }:
        return {
            "status": "duplicate_skipped",
            "reason": "duplicate_rotation_execution",
            "execution_id": execution_id,
        }

    analyses = analysis_by_coin_from_scan(scan)
    old_coin = str(portfolio_decision.get("current_coin", "")).upper()
    candidate_coin = str(portfolio_decision.get("candidate_coin", "")).upper()
    current_analysis = analyses.get(old_coin)
    candidate_analysis = analyses.get(candidate_coin)

    rotation_state = load()
    rotation_evaluation = evaluate_paper_rotation(
        rotation_state,
        portfolio_decision,
        current_analysis=current_analysis,
        candidate_analysis=candidate_analysis,
        scan_scanned_at=scan.scanned_at,
        proposal_id=execution_id,
    )
    rotation_evaluation.audit["execution_id"] = execution_id
    save(rotation_state)

    if not rotation_evaluation.allowed:
        return {
            "status": "rejected",
            "reason": rotation_evaluation.reason,
            "execution_id": execution_id,
            "audit": rotation_evaluation.audit,
            "event": rotation_evaluation.event,
        }

    closed_position, rotation_evaluation = execute_rotation_close(
        rotation_state,
        rotation_evaluation,
        proposal_id=execution_id,
    )
    if closed_position is None:
        return {
            "status": "rejected",
            "reason": rotation_evaluation.reason,
            "execution_id": execution_id,
            "audit": rotation_evaluation.audit,
            "event": rotation_evaluation.event,
        }

    # learning_events already written by canonical close_position →
    # record_position_experience (idempotent). Do not double-feed here.

    if proposal is None:
        if candidate_analysis is None:
            return {
                "status": "aborted_after_close",
                "reason": "candidate_analysis_missing_post_close",
                "execution_id": execution_id,
                "closed_position": closed_position,
                "audit": rotation_evaluation.audit,
            }
        proposal = create_proposal_from_analysis(candidate_analysis)
        proposal.setdefault("metadata", {})
        proposal["metadata"]["portfolio_decision"] = dict(portfolio_decision)
        proposal["metadata"]["rotation_execution_id"] = execution_id

    proposal = dict(proposal)
    metadata = dict(proposal.get("metadata") or {})
    authoritative = dict(metadata.get("authoritative_trade") or {})
    if not authoritative and candidate_analysis is not None:
        authoritative = dict(
            getattr(candidate_analysis, "execution_snapshot", {}) or {}
        )
    strategy = _normalise_strategy(proposal.get("strategy", ""))
    entry_price = _safe_float(proposal.get("risk_assessment", {}).get("current_price"))
    if entry_price <= 0 and candidate_analysis is not None:
        entry_price = _safe_float(
            getattr(getattr(candidate_analysis, "metrics", None), "price_usdt", 0.0)
        )

    allocation_decision = _build_capital_allocation(
        candidate_analysis,
        strategy=strategy,
        authoritative_trade=authoritative or None,
    )
    allocation_usdt = (
        allocation_decision.approved_allocation_usdt
        if allocation_decision is not None and allocation_decision.allowed
        else 0.0
    )

    rotation_state = load()
    open_evaluation = validate_rotation_open(
        rotation_state,
        rotation_evaluation,
        proposal=proposal,
        candidate_analysis=candidate_analysis,
        scan_scanned_at=scan.scanned_at,
        allocation_usdt=allocation_usdt,
    )
    if not open_evaluation.allowed:
        return {
            "status": "aborted_after_close",
            "reason": open_evaluation.reason,
            "execution_id": execution_id,
            "closed_position": closed_position,
            "audit": open_evaluation.audit,
            "event": open_evaluation.event,
        }

    if allocation_decision is not None and allocation_decision.allowed:
        proposal["size_usdt"] = allocation_decision.approved_allocation_usdt
        metadata["capital_allocation"] = allocation_decision.to_dict()
        proposal["metadata"] = metadata

    plan_id = str(proposal.get("plan_id") or f"plan_{uuid.uuid4().hex[:12]}")
    intent = ExecutionIntent(
        intent_id=f"intent-{uuid.uuid4().hex[:14]}",
        proposal_id=str(proposal.get("id") or execution_id),
        plan_id=plan_id,
        coin=candidate_coin,
        strategy=strategy,
        slot_type=(
            "BOT"
            if strategy in {"grid", "spot_grid", "dca", "flywheel"}
            else "DIRECT_SPOT"
        ),
        size_usdt=_proposal_size(proposal),
        entry_price=entry_price,
        confidence=_proposal_confidence(proposal),
        risk_level=str(proposal.get("risk_level", "medium")),
        reason=str(
            proposal.get("rationale")
            or metadata.get("decision_rationale")
            or "Validated PAPER portfolio rotation."
        ),
        params=dict(proposal.get("plan_params", {}) or {}),
        decision_id=str(metadata.get("decision_id") or ""),
        decision_status=str(metadata.get("decision_status") or ""),
        preferred_strategy=str(metadata.get("preferred_strategy") or ""),
        market_intelligence=compact_market_intelligence(
            dict(metadata.get("market_intelligence") or {})
        ),
        trade_plan=compact_trade_plan(dict(metadata.get("trade_plan") or {})),
        execution_snapshot=compact_replay_snapshot(authoritative),
        capital_allocation=deepcopy(dict(metadata.get("capital_allocation") or {})),
    )
    outcome = execute_paper_intent_sync(intent)
    position = outcome.position

    if not outcome.success or position is None:
        return {
            "status": "aborted_after_close",
            "reason": outcome.error or outcome.message or "open_failed_after_close",
            "execution_id": execution_id,
            "closed_position": closed_position,
            "audit": rotation_evaluation.audit,
        }

    finalize_rotation_execution(
        load(),
        execution_id=execution_id,
        evaluation=rotation_evaluation,
        new_position=position,
    )

    return {
        "status": "executed",
        "reason": "rotation_executed",
        "execution_id": execution_id,
        "closed_position": closed_position,
        "new_position": position,
        "audit": rotation_evaluation.audit,
        "event": "ROTATION_EXECUTED",
        "pionex_write_calls": 0,
    }


def run_portfolio_cycle(
    watchlist: list[str] | None = None,
    size_usdt: float | None = None,
) -> dict[str, Any]:
    """
    Volledige APEXION v3.1 portfolio-cyclus:

    1. scan volledige markt;
    2. update prijzen/PnL van open posities;
    3. beoordeel alle bestaande posities (harde PAPER-exits waar toegestaan);
    4. portfolio intelligence + optionele autonome veilige PAPER-rotatie;
    5. rangschik nieuwe kansen en maak voorstellen voor vrije slots;
    6. normale opens volgen bestaande approval/auto_execute_paper policy;
    7. LIVE execution en broker-writes blijven geblokkeerd in PAPER mode.

    Autonomous PAPER rotation (FASE 4/4B) voert een gated replace uit zonder
    handmatige proposal-approval wanneer ``auto_execute_paper_rotation`` actief
    is en alle safety-gates slagen. Dit wijzigt ``approval_required`` niet voor
    andere flows.
    """
    refresh_if_due()
    reconcile_proposal_reservations()
    if is_emergency_stopped():
        emergency_waits = {
            "bot": [_wait_reason("emergency_stop"), _wait_reason("emergency_stop")],
            "spot": [_wait_reason("emergency_stop")],
        }
        return {
            "status": "blocked",
            "reason": "Emergency stop is actief.",
            "report": None,
            "proposals": [],
            "proposal": None,
            "slot_states": _slot_state_snapshot(emergency_waits),
            "last_paper_action": {
                "action": "WAIT",
                "category": "emergency_stop",
                "reason": WAIT_REASON_TEXT["emergency_stop"],
            },
        }

    scan = scan_watchlist(watchlist)
    if getattr(scan, "market_data_status", "") == "error":
        state = load()
        guard = _observe_scan_and_save(state, scan)
        return _network_guard_failure_result(scan, guard)
    state = load()
    guard = _observe_scan_and_save(state, scan)
    if not bool(guard.get("mode") == "ARMED" and guard.get("resume_ready", False)) or not actions_allowed(state):
        return _network_guard_failure_result(scan, guard)
    position_updates = update_open_positions_from_scan(scan)
    # Open exposure is managed by the independent fast PAPER runner. The heavy
    # intelligence cycle only reads its status and therefore cannot duplicate fills.
    active_execution = get_paper_execution_status()
    positions = get_open_positions()
    position_decisions = _position_management_snapshot(positions)
    trade_manager_decisions = run_ai_trade_manager(scan)
    hard_exit_closed = _execute_paper_hard_exits(trade_manager_decisions)
    if hard_exit_closed:
        positions = get_open_positions()
        active_execution = dict(active_execution or {})
        closed_count = len(hard_exit_closed)
        active_execution["managed_positions"] = int(
            active_execution.get("managed_positions", 0) or 0
        ) + closed_count
        active_execution["closed_positions"] = int(
            active_execution.get("closed_positions", 0) or 0
        ) + closed_count
        active_execution["events"] = int(
            active_execution.get("events", 0) or 0
        ) + closed_count
        active_execution["hard_exit_closed_positions"] = closed_count
        active_execution["pionex_write_calls"] = 0

    # Behoud het bestaande rapport voor dashboardcompatibiliteit.
    report = run_portfolio_intelligence(watchlist=watchlist, scan=scan)
    intelligence_decision = dict(report.get("decision") or {})

    rotation_result: dict[str, Any] | None = None
    auto_rotated: list[dict[str, Any]] = []
    from app.paper_rotation import (
        autonomous_paper_rotation_enabled,
        build_rotation_execution_id,
    )

    from app.regime_filter import regime_allows_entries

    if (
        autonomous_paper_rotation_enabled()
        and str(intelligence_decision.get("action", "")).lower() == "replace"
        and regime_allows_entries()[0]
    ):
        rotation_result = run_validated_paper_rotation(
            intelligence_decision,
            scan,
            build_rotation_execution_id(scan.scanned_at, intelligence_decision),
        )
        if str(rotation_result.get("status", "")).lower() == "executed":
            new_position = dict(rotation_result.get("new_position") or {})
            if new_position:
                auto_rotated.append(new_position)

    proposal_batch = create_ranked_portfolio_proposals(
        scan,
        size_usdt=size_usdt,
    )

    proposals = proposal_batch["created"]
    primary = proposals[0] if proposals else None

    auto_opened: list[dict[str, Any]] = []
    paper_execution_mode = is_paper_execution_mode(
        str(getattr(settings, "mode", "paper"))
    )
    micro_live_execution_mode = is_micro_live_execution_mode(
        str(getattr(settings, "mode", "paper")),
        bool(getattr(settings, "live_execution_enabled", False)),
    )
    approval_required = bool(
        getattr(settings, "effective_approval_required", True)
    )
    auto_paper = (
        bool(getattr(settings, "auto_execute_paper", True))
        and (paper_execution_mode or micro_live_execution_mode)
        and not approval_required
    )
    if auto_paper:
        for item in list(proposals):
            if item.get("status") == "pending":
                analysis = next(
                    (
                        candidate for candidate in scan.analyses
                        if candidate.coin.upper()
                        == str(item.get("coin", "")).upper()
                    ),
                    None,
                )
                opened = approve_proposal_by_id(str(item.get("id", "")))
                if opened is not None:
                    auto_opened.append(opened)
                    if analysis is not None:
                        _set_execution_state(
                            scan,
                            analysis,
                            status="EXECUTED",
                            snapshot=analysis.execution_snapshot,
                        )
                    continue
                stored = _find_proposal(str(item.get("id", ""))) or item
                detail = str(stored.get("execution_error") or "")
                if not detail:
                    detail = "; ".join(
                        str(reason)
                        for reason in list(
                            dict(stored.get("metadata") or {}).get(
                                "rejection_reasons", []
                            )
                            or []
                        )
                    )
                detail = detail or "PAPER execution heeft geen positie aangemaakt"
                if is_active_reservation_status(stored.get("status")):
                    finalize_proposal_failed(
                        str(item.get("id", "")),
                        reason=detail,
                        extra={
                            "status": "execution_failed",
                            "execution_failed_at": _utc_now(),
                        },
                    )
                strategy = _normalise_strategy(item.get("strategy", ""))
                group = (
                    "bot"
                    if strategy in {"grid", "spot_grid", "dca", "flywheel"}
                    else "spot"
                )
                wait = _wait_reason(_categorize_wait_detail(detail), detail)
                proposal_batch.setdefault("slot_wait_reasons", {}).setdefault(
                    group, []
                ).append(wait)
                if analysis is not None:
                    _set_execution_state(
                        scan,
                        analysis,
                        status="BLOCKED",
                        blocker=detail,
                        snapshot=analysis.execution_snapshot,
                    )
    elif proposals:
        for item in proposals:
            analysis = next(
                (
                    candidate for candidate in scan.analyses
                    if candidate.coin.upper()
                    == str(item.get("coin", "")).upper()
                ),
                None,
            )
            if analysis is not None:
                _set_execution_state(
                    scan,
                    analysis,
                    status="PENDING_APPROVAL",
                    blocker="Wacht op expliciete PAPER approval.",
                    snapshot=analysis.execution_snapshot,
                )

    slot_states = _slot_state_snapshot(proposal_batch.get("slot_wait_reasons"))
    if hard_exit_closed:
        last_paper_action: dict[str, Any] = {
            "action": "CLOSE",
            "timestamp": _utc_now(),
            "positions": [
                {
                    "position_id": item.get("position_id"),
                    "coin": item.get("coin"),
                    "strategy": item.get("strategy"),
                    "realized_pnl": item.get("realized_pnl"),
                    "exit_reason": item.get("exit_reason"),
                }
                for item in hard_exit_closed
            ],
            "reason": "Harde AI-exit(s) zijn uitsluitend in PAPER uitgevoerd.",
            "pionex_write_calls": 0,
        }
    elif auto_rotated:
        last_paper_action = {
            "action": "REPLACE",
            "timestamp": _utc_now(),
            "positions": [
                {
                    "position_id": item.get("position_id"),
                    "coin": item.get("coin"),
                    "strategy": item.get("strategy"),
                    "size_usdt": item.get("size_usdt"),
                }
                for item in auto_rotated
            ],
            "reason": "Autonome veilige PAPER-rotatie uitgevoerd.",
            "pionex_write_calls": 0,
        }
    elif rotation_result and str(rotation_result.get("status")) == "aborted_after_close":
        last_paper_action = {
            "action": "WAIT",
            "timestamp": _utc_now(),
            "reason": (
                "Rotatie-close voltooid; open geblokkeerd. Cash vrij — "
                "volgende scan zoekt nieuwe kansen."
            ),
            "category": "rotation_aborted_after_close",
            "pionex_write_calls": 0,
        }
    elif auto_opened:
        last_paper_action = {
            "action": "OPEN",
            "timestamp": _utc_now(),
            "positions": [
                {
                    "position_id": item.get("position_id"),
                    "coin": item.get("coin"),
                    "strategy": item.get("strategy"),
                    "slot_id": item.get("slot_id"),
                }
                for item in auto_opened
            ],
            "reason": "Gekwalificeerde ExecutionIntent(s) zijn paper-only uitgevoerd.",
        }
    else:
        first_wait = next(
            (item.get("wait") for item in slot_states if item.get("wait")),
            _wait_reason("no_valid_candidate"),
        )
        last_paper_action = {
            "action": "WAIT",
            "timestamp": _utc_now(),
            **dict(first_wait),
        }

    approval_required_flag = bool(
        getattr(settings, "effective_approval_required", True)
    )
    final_decision = arbitrate_portfolio_cycle_decision(
        intelligence_decision=intelligence_decision,
        hard_exit_closed=hard_exit_closed,
        trade_manager_decisions=trade_manager_decisions,
        rotation_result=rotation_result,
        proposal_batch=proposal_batch,
        approval_required=approval_required_flag,
    )
    report["intelligence_decision"] = intelligence_decision
    report["decision"] = final_decision
    report["final_decision"] = final_decision
    report["decision_owner"] = "portfolio_cycle_arbitration"
    if rotation_result is not None:
        report["rotation_result"] = rotation_result
    _store_rotation_report(report)

    result = {
        "status": "completed",
        "scan": {
            "scanned_at": scan.scanned_at,
            "coins_analyzed": scan.coins_analyzed,
            "coins_skipped": scan.coins_skipped,
            "universe_count": getattr(scan, "universe_count", len(scan.analyses)),
            "valid_market_count": getattr(scan, "valid_market_count", len(scan.analyses)),
        },
        "market_scan": asdict(scan),
        "position_updates": position_updates,
        "active_execution": active_execution,
        "auto_opened_positions": auto_opened,
        "position_decisions": position_decisions,
        "trade_manager_decisions": trade_manager_decisions,
        "hard_exit_closed_positions": hard_exit_closed,
        "autonomous_rotation": rotation_result,
        "auto_rotated_positions": auto_rotated,
        "report": report,
        "proposals": proposals,
        # Compatibiliteit met bestaand dashboard/scheduler.
        "proposal": primary,
        "ranked_opportunities": proposal_batch["ranked_opportunities"],
        "slot_state": proposal_batch["simulated_slot_state"],
        "slot_states": slot_states,
        "selected_candidates": [
            {
                "rank": index,
                "coin": analysis.coin,
                "strategy": analysis.recommended_strategy,
                "opportunity_score": analysis.opportunity_score,
            }
            for index, analysis in enumerate(scan.analyses, start=1)
        ],
        "last_paper_action": last_paper_action,
        "last_error": None,
        "skipped_candidates": proposal_batch["skipped"],
    }

    _safe_audit(
        "INFO",
        "portfolio",
        (
            f"V3.1 portfolio cycle: {len(positions)} open, "
            f"{len(proposals)} nieuw(e) voorstel(len)."
        ),
        details=_compact_portfolio_audit_details(result),
    )

    return result


# ---------------------------------------------------------------------------
# Proposal lifecycle
# ---------------------------------------------------------------------------


def _emergency_blocks_proposal_approval(
    proposal_id: str,
) -> bool:
    if is_emergency_stopped():
        _update_stored_proposal(
            proposal_id,
            {
                "status": "execution_failed",
                "execution_error": "emergency stop blokkeert PAPER execution",
            },
        )
        return True
    return False


def _allocation_experience(
    *,
    coin: str,
    strategy: str,
) -> tuple[int, float]:
    """Return minimum-sample-gated PAPER prediction quality inputs."""

    state = load()
    matching = [
        item
        for item in list(state.get("experience_memory_v2", []) or [])
        if str(item.get("mode", "paper")).lower() == "paper"
        and str(item.get("coin", "")).upper() == str(coin).upper()
        and _normalise_strategy(item.get("strategy", "")) == strategy
    ]
    unique = {
        str(item.get("evidence_id") or item.get("decision_id") or id(item)): item
        for item in matching
    }
    samples = list(unique.values())
    if not samples:
        return 0, 0.50
    average_quality = sum(
        _safe_float(item.get("quality")) for item in samples
    ) / len(samples)
    return len(samples), max(0.0, min(1.0, (average_quality + 1.0) / 2.0))


def _build_capital_allocation(
    analysis: CoinAnalysis,
    *,
    strategy: str,
    authoritative_trade: Mapping[str, Any] | None,
) -> CapitalAllocationDecision | None:
    authoritative = dict(authoritative_trade or {})
    if (
        not bool(getattr(settings, "dynamic_capital_allocation_enabled", True))
        or not is_paper_execution_mode(getattr(settings, "mode", "paper"))
        or not authoritative
        or str(authoritative.get("final_decision", "")).upper() != "TRADE"
    ):
        return None

    intelligence = dict(getattr(analysis, "intelligence", {}) or {})
    grid = dict(intelligence.get("grid_suitability") or {})
    profile = dict(intelligence.get("coin_profile") or {})
    portfolio = dict(intelligence.get("portfolio_context") or {})
    trade_plan = dict(
        getattr(analysis, "trade_plan", {})
        or intelligence.get("trade_plan", {})
        or {}
    )
    candidate = dict(authoritative.get("candidate") or {})
    positions = get_open_positions()
    ledger = capital_ledger()
    coin = str(getattr(analysis, "coin", "") or "").upper()
    samples, prediction_quality = _allocation_experience(
        coin=coin,
        strategy=strategy,
    )
    maximum_correlation = _safe_float(portfolio.get("maximum_correlation"))
    correlated_coin = str(portfolio.get("correlated_coin") or "").upper()
    existing_exposure = _safe_float(ledger.get("allocated_capital_usdt"))
    grid_count = max(5, int(_safe_float(trade_plan.get("grid_count"), 12.0)))
    buy_levels = sell_levels = 0
    if strategy in {"grid", "spot_grid"}:
        allocation_preview = preview_grid(
            entry_price=_safe_float(
                getattr(getattr(analysis, "metrics", None), "price_usdt", 0.0)
            ),
            size_usdt=max(
                1.0,
                _safe_float(
                    getattr(settings, "default_investment_usdt", 25.0),
                    25.0,
                ),
            ),
            params={
                "lower_price": trade_plan.get("grid_lower"),
                "upper_price": trade_plan.get("grid_upper"),
                "grid_count": grid_count,
                "grid_type": trade_plan.get("grid_type", "arithmetic"),
            },
        )
        buy_levels = len(list(allocation_preview.get("buy_levels") or []))
        sell_levels = len(list(allocation_preview.get("sell_levels") or []))
    inactive_ratio = max(0.0, min(1.0, _safe_float(grid.get("inactive_time_ratio"))))
    breakout_risk = max(0.0, min(1.0, _safe_float(grid.get("breakout_risk"))))
    trend_penalty = max(0.0, min(1.0, _safe_float(grid.get("trend_penalty"))))

    snapshot = CapitalAllocationSnapshot(
        coin=coin,
        strategy=strategy,
        total_equity_usdt=_safe_float(
            ledger.get("equity_usdt"),
            _safe_float(ledger.get("starting_capital_usdt")),
        ),
        cash_usdt=_safe_float(ledger.get("cash_usdt")),
        ledger_free_capital_usdt=_safe_float(ledger.get("free_capital_usdt")),
        current_portfolio_exposure_usdt=existing_exposure,
        current_asset_class_exposure_usdt=existing_exposure,
        current_correlated_exposure_usdt=correlated_exposure_usdt(
            positions,
            maximum_correlation=maximum_correlation,
            correlated_coin=correlated_coin,
        ),
        open_positions=len(positions),
        max_active_positions=int(getattr(settings, "max_active_bots", 1) or 1),
        confidence=_safe_float(authoritative.get("calibrated_confidence")),
        expected_net_edge_pct=_safe_float(authoritative.get("net_edge_pct")),
        volatility_pct=_safe_float(profile.get("typical_volatility_pct")),
        grid_suitability_score=_safe_float(
            grid.get("score"),
            _safe_float(authoritative.get("grid_suitability_score")),
        ),
        expected_fills_per_hour=_safe_float(grid.get("expected_fills_per_hour")),
        expected_cycles_per_hour=_safe_float(grid.get("expected_cycles_per_hour")),
        expected_net_profit_per_hour=_safe_float(
            grid.get("expected_net_profit_per_hour")
        ),
        expected_profit_reference_allocation_usdt=_safe_float(
            getattr(settings, "default_investment_usdt", 25.0), 25.0
        ),
        historical_quality=_safe_float(
            candidate.get("historical_fit"),
            _safe_float(
                dict(profile.get("strategy_fit") or {}).get(strategy),
                0.50,
            ),
        ),
        prediction_quality=prediction_quality,
        prediction_samples=samples,
        maximum_correlation=maximum_correlation,
        correlated_coin=correlated_coin,
        concentration_score=_safe_float(portfolio.get("concentration_score")),
        trend_penalty=trend_penalty,
        breakout_risk=breakout_risk,
        inventory_risk=0.0,
        stale_rebuild_risk=max(
            0.0,
            min(1.0, 0.50 * inactive_ratio + 0.30 * breakout_risk + 0.20 * trend_penalty),
        ),
        requested_grid_count=grid_count,
        requested_buy_levels=buy_levels,
        requested_sell_levels=sell_levels,
    )
    allocation = allocate_dynamic_capital(snapshot, policy_from_settings(settings))
    gex_overlay = dict(intelligence.get("gex_overlay") or {})
    return scale_allocation(allocation, gex_overlay)


def _sync_plan_allocation(
    plan: ExecutionPlan,
    decision: CapitalAllocationDecision,
) -> None:
    allocation = decision.to_dict()
    plan.size_usdt = decision.approved_allocation_usdt
    plan.paper_metadata = dict(getattr(plan, "paper_metadata", {}) or {})
    plan.paper_metadata["capital_allocation"] = allocation
    if plan.strategy in {"grid", "spot_grid"}:
        plan.params = dict(plan.params or {})
        plan.params["grid_count"] = decision.effective_grid_count
        plan.params["allocation_risk_band"] = decision.risk_band
        plan.params["min_economic_order_usdt"] = (
            decision.policy.min_economic_grid_order_usdt
        )
        plan.params["single_position_cap_usdt"] = (
            decision.single_position_cap_usdt
        )

    # build_execution_plan persisted the initial object before this enrichment.
    # Keep that diagnostic record consistent with the immutable proposal/intent.
    state = load()
    plans = list(state.get("execution_plans", []) or [])
    for item in plans:
        if item.get("plan_id") == plan.plan_id:
            item["size_usdt"] = plan.size_usdt
            item["params"] = dict(plan.params or {})
            item["paper_metadata"] = dict(plan.paper_metadata or {})
            break
    state["execution_plans"] = plans
    save(state)


def approve_proposal_by_id(
    proposal_id: str,
) -> dict[str, Any] | None:
    if _emergency_blocks_proposal_approval(proposal_id):
        return None

    proposal = _find_proposal(
        proposal_id,
        pending_only=False,
    )

    if proposal is None:
        return None

    current_status = str(proposal.get("status", "")).lower()
    if current_status not in {"pending", "approved", "executing"}:
        return None

    state = load()
    allowed, guard_reason = action_gate(
        state,
        "approve_proposal",
        action_timestamp=_proposal_created_at(proposal),
    )
    if not allowed:
        if "stale action" in guard_reason.lower():
            for stored_proposal in state.get("proposals", []):
                if stored_proposal.get("id") == proposal_id:
                    stored_proposal.update(
                        {
                            "status": "blocked",
                            "execution_error": guard_reason,
                            "execution_blocked_at": _utc_now(),
                        }
                    )
                    break
        save(state)
        _safe_audit(
            "WARNING",
            "execution",
            f"Proposal approval geblokkeerd door network guard: {guard_reason}",
            coin=proposal.get("coin"),
            plan_id=proposal.get("plan_id", ""),
            details={"network_guard": network_guard_snapshot(state)},
        )
        return None

    if not bool(proposal.get("qualifies", True)):
        return None

    metadata = dict(proposal.get("metadata") or {})
    authoritative = dict(metadata.get("authoritative_trade") or {})
    if authoritative:
        authoritative_errors: list[str] = []
        proposal_strategy = _normalise_strategy(proposal.get("strategy", ""))
        if str(authoritative.get("final_decision", "")).upper() != "TRADE":
            authoritative_errors.append("final intelligence decision is geen TRADE")
        if proposal_strategy != _normalise_strategy(authoritative.get("winner", "")):
            authoritative_errors.append("proposalstrategie wijkt af van intelligence-winnaar")
        if not bool(authoritative.get("candidate_eligible")):
            authoritative_errors.append("competition-winnaar is niet eligible")
        if not bool(authoritative.get("data_quality_trade_allowed")):
            authoritative_errors.append("data-quality gate blokkeert execution")
        if not bool(authoritative.get("snapshot_consistent")):
            authoritative_errors.append("decision- en execution-snapshot verschillen")
        authoritative_errors.extend(
            str(item)
            for item in list(authoritative.get("safety_blockers") or [])
            if str(item).strip()
        )
        if proposal_strategy in {"grid", "spot_grid"} and not bool(
            authoritative.get("grid_suitable")
        ):
            authoritative_errors.append("grid suitability gate blokkeert execution")
        if authoritative_errors:
            detail = "; ".join(dict.fromkeys(authoritative_errors))
            finalize_proposal_failed(proposal_id, reason=detail)
            return None
    else:
        confidence = _proposal_confidence(proposal)
        if confidence < ProposalEngine.MIN_CONFIDENCE:
            return None

        ai_score = _safe_float(
            proposal.get(
                "ai_score",
                _safe_float(proposal.get("composite_score"))
                * 100.0,
            )
        )
        ai_score = max(0.0, min(100.0, ai_score))
        if ai_score < ProposalEngine.MIN_AI_SCORE:
            return None

    from app.capital_stops import live_risk_level_from_state

    live_risk = live_risk_level_from_state(
        state,
        proposal.get("coin"),
        fallback=proposal.get("risk_level"),
        intelligence=dict(metadata.get("market_intelligence") or {}),
    )
    if live_risk == "high":
        return None

    mode = str(getattr(settings, "mode", "paper")).lower()
    live_enabled = bool(getattr(settings, "live_execution_enabled", False))
    paper_entry = is_paper_execution_mode(mode) and not live_enabled
    live_entry = is_micro_live_execution_mode(mode, live_enabled)
    if not paper_entry and not live_entry:
        finalize_proposal_failed(
            proposal_id,
            reason=(
                "execution entrypoint is PAPER/LIVE_DRY_RUN with live disabled, "
                "or MICRO_LIVE with live enabled"
            ),
        )
        return None

    portfolio_decision = (
        proposal.get("metadata", {})
        .get("portfolio_decision", {})
    )

    is_rotation_replace = (
        str(portfolio_decision.get("action", "")).lower() == "replace"
    )

    # Already-executing proposals must never open a second position.
    # Recovery is reconcile-only (timeout / linked position), not re-execution.
    if current_status == "executing":
        reconcile_proposal_reservations()
        existing = _find_proposal(proposal_id, pending_only=False) or {}
        existing_status = str(existing.get("status", "")).lower()
        if existing_status == "executed":
            position_id = str(existing.get("position_id") or "")
            if position_id:
                for item in load().get("positions", []) or []:
                    if str(item.get("position_id") or "") == position_id:
                        return dict(item)
            return existing
        return None

    # A. Atomically claim pending/approved → executing (held-lock transaction).
    # Only the winner of this claim may proceed to PAPER execution.
    claimed = claim_proposal_for_execution(proposal_id)
    if claimed is None:
        # Lost the race, or terminal — never duplicate execution.
        existing = _find_proposal(proposal_id, pending_only=False)
        if existing and str(existing.get("status", "")).lower() == "executed":
            position_id = str(existing.get("position_id") or "")
            if position_id:
                for item in load().get("positions", []) or []:
                    if str(item.get("position_id") or "") == position_id:
                        return dict(item)
            return existing
        return None

    if is_rotation_replace:
        if live_entry:
            finalize_proposal_failed(
                proposal_id,
                reason="live rotation is not supported",
            )
            return None
        old_coin = str(portfolio_decision.get("current_coin", "")).upper()
        candidate_coin = str(portfolio_decision.get("candidate_coin", "")).upper()
        scan = scan_watchlist([coin for coin in [old_coin, candidate_coin] if coin])
        # B. Execute PAPER rotation outside the claim lock.
        rotation_outcome = run_validated_paper_rotation(
            portfolio_decision,
            scan,
            execution_id=proposal_id,
            proposal=claimed,
        )
        status = str(rotation_outcome.get("status", "")).lower()
        if status == "executed":
            position = dict(rotation_outcome.get("new_position") or {})
            # C. Atomically finalize executing → executed.
            finalize_proposal_executed(
                proposal_id,
                position_id=str(position.get("position_id") or ""),
                execution_id=str(
                    rotation_outcome.get("execution_id")
                    or proposal_id
                ),
                extra={"rotation_audit": rotation_outcome.get("audit")},
            )
            return position
        if status == "aborted_after_close":
            finalize_proposal_failed(
                proposal_id,
                reason=str(rotation_outcome.get("reason") or "rotation_aborted_after_close"),
                extra={
                    "status": "rotation_aborted_after_close",
                    "rotation_audit": rotation_outcome.get("audit"),
                },
            )
            return _find_proposal(proposal_id, pending_only=False)
        if status == "duplicate_skipped":
            finalize_proposal_failed(
                proposal_id,
                reason="duplicate_skipped",
            )
            return _find_proposal(proposal_id, pending_only=False)
        finalize_proposal_failed(
            proposal_id,
            reason=str(rotation_outcome.get("reason") or "rotation_failed"),
            extra={"rotation_audit": rotation_outcome.get("audit")},
        )
        return None

    plan_id = str(claimed.get("plan_id") or f"plan_{uuid.uuid4().hex[:12]}")
    coin = str(claimed.get("coin", "")).upper()
    strategy = _normalise_strategy(claimed.get("strategy", ""))
    entry_price = _safe_float(
        claimed.get("risk_assessment", {}).get("current_price")
    )
    if entry_price <= 0:
        finalize_proposal_failed(proposal_id, reason="invalid current price")
        return None

    execution_id = f"exec-{uuid.uuid4().hex[:14]}"
    intent = ExecutionIntent(
        intent_id=f"intent-{uuid.uuid4().hex[:14]}",
        proposal_id=proposal_id,
        plan_id=plan_id,
        coin=coin,
        strategy=strategy,
        slot_type=(
            "BOT"
            if strategy in {"grid", "spot_grid", "dca", "flywheel"}
            else "DIRECT_SPOT"
        ),
        size_usdt=_proposal_size(claimed),
        entry_price=entry_price,
        confidence=_proposal_confidence(claimed),
        ai_score=round(max(0.0, min(100.0, _safe_float(claimed.get("ai_score")))), 2),
        risk_level=str(claimed.get("risk_level", "medium")),
        reason=str(
            claimed.get("rationale")
            or claimed.get("metadata", {}).get("decision_rationale")
            or "Gekwalificeerde risk-adjusted paper opportunity."
        ),
        params=dict(claimed.get("plan_params", {}) or {}),
        decision_id=str(metadata.get("decision_id") or ""),
        decision_status=str(metadata.get("decision_status") or ""),
        preferred_strategy=str(metadata.get("preferred_strategy") or ""),
        market_intelligence=compact_market_intelligence(
            dict(metadata.get("market_intelligence") or {})
        ),
        trade_plan=compact_trade_plan(dict(metadata.get("trade_plan") or {})),
        execution_snapshot=compact_replay_snapshot(authoritative),
        capital_allocation=deepcopy(
            dict(metadata.get("capital_allocation") or {})
        ),
    )
    # B. Execute PAPER or MICRO_LIVE DIRECT_SPOT outside the claim lock.
    if live_entry:
        if not is_micro_live_direct_spot_strategy(strategy):
            finalize_proposal_failed(
                proposal_id,
                reason=f"MICRO_LIVE DIRECT_SPOT ondersteunt '{strategy}' niet",
            )
            return None
        if not is_live_strategy_supported(strategy):
            finalize_proposal_failed(
                proposal_id,
                reason=f"LIVE is niet vrijgegeven voor '{strategy}'",
            )
            return None
        outcome = execute_live_intent_sync(intent)
    else:
        outcome = execute_paper_intent_sync(intent)
    position = outcome.position

    if not outcome.success or position is None:
        # D. Execution failure: executing → failed.
        finalize_proposal_failed(
            proposal_id,
            reason=str(outcome.error or outcome.message or "paper_execution_failed"),
            extra={
                "execution_intent_id": intent.intent_id,
                "execution_outcome_id": outcome.outcome_id,
                "execution_id": execution_id,
            },
        )
        return None

    # Persist proposal/execution linkage on the position.
    if isinstance(position, dict):
        position["proposal_id"] = proposal_id
        position["execution_id"] = str(
            outcome.execution_id or execution_id or intent.intent_id
        )
        state = load()
        for stored in list(state.get("positions", []) or []):
            if str(stored.get("position_id") or "") == str(position.get("position_id") or ""):
                stored["proposal_id"] = proposal_id
                stored["execution_id"] = position["execution_id"]
                break
        save(state)

    # C. Atomically finalize executing → executed.
    finalize_proposal_executed(
        proposal_id,
        position_id=str(position.get("position_id") or ""),
        execution_id=str(position.get("execution_id") or execution_id),
        extra={
            "execution_intent_id": intent.intent_id,
            "execution_outcome_id": outcome.outcome_id,
        },
    )

    _safe_audit(
        "INFO",
        "execution",
        f"Paper position opened: {coin}",
        coin=coin,
        plan_id=plan_id,
        position_id=position.get("position_id"),
        details={
            "intent": intent.to_dict(),
            "outcome": outcome.to_dict(),
            "position": position,
            "proposal_id": proposal_id,
            "execution_id": position.get("execution_id"),
        },
    )

    return position


def reject_proposal_by_id(
    proposal_id: str,
    reason: str = "",
) -> dict[str, Any] | None:
    proposal = _find_proposal(proposal_id)

    if proposal is None:
        return None

    if proposal.get("status") not in {"pending", "blocked"}:
        return None

    rejected = _update_stored_proposal(
        proposal_id,
        {
            "status": "rejected",
            "rejected_at": _utc_now(),
            "rejection_reason": str(reason or "").strip(),
        },
    )

    if rejected is not None:
        _safe_audit(
            "INFO",
            "execution",
            f"Proposal rejected: {proposal_id}",
            plan_id=str(rejected.get("plan_id", "")) or None,
            coin=str(rejected.get("coin", "")) or None,
            details={"reason": str(reason or "").strip()},
        )

    return rejected


def approve_proposal(proposal_id: str) -> bool:
    return approve_proposal_by_id(proposal_id) is not None


def reject_proposal(
    proposal_id: str,
    reason: str = "",
) -> bool:
    return reject_proposal_by_id(
        proposal_id,
        reason=reason,
    ) is not None


# ---------------------------------------------------------------------------
# Emergency stop
# ---------------------------------------------------------------------------


def engine_trigger_emergency_stop(
    reason: str = "",
) -> dict[str, Any]:
    resolved_reason = reason or "manual"
    trigger_emergency_stop(resolved_reason)

    return {
        "status": "emergency_stop_active",
        "reason": resolved_reason,
        "timestamp": _utc_now(),
    }


def engine_clear_emergency_stop() -> dict[str, Any]:
    clear_emergency_stop()

    return {
        "status": "emergency_stop_cleared",
        "timestamp": _utc_now(),
    }


# ---------------------------------------------------------------------------
# Position management
# ---------------------------------------------------------------------------


def engine_close_position(
    position_id: str,
    *,
    exit_reason: str = "MANUAL_PAPER_CLOSE",
) -> dict[str, Any] | None:
    """Close via canonical portfolio path.

    Live DIRECT_SPOT inventory is sold on the broker first. Local
    ``close_position`` remains the paper/learning writer and never places
    exchange orders itself.
    """
    refresh_open_position_price(position_id)
    state = load()
    target = None
    for item in list(state.get("positions", []) or []):
        if (
            str(item.get("position_id") or "") == str(position_id)
            and str(item.get("status", "open")) == "open"
        ):
            target = item
            break
    if target is not None and is_live_reservation(target):
        outcome, cancel_error, fill = cancel_live_spot_order_sync(target)
        if outcome == "filled" and fill is not None:
            from app.portfolio_manager import confirm_live_fill

            confirmed = confirm_live_fill(
                str(target.get("position_id") or position_id),
                fill,
            )
            if confirmed is None:
                return {
                    "status": "blocked",
                    "reason": "live reservation filled but local confirm failed",
                    "position_id": position_id,
                }
            sold_outcome, sell_error, fill = place_live_spot_sell_sync(confirmed)
            if sold_outcome == "failed" or fill is None:
                return {
                    "status": "blocked",
                    "reason": sell_error or "live sell failed after late fill",
                    "position_id": position_id,
                }
            return book_live_exchange_exit(
                position_id,
                fill,
                exit_reason=exit_reason,
                fallback=confirmed,
            )
        if outcome == "blocked":
            return {
                "status": "blocked",
                "reason": cancel_error or "live reservation cancel failed",
                "position_id": position_id,
            }
        return release_live_reservation(
            position_id,
            reason=exit_reason,
        )
    if target is not None and bool(target.get("live", False)):
        sold_outcome, sell_error, fill = place_live_spot_sell_sync(target)
        if sold_outcome == "failed" or fill is None:
            return {
                "status": "blocked",
                "reason": sell_error or "live sell failed",
                "position_id": position_id,
            }
        return book_live_exchange_exit(
            position_id,
            fill,
            exit_reason=exit_reason,
            fallback=target,
        )
    return close_position(position_id, exit_reason=exit_reason)


def engine_update_position(
    position_id: str,
    current_price: float,
) -> dict[str, Any] | None:
    return update_position(
        position_id,
        current_price,
    )


def engine_open_positions() -> list[dict[str, Any]]:
    return get_open_positions()


def engine_trade_events(limit: int = 200) -> list[dict[str, Any]]:
    """Dashboard/API helper voor echte paper fills en actieve exits."""
    return get_trade_events(limit=limit)
