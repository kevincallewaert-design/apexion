"""Centrale, niet-nul PAPER fees en slippage voor alle virtuele fills.

Deze module kent geen brokerclient en kan dus nooit een echte order plaatsen.
Alle bedragen zijn in USDT; fee/slippage-rates zijn fracties (0.0005 = 0.05%).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.config import settings


@dataclass(frozen=True)
class PaperCostModel:
    maker_fee_rate: float
    taker_fee_rate: float
    slippage_rate: float
    min_grid_net_profit_pct: float
    maker_slippage_rate: float = 0.0

    def fee_rate(self, liquidity: str) -> float:
        return (
            self.maker_fee_rate
            if str(liquidity).strip().lower() == "maker"
            else self.taker_fee_rate
        )

    def slippage_for(self, liquidity: str) -> float:
        """Resting maker limits do not inherit market-order slippage."""
        return (
            self.maker_slippage_rate
            if str(liquidity).strip().lower() == "maker"
            else self.slippage_rate
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "maker_fee_rate": self.maker_fee_rate,
            "taker_fee_rate": self.taker_fee_rate,
            "slippage_rate": self.slippage_rate,
            "maker_slippage_rate": self.maker_slippage_rate,
            "min_grid_net_profit_pct": self.min_grid_net_profit_pct,
        }


def _positive_setting(name: str, default: float) -> float:
    try:
        value = float(getattr(settings, name, default))
    except (TypeError, ValueError):
        value = default
    return value if value > 0.0 else default


def _nonnegative_setting(name: str, default: float) -> float:
    try:
        value = float(getattr(settings, name, default))
    except (TypeError, ValueError):
        value = default
    return value if value >= 0.0 else default


def get_paper_cost_model() -> PaperCostModel:
    """Lees één centrale kostenconfiguratie met veilige niet-nul defaults."""
    return PaperCostModel(
        maker_fee_rate=_positive_setting("paper_maker_fee_rate", 0.0005),
        taker_fee_rate=_positive_setting("paper_taker_fee_rate", 0.0005),
        slippage_rate=_positive_setting("paper_slippage_rate", 0.0003),
        min_grid_net_profit_pct=_positive_setting(
            "paper_min_grid_net_profit_pct",
            0.05,
        ),
        maker_slippage_rate=_nonnegative_setting(
            "paper_maker_slippage_rate",
            0.0,
        ),
    )


def simulate_paper_fill(
    *,
    side: str,
    reference_price: float,
    quantity: float,
    liquidity: str,
    cost_model: PaperCostModel | None = None,
) -> dict[str, Any]:
    """Bereken een PAPER fill inclusief adversarial slippage en fee."""
    side = str(side).strip().lower()
    if side not in {"buy", "sell"}:
        raise ValueError("side moet buy of sell zijn")
    if reference_price <= 0 or quantity <= 0:
        raise ValueError("reference_price en quantity moeten positief zijn")

    model = cost_model or get_paper_cost_model()
    direction = 1.0 if side == "buy" else -1.0
    slippage_rate = model.slippage_for(liquidity)
    execution_price = reference_price * (1.0 + direction * slippage_rate)
    gross_value = execution_price * quantity
    fee = gross_value * model.fee_rate(liquidity)
    slippage = abs(execution_price - reference_price) * quantity
    net_value = -(gross_value + fee) if side == "buy" else gross_value - fee

    return {
        "side": side,
        "reference_price": round(reference_price, 12),
        "price": round(execution_price, 12),
        "quantity": round(quantity, 16),
        "gross_value": round(gross_value, 12),
        "fee": round(fee, 12),
        "slippage": round(slippage, 12),
        "net_value": round(net_value, 12),
        "liquidity": str(liquidity).strip().lower(),
        "fee_rate": model.fee_rate(liquidity),
        "slippage_rate": slippage_rate,
    }


def simulate_paper_buy_with_budget(
    *,
    reference_price: float,
    total_budget: float,
    liquidity: str,
    cost_model: PaperCostModel | None = None,
) -> dict[str, Any]:
    """Koop waarbij gross + fee exact binnen het gereserveerde budget blijft."""
    if total_budget <= 0:
        raise ValueError("total_budget moet positief zijn")
    model = cost_model or get_paper_cost_model()
    fee_rate = model.fee_rate(liquidity)
    gross_budget = total_budget / (1.0 + fee_rate)
    execution_price = reference_price * (1.0 + model.slippage_for(liquidity))
    quantity = gross_budget / execution_price
    result = simulate_paper_fill(
        side="buy",
        reference_price=reference_price,
        quantity=quantity,
        liquidity=liquidity,
        cost_model=model,
    )
    # Vermijd cumulatieve floatafronding in reserveboekhouding.
    result["net_value"] = round(-total_budget, 12)
    return result


def round_trip_break_even_move(
    *,
    liquidity: str = "maker",
    cost_model: PaperCostModel | None = None,
) -> float:
    """Minimale referentieprijsbeweging (fractie) voor netto break-even."""
    model = cost_model or get_paper_cost_model()
    buy_fee = model.fee_rate(liquidity)
    sell_fee = model.fee_rate(liquidity)
    slippage_rate = model.slippage_for(liquidity)
    denominator = (1.0 - slippage_rate) * (1.0 - sell_fee)
    numerator = (1.0 + slippage_rate) * (1.0 + buy_fee)
    return max(0.0, numerator / denominator - 1.0)


def initialize_spot_position(
    position: dict[str, Any],
    reference_price: float,
) -> dict[str, Any]:
    """Boek een DIRECT SPOT market-entry met taker fee en slippage."""
    fill = simulate_paper_buy_with_budget(
        reference_price=reference_price,
        total_budget=float(position.get("size_usdt", 0.0) or 0.0),
        liquidity="taker",
    )
    position.update(
        {
            "entry_reference_price": round(reference_price, 12),
            "entry_price": fill["price"],
            "current_price": round(reference_price, 12),
            "high_watermark": max(reference_price, fill["price"]),
            "low_watermark": min(reference_price, fill["price"]),
            "quantity": fill["quantity"],
            "entry_gross_value_usdt": fill["gross_value"],
            "entry_fee_usdt": fill["fee"],
            "entry_slippage_usdt": fill["slippage"],
            "entry_net_value_usdt": fill["net_value"],
            "entry_total_cost_usdt": round(abs(fill["net_value"]), 12),
            "estimated_entry_fee_usdt": fill["fee"],
            "fees_paid": fill["fee"],
            "slippage_paid": fill["slippage"],
            "gross_unrealized_pnl": 0.0,
            "estimated_exit_fee_usdt": 0.0,
            "estimated_exit_slippage_usdt": 0.0,
            "net_unrealized_pnl": 0.0,
        }
    )
    # Toon vanaf de entry al de netto liquidatiewaarde inclusief exitkosten.
    mark_spot_to_market(position, reference_price)
    return fill


def initialize_dca_position(
    position: dict[str, Any],
    reference_price: float,
) -> dict[str, Any]:
    """Open only the first configured DCA tranche and persist staged state."""
    params = dict(position.get("strategy_params") or {})
    max_rounds = max(1, int(params.get("max_rounds", 1) or 1))
    configured = float(params.get("dca_amount_per_order", 0.0) or 0.0)
    planned_budget = max(0.0, float(position.get("size_usdt", 0.0) or 0.0))
    first_budget = configured if configured > 0.0 else planned_budget / max_rounds
    first_budget = min(planned_budget, max(0.01, first_budget))

    fill = simulate_paper_buy_with_budget(
        reference_price=reference_price,
        total_budget=first_budget,
        liquidity="taker",
    )
    quantity = float(fill.get("quantity", 0.0) or 0.0)
    total_cost = abs(float(fill.get("net_value", 0.0) or 0.0))
    if quantity <= 0.0 or total_cost <= 0.0:
        raise ValueError("DCA initial fill heeft geen betrouwbare cost basis")

    now = datetime.now(timezone.utc).isoformat()
    position.update(
        {
            "entry_reference_price": round(reference_price, 12),
            "entry_price": round(total_cost / quantity, 12),
            "current_price": round(reference_price, 12),
            "high_watermark": max(reference_price, float(fill["price"])),
            "low_watermark": min(reference_price, float(fill["price"])),
            "quantity": quantity,
            "entry_gross_value_usdt": float(fill["gross_value"]),
            "entry_fee_usdt": float(fill["fee"]),
            "entry_slippage_usdt": float(fill["slippage"]),
            "entry_net_value_usdt": -total_cost,
            "entry_total_cost_usdt": total_cost,
            "estimated_entry_fee_usdt": float(fill["fee"]),
            "fees_paid": float(fill["fee"]),
            "slippage_paid": float(fill["slippage"]),
            "gross_unrealized_pnl": 0.0,
            "estimated_exit_fee_usdt": 0.0,
            "estimated_exit_slippage_usdt": 0.0,
            "net_unrealized_pnl": 0.0,
            "dca_cost_basis_status": "trusted",
            "dca_cost_basis_version": 2,
            "strategy_state": {
                **dict(position.get("strategy_state") or {}),
                "initialized": True,
                "rounds": 1,
                "last_round_at": now,
                "next_round_at": None,
                "planned_budget_usdt": round(planned_budget, 8),
                "spent_budget_usdt": round(total_cost, 8),
            },
        }
    )
    mark_spot_to_market(position, reference_price)
    return fill


def initialize_flywheel_position(
    position: dict[str, Any],
    reference_price: float,
) -> dict[str, Any]:
    """Arm a Flywheel budget without an immediate market buy.

    The budget remains reserved by ``size_usdt`` while actual inventory is
    created only after a configured dip from the anchor.
    """
    now = datetime.now(timezone.utc).isoformat()
    params = dict(position.get("strategy_params") or {})
    position.update(
        {
            "entry_reference_price": round(reference_price, 12),
            "session_entry_price": round(reference_price, 12),
            "entry_price": round(reference_price, 12),
            "current_price": round(reference_price, 12),
            "high_watermark": round(reference_price, 12),
            "low_watermark": round(reference_price, 12),
            "quantity": 0.0,
            "entry_gross_value_usdt": 0.0,
            "entry_fee_usdt": 0.0,
            "entry_slippage_usdt": 0.0,
            "entry_net_value_usdt": 0.0,
            "entry_total_cost_usdt": 0.0,
            "estimated_entry_fee_usdt": 0.0,
            "fees_paid": 0.0,
            "slippage_paid": 0.0,
            "unrealized_pnl": 0.0,
            "net_unrealized_pnl": 0.0,
            "strategy_realized_pnl": 0.0,
            "strategy_state": {
                **dict(position.get("strategy_state") or {}),
                "initialized": True,
                "anchor": round(reference_price, 12),
                "holding": {},
                "cycles": 0,
                "max_cycles": max(1, int(params.get("max_cycles", 1) or 1)),
                "total_net_pnl_usdt": 0.0,
                "armed_at": now,
                "last_action_at": now,
                "paused_for_downtrend": False,
                "complete": False,
            },
        }
    )
    return dict(position.get("strategy_state") or {})


def apply_dca_buy(
    position: dict[str, Any],
    reference_price: float,
    total_budget: float,
) -> dict[str, Any]:
    """Voeg exact één PAPER DCA-buy toe aan quantity en volledige cost basis.

    Slippage zit al in de execution price en daarmee in ``gross_value``. De
    afzonderlijke slippagekolom is informatief en wordt niet nogmaals bij de
    cost basis opgeteld.
    """
    previous_quantity = float(position.get("quantity", 0.0) or 0.0)
    previous_cost = float(position.get("entry_total_cost_usdt", 0.0) or 0.0)
    if (
        previous_quantity <= 0.0
        or previous_cost <= 0.0
        or str(position.get("dca_cost_basis_status", "")) != "trusted"
    ):
        position["dca_cost_basis_status"] = "untrusted"
        raise ValueError("DCA add geblokkeerd: bestaande cost basis is niet betrouwbaar")

    fill = simulate_paper_buy_with_budget(
        reference_price=reference_price,
        total_budget=total_budget,
        liquidity="taker",
    )
    quantity = previous_quantity + float(fill["quantity"])
    total_cost = previous_cost + abs(float(fill["net_value"]))
    gross = float(position.get("entry_gross_value_usdt", 0.0) or 0.0) + float(
        fill["gross_value"]
    )
    entry_fee = float(position.get("entry_fee_usdt", 0.0) or 0.0) + float(
        fill["fee"]
    )
    entry_slippage = float(
        position.get("entry_slippage_usdt", 0.0) or 0.0
    ) + float(fill["slippage"])

    position.update(
        {
            "quantity": round(quantity, 16),
            "entry_price": round(total_cost / quantity, 12),
            "entry_gross_value_usdt": round(gross, 12),
            "entry_fee_usdt": round(entry_fee, 12),
            "entry_slippage_usdt": round(entry_slippage, 12),
            "entry_net_value_usdt": round(-total_cost, 12),
            "entry_total_cost_usdt": round(total_cost, 12),
            "estimated_entry_fee_usdt": round(entry_fee, 12),
            "fees_paid": round(
                float(position.get("fees_paid", 0.0) or 0.0) + float(fill["fee"]),
                12,
            ),
            "slippage_paid": round(
                float(position.get("slippage_paid", 0.0) or 0.0)
                + float(fill["slippage"]),
                12,
            ),
            "dca_cost_basis_status": "trusted",
            "dca_cost_basis_version": 2,
        }
    )
    mark_spot_to_market(position, reference_price)
    return fill


def mark_spot_to_market(
    position: dict[str, Any],
    reference_price: float,
) -> dict[str, Any]:
    """Bereken DIRECT SPOT unrealized PnL alsof nu netto wordt verkocht."""
    quantity = float(position.get("quantity", 0.0) or 0.0)
    if quantity <= 0 or reference_price <= 0:
        return {}
    exit_fill = simulate_paper_fill(
        side="sell",
        reference_price=reference_price,
        quantity=quantity,
        liquidity="taker",
    )
    entry_gross = float(position.get("entry_gross_value_usdt", 0.0) or 0.0)
    if entry_gross <= 0:
        entry_gross = float(position.get("entry_price", 0.0) or 0.0) * quantity
    entry_cost = float(position.get("entry_total_cost_usdt", 0.0) or 0.0)
    if entry_cost <= 0:
        entry_cost = entry_gross + float(position.get("entry_fee_usdt", 0.0) or 0.0)

    gross_unrealized = exit_fill["gross_value"] - entry_gross
    net_unrealized = exit_fill["net_value"] - entry_cost
    position.update(
        {
            "current_price": round(reference_price, 12),
            "gross_unrealized_pnl": round(gross_unrealized, 12),
            "estimated_exit_fee_usdt": exit_fill["fee"],
            "estimated_exit_slippage_usdt": exit_fill["slippage"],
            "net_unrealized_pnl": round(net_unrealized, 12),
            "unrealized_pnl": round(net_unrealized, 12),
        }
    )
    return exit_fill


def close_spot_position(
    position: dict[str, Any],
    reference_price: float,
) -> dict[str, Any]:
    """Boek de definitieve DIRECT SPOT-exit en netto realized PnL."""
    exit_fill = mark_spot_to_market(position, reference_price)
    if not exit_fill:
        return {}
    entry_cost = float(position.get("entry_total_cost_usdt", 0.0) or 0.0)
    realized = exit_fill["net_value"] - entry_cost
    position.update(
        {
            "exit_reference_price": round(reference_price, 12),
            "exit_price": exit_fill["price"],
            "exit_gross_value_usdt": exit_fill["gross_value"],
            "exit_fee_usdt": exit_fill["fee"],
            "exit_slippage_usdt": exit_fill["slippage"],
            "exit_net_value_usdt": exit_fill["net_value"],
            "fees_paid": round(
                float(position.get("entry_fee_usdt", 0.0) or 0.0)
                + exit_fill["fee"],
                12,
            ),
            "slippage_paid": round(
                float(position.get("entry_slippage_usdt", 0.0) or 0.0)
                + exit_fill["slippage"],
                12,
            ),
            "realized_pnl": round(realized, 12),
            "unrealized_pnl": 0.0,
            "net_unrealized_pnl": 0.0,
        }
    )
    return exit_fill
