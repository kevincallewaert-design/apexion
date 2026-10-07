from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict




def _install_legacy_aegis_env_aliases() -> None:
    """Make historical AEGIS_* settings work while APEXION_* is canonical.

    APEXION v0.1.6 uses the APEXION_ namespace. Existing user environments
    and old .env files may still contain AEGIS_ keys, so copy those values to
    the new namespace only when the APEXION_ key is not already defined.
    """
    for key, value in list(os.environ.items()):
        if key.upper().startswith("AEGIS_"):
            apexion_key = "APEXION_" + key[6:]
            os.environ.setdefault(apexion_key, value)

    env_path = Path(".env")
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key.upper().startswith("AEGIS_"):
                continue
            apexion_key = "APEXION_" + key[6:]
            if apexion_key in os.environ:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
                value = value[1:-1]
            os.environ[apexion_key] = value
    except OSError:
        # Settings validation remains fail-safe; inability to read an optional
        # legacy .env file must not mutate trading safety.
        pass


_install_legacy_aegis_env_aliases()

PAPER_EXECUTION_MODES = frozenset({"paper", "live_dry_run"})


def is_paper_execution_mode(mode: str) -> bool:
    return str(mode or "paper").strip().lower() in PAPER_EXECUTION_MODES


def is_micro_live_execution_mode(mode: str, live_execution_enabled: bool) -> bool:
    """True when the process is in MICRO_LIVE with broker execution enabled.

    Arm token, credentials and LIVE_RELEASED are separate gates. This helper
    only distinguishes the continuous-management / live-intent envelope from
    PAPER/LIVE_DRY_RUN.
    """
    return (
        str(mode or "").strip().lower() == "micro_live"
        and bool(live_execution_enabled)
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="APEXION_",
        case_sensitive=False,
        extra="ignore",
    )

    mode: str = "paper"
    micro_live_arm: str = ""
    live_max_envelope_usdt: float = Field(default=50.0, gt=0.0, le=50.0)
    live_max_order_usdt: float = Field(default=10.0, gt=0.0, le=10.0)
    live_max_deployed_usdt: float = Field(default=25.0, gt=0.0, le=25.0)
    live_max_positions: int = Field(default=1, ge=1, le=1)
    live_daily_loss_limit_usdt: float = Field(default=5.0, gt=0.0, le=5.0)
    secret_key: str = "dev-change-me"

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    username: str = "admin"
    password: str = "change-me"

    pionex_api_key: str = ""
    pionex_api_secret: str = ""
    pionex_base_url: str = "https://api.pionex.com"
    pionex_public_ws_url: str = "wss://ws.pionex.com/wsPub"
    pionex_private_ws_url: str = "wss://ws.pionex.com/ws"
    market_stream_enabled: bool = True
    market_stream_auto_start: bool = True
    market_stream_private_enabled: bool = False
    market_stream_depth_levels: int = Field(default=10, ge=1, le=100)
    market_stream_stale_seconds: float = Field(default=20.0, gt=0.0, le=300.0)
    market_stream_max_symbols: int = Field(default=15, ge=1, le=15)
    market_stream_reconnect_initial_seconds: float = Field(
        default=1.0,
        gt=0.0,
        le=30.0,
    )
    market_stream_reconnect_max_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
    )

    max_active_bots: int = Field(default=4, ge=1, le=15)
    max_active_executions: int = Field(default=4, ge=1, le=15)
    max_grid_dca_positions: int = Field(default=3, ge=0, le=15)
    reserved_spot_slots: int = Field(default=1, ge=0, le=15)
    max_one_position_per_coin: bool = True

    # APEXION 50-USDT envelope (PAPER and MICRO-LIVE share these hard caps).
    # Dynamic allocation can only lower effective exposure, never bypass hard caps.
    default_investment_usdt: float = Field(default=10.0, gt=0)
    max_per_coin_usdt: float = Field(default=10.0, gt=0)
    max_total_exposure_usdt: float = Field(default=50.0, gt=0)
    reserve_ratio: float = Field(default=0.25, ge=0.0, lt=1.0)

    # PAPER/LIVE_DRY_RUN dynamic allocation. Absolute caps above remain authoritative;
    # ratio caps can only make the effective limit stricter, never looser.
    dynamic_capital_allocation_enabled: bool = True
    max_portfolio_deployed_ratio: float = Field(default=0.50, gt=0.0, le=0.50)
    max_single_position_ratio: float = Field(default=0.20, gt=0.0, le=0.20)
    max_single_grid_ratio: float = Field(default=0.22, gt=0.0, le=0.25)
    min_cash_reserve_ratio: float = Field(default=0.25, ge=0.0, lt=0.95)
    max_asset_class_exposure_ratio: float = Field(default=0.50, gt=0.0, le=0.50)
    max_correlated_exposure_ratio: float = Field(default=0.15, gt=0.0, le=0.30)
    paper_grid_min_economic_order_usdt: float = Field(default=5.0, gt=0.0)
    paper_micro_grid_enabled: bool = True
    paper_micro_grid_max_equity_usdt: float = Field(
        default=100.0,
        gt=0.0,
        le=250.0,
    )
    paper_micro_grid_max_allocation_ratio: float = Field(
        default=0.70,
        gt=0.0,
        le=0.80,
    )

    # Centrale PAPER execution-kosten. Rates zijn fracties: 0.0005 = 0.05%.
    # Zowel BOT-fills als DIRECT SPOT gebruiken uitsluitend deze waarden.
    paper_maker_fee_rate: float = Field(default=0.0005, gt=0.0, lt=0.05)
    paper_taker_fee_rate: float = Field(default=0.0005, gt=0.0, lt=0.05)
    paper_slippage_rate: float = Field(default=0.0003, gt=0.0, lt=0.05)
    paper_maker_slippage_rate: float = Field(default=0.0, ge=0.0, lt=0.05)
    paper_min_grid_net_profit_pct: float = Field(default=0.05, gt=0.0, le=5.0)
    paper_grid_activity_stale_minutes: int = Field(default=30, ge=5, le=1440)
    paper_grid_activity_min_movement_pct: float = Field(
        default=0.50,
        gt=0.0,
        le=25.0,
    )
    paper_grid_activity_min_improvement_pct: float = Field(
        default=0.01,
        gt=0.0,
        le=5.0,
    )
    paper_grid_rebuild_payback_hours: float = Field(
        default=4.0,
        gt=0.0,
        le=48.0,
    )
    paper_grid_rebuild_cooldown_minutes: int = Field(
        default=60,
        ge=5,
        le=1440,
    )
    paper_grid_inventory_skew_enabled: bool = True
    paper_grid_inventory_target_base_ratio: float = Field(
        default=0.50,
        gt=0.0,
        lt=1.0,
    )
    paper_grid_inventory_max_base_ratio: float = Field(
        default=0.78,
        gt=0.0,
        lt=1.0,
    )
    paper_grid_filled_order_delay_seconds: int = Field(
        default=45,
        ge=0,
        le=3600,
    )
    paper_grid_stress_buy_cooldown_seconds: int = Field(
        default=120,
        ge=0,
        le=7200,
    )
    paper_grid_order_refresh_tolerance_pct: float = Field(
        default=0.12,
        ge=0.0,
        le=5.0,
    )
    paper_grid_adverse_selection_enabled: bool = True
    paper_grid_adverse_book_imbalance: float = Field(
        default=0.35,
        gt=0.0,
        lt=1.0,
    )
    paper_grid_adverse_trade_imbalance: float = Field(
        default=0.45,
        gt=0.0,
        lt=1.0,
    )
    paper_grid_max_market_spread_pct: float = Field(
        default=0.40,
        gt=0.0,
        le=10.0,
    )
    paper_execution_interval_seconds: int = Field(default=30, ge=2, le=60)
    paper_execution_interval_floor_seconds: int = Field(default=30, ge=2, le=60)
    paper_execution_interval_ceiling_seconds: int = Field(default=60, ge=15, le=120)
    paper_execution_misfire_grace_seconds: int = Field(default=5, ge=1, le=30)
    scheduler_misfire_grace_seconds: int = Field(default=120, ge=30, le=900)
    execution_timing_history_limit: int = Field(default=200, ge=20, le=1000)
    paper_execution_cache_seconds: float = Field(default=2.0, ge=0.25, le=15.0)
    paper_execution_candle_interval: str = "1m"
    paper_grid_suitability_min_score: float = Field(
        default=58.0,
        ge=0.0,
        le=100.0,
    )
    paper_grid_max_entry_breakout_risk: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
    )
    paper_grid_block_risk_off_bearish_entry: bool = True
    paper_grid_rotation_improvement_gate_pct: float = Field(
        default=0.10,
        gt=0.0,
        le=10.0,
    )
    paper_spot_trailing_activation_pct: float = Field(
        default=1.0,
        gt=0.0,
        le=25.0,
    )
    paper_spot_trailing_distance_pct: float = Field(
        default=0.9,
        gt=0.0,
        le=25.0,
    )
    paper_spot_min_rotation_improvement_pct: float = Field(
        default=0.25,
        gt=0.0,
        le=25.0,
    )
    paper_spot_min_hold_minutes: int = Field(
        default=30,
        ge=0,
        le=10080,
    )
    paper_spot_edge_decay_score: float = Field(
        default=35.0,
        ge=0.0,
        le=100.0,
    )

    # Market Theory + Historical Intelligence. Deze instellingen sturen
    # uitsluitend analysegewicht en publieke candlevensters; nooit safety.
    intelligence_entry_timeframe: str = "15m"
    intelligence_setup_timeframe: str = "1h"
    intelligence_regime_timeframe: str = "4h"
    intelligence_candle_limit: int = Field(default=180, ge=60, le=500)
    intelligence_historical_min_samples: int = Field(default=12, ge=5, le=100)
    intelligence_historical_max_samples: int = Field(default=40, ge=5, le=100)
    intelligence_historical_forward_bars: int = Field(default=8, ge=2, le=30)
    intelligence_experience_min_samples: int = Field(default=12, ge=5, le=100)
    intelligence_min_confluence_score: float = Field(
        # APEXION: opportunity threshold, not a safety veto. Activity control
        # may lower it within a bounded floor when the bot remains idle.
        default=0.42,
        ge=0.25,
        le=0.95,
    )
    intelligence_min_net_edge_pct: float = Field(
        default=0.15,
        gt=0.0,
        le=10.0,
    )
    # GEX-led spot routing.  A missing/stale options snapshot may use the
    # ordinary price engine as an explicitly labelled fallback; valid GEX is
    # authoritative for the selected spot strategy.
    gex_strategy_router_enabled: bool = True
    gex_context_ttl_seconds: float = Field(default=1800.0, ge=60.0, le=7200.0)
    gex_block_near_wall_entry: bool = False
    gex_wall_entry_block_pct: float = Field(default=0.30, ge=0.0, le=2.0)
    gex_block_gamma_transition_entry: bool = False
    market_metadata_cache_seconds: int = Field(default=300, ge=30, le=3600)
    market_candle_cache_seconds: int = Field(default=90, ge=10, le=1800)
    market_rate_limit_min_interval_ms: int = Field(default=100, ge=0, le=1000)
    market_rate_limit_max_retries: int = Field(default=2, ge=0, le=5)
    market_rate_limit_backoff_seconds: float = Field(default=0.20, ge=0.0, le=10.0)
    market_rate_limit_cooldown_seconds: int = Field(default=120, ge=10, le=3600)

    # Comma-separated strategy names that may not open NEW positions, e.g.
    # "mean_reversion,trend_follow". Positions already open are still managed.
    disabled_strategies: str = ""
    # Experiment switch: when true only take-profit and protective-stop close a
    # position (no trailing, rotation, edge-decay or AI hard exits).
    # Market-wide gate: no NEW positions while BTC's last completed daily close is
    # below its N-day average (0 = off). Fails closed if BTC data is unavailable.
    regime_filter_sma_days: int = Field(default=0, ge=0, le=200)
    simple_exits: bool = False
    # Trend-hold sleeve (PAPER only, own virtual capital): hold the listed coins
    # while BTC is above its SMA (+/- band), otherwise cash. Off by default.
    trend_hold_enabled: bool = False
    trend_hold_sma_days: int = Field(default=100, ge=20, le=200)
    trend_hold_band_pct: float = Field(default=2.0, ge=0.0, le=10.0)
    trend_hold_coins: str = "BTC,ETH"
    trend_hold_capital_usdt: float = Field(default=50.0, gt=0.0, le=100000.0)
    # Net take-profit / stop distances (%) used only while simple_exits is on;
    # 0 keeps the per-position values. Minimum effective value is 0.5.
    simple_take_profit_pct: float = Field(default=0.0, ge=0.0, le=20.0)
    simple_stop_loss_pct: float = Field(default=0.0, ge=0.0, le=20.0)
    # Optional comma-separated coin filters for the market scan (empty = no filter).
    scan_allowlist: str = ""
    scan_excluded_coins: str = ""
    live_execution_enabled: bool = False
    approval_required: bool = True
    auto_execute_paper: bool = True
    # Safe PAPER rotation runs autonomously when all FASE 4 gates pass,
    # independent of approval_required (LIVE remains blocked).
    auto_execute_paper_rotation: bool = True
    rotation_cost_to_quality_multiplier: float = Field(
        default=2.5,
        gt=0.0,
        le=20.0,
        description=(
            "Maps estimated round-trip rotation cost (%) into quality points. "
            "Default 2.5 → 1% all-in cost ≈ 2.5 quality points."
        ),
    )
    rotation_risk_edge_cover_ratio: float = Field(
        default=0.35,
        gt=0.0,
        le=1.0,
        description=(
            "Required net rotation edge per unit of additional risk penalty. "
            "Default 0.35 → +8 risk points need ≥ 8/0.35 net edge."
        ),
    )
    rotation_min_net_edge_points: float = Field(
        default=2.0,
        ge=0.0,
        le=50.0,
        description="Minimum net rotation edge after cost/risk penalties.",
    )

    scheduler_auto_start: bool = False
    scheduler_interval_minutes: int = Field(default=15, ge=1, le=1440)
    live_price_refresh_seconds: int = Field(default=5, ge=2, le=300)

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"paper", "live_dry_run", "micro_live"}:
            raise ValueError("APEXION_MODE moet paper, live_dry_run of micro_live zijn.")
        return value

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError(
                "APEXION_HOST moet een lokaal loopback-adres zijn."
            )
        return value

    @field_validator(
        "intelligence_entry_timeframe",
        "intelligence_setup_timeframe",
        "intelligence_regime_timeframe",
        "paper_execution_candle_interval",
    )
    @classmethod
    def validate_intelligence_timeframe(cls, value: str) -> str:
        value = value.strip()
        allowed = {"1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"}
        if value.lower() not in allowed:
            raise ValueError("ongeldig intelligence-timeframe")
        return value.lower()

    @property
    def paper_execution_mode(self) -> bool:
        """True when execution must remain fully virtual and broker-write-free."""
        return is_paper_execution_mode(self.mode)

    @property
    def paper_mode(self) -> bool:
        """Backward-compatible alias for paper-only execution semantics."""
        return self.paper_execution_mode

    @property
    def effective_approval_required(self) -> bool:
        """Manual approval defaults to on at the MICRO_LIVE boundary, but the
        operator can explicitly opt out via APEXION_APPROVAL_REQUIRED=false
        in .env once the live pipeline has been verified with a human in the
        loop. Outside MICRO_LIVE this is always False regardless of the flag."""
        if str(self.mode).lower() != "micro_live":
            return False
        return bool(self.approval_required)

    @property
    def authentication_configured(self) -> bool:
        weak_usernames = {
            "",
            "admin",
            "change-me",
            "replace-with-local-username",
        }
        weak_passwords = {
            "",
            "admin",
            "change-me",
            "replace-with-a-strong-local-password",
        }
        return (
            self.username.strip().lower() not in weak_usernames
            and self.password.strip().lower() not in weak_passwords
            and len(self.password) >= 16
        )

    def live_block_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.mode != "micro_live":
            reasons.append("APEXION_MODE is not micro_live")
        if not self.live_execution_enabled:
            reasons.append("APEXION_LIVE_EXECUTION_ENABLED is false")
        if self.micro_live_arm.strip().upper() != "LIVE":
            reasons.append("APEXION_MICRO_LIVE_ARM is not LIVE")
        if not self.pionex_api_key or not self.pionex_api_secret:
            reasons.append("Pionex API credentials missing")
        if not self.authentication_configured:
            reasons.append("Dashboard password is insecure")
        if self.max_grid_dca_positions + self.reserved_spot_slots > self.max_active_bots:
            reasons.append("Portfolio slot configuration is invalid")
        if self.gex_strategy_router_enabled:
            try:
                from app.gex_overlay import for_coin

                gex = for_coin(
                    "BTC",
                    context_ttl_seconds=self.gex_context_ttl_seconds,
                )
                if not gex.valid:
                    reasons.append(
                        "Fresh GEX-context vereist voor live spot-routing"
                    )
            except Exception:
                reasons.append("GEX live-readiness check failed")
        return reasons


settings = Settings()
