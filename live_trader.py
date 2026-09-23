#!/usr/bin/env python3
"""
AlgoFlow Live Paper Trading Runner
==================================
Runs the main trader on Alpaca paper trading.  DEFAULT strategy (as of
PR #37) is the ScalpSet — three modular intraday scalp models
(ICT IFVG / Box Theory / VolProfile+FIB) arbitrated per tick — with the
original mean-reversion strategy importable for rollback via
``MAIN_STRATEGY=mean_reversion``.  Waits for market open, trades
throughout the day, holds positions overnight (GTC-protected) at close.

Start: python3 live_trader.py
Logs: /home/team/shared/engine/logs/trades_YYYYMMDD.log
"""
import asyncio
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project imports ────────────────────────────────────────────────
import src.strategies  # registers strategies
from src.data.yfinance_provider import YFinanceProvider
from src.execution.alpaca_broker import AlpacaBroker, account_equity, is_order_alive
from src.execution.broker import Order, OrderSide, OrderType
from src.execution.position_manager import PositionManager
from src.execution.session_state import load_start_equity, save_start_equity
from src.execution.exit_structure import (
    ExitPlan,
    format_exit_plan,
    plan_exit_structure,
)
from src.execution.verified_close import close_position_verified
from src.strategies.base import SignalType, StrategyConfig
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.indicators import sma
from src.strategies.scalp.types import (
    CandleFrame,
    Direction,
    EntryType,
    LiquidityMap,
    ScalpContext,
    ScalpSignal,
    SessionWindow,
)
from src.strategies.scalp.indicators import session_extremes
from src.strategies.scalp import scalp_box_theory, scalp_ict_ifvg, scalp_volprofile_fib

# ── Configuration ──────────────────────────────────────────────────
SYMBOLS = ["NVDA", "META", "QQQ", "TSLA", "COIN", "AVGO"]
CHECK_INTERVAL = 60  # seconds between polls
# ── Market-gate hardening (2026-09-15, PR #35) ────────────────────
# The gate must never wait silently for hours (observed 2026-09-14: a
# whole-container freeze hid the traders' stall for 9h because a
# confirmed-closed clock answer is the ONLY valid reason to defer —
# anything else must retry loudly and FAIL OPEN once local ET passes
# 09:30 + buffer).  Every knob below is env-overridable for drills/tests.
MARKET_GATE_RETRY_SECONDS = float(          # unknown/error backoff    (15-30s)
    os.environ.get("MARKET_GATE_RETRY_SECONDS", "20"))
MARKET_GATE_HEARTBEAT_SECONDS = float(      # confirmed-closed heartbeat
    os.environ.get("MARKET_GATE_HEARTBEAT_SECONDS", "240"))
MARKET_GATE_FAIL_OPEN_BUFFER_SECONDS = float(  # local-ET fail-open buffer
    os.environ.get("MARKET_GATE_FAIL_OPEN_BUFFER_SECONDS", "600"))
MARKET_GATE_OVERDUE_LOG_SECONDS = float(    # loud overdue-open cadence
    os.environ.get("MARKET_GATE_OVERDUE_LOG_SECONDS", "60"))
MARKET_LOOP_UNKNOWN_RETRY_SECONDS = float(  # intraday unknown-clock retry
    os.environ.get("MARKET_LOOP_UNKNOWN_RETRY_SECONDS", "20"))
MARKET_TICK_HEARTBEAT_SECONDS = float(      # intraday tick-loop liveness
    os.environ.get("MARKET_TICK_HEARTBEAT_SECONDS", "240"))
# NOTE: Analysis shows low-confidence trades (0.4-0.6 bucket) average +$33.46 —
# actually MORE profitable than high-confidence trades. More signals = more opportunities.
CONFIDENCE_THRESHOLD = 0.3
MAX_POSITIONS = 6
POSITION_SIZE_PCT = 0.15  # 15% of equity per position

# Market close in UTC (4 PM ET = 20:00 UTC)
MARKET_CLOSE_UTC_HOUR = 20
MARKET_CLOSE_UTC_MINUTE = 0
MANDATORY_CLOSE_MINUTES = 15  # Liquidate all positions 15 min before close
MAX_HOLD_MINUTES = 30          # Max time to hold a position — recycle capital

STRATEGY_CONFIG = StrategyConfig(
    entry_threshold=0.5,       # z-score to enter (lower = more trades)
    exit_threshold=0.1,        # z-score to exit
    stop_loss_pct=0.03,        # 3% stop-loss
    take_profit_pct=0.03,      # 3% take-profit
    max_position_pct=POSITION_SIZE_PCT,
    extra={"lookback": 20, "std_dev_multiplier": 2.0},
)

# ── Regime gate (Recommendation: stop bleeding into falling markets) ──
# The main trader is pure mean reversion — it buys oversold dips, which is
# exactly the wrong thing to do in a confirmed downtrend (falling knives).
# This gate mirrors the turbo trader's `_regime_gate_allows_long`: it SKIPS
# the mean-reversion LONG when price is below the 10-bar MA AND RSI(14) < 40
# (a confirmed downtrend).  Flippable for paper-trading A/B comparison.
ENABLE_REGIME_GATE = True   # False → restore unconditional dip-buying
REGIME_MA_PERIOD = 10       # short MA that defines the trend reference
REGIME_RSI_PERIOD = 14      # RSI period used by the weakness filter
REGIME_RSI_THRESHOLD = 40.0 # RSI below this = oversold/weak → block the long

# ── Broker-side protective stops ────────────────────────────────────
# Every held AND every newly-opened main position also gets a GTC
# stop-loss order at the broker (Alpaca holds it even if every process on
# this machine dies — the whole point: positions must be protected with
# zero processes running).  The in-process 3% risk stop fires first while
# the process is alive; the broker stop is the hard backstop at
# entry − PROTECTIVE_STOP_PCT (6% default, mirroring the turbo trader's
# protective-stop band for the mean-reversion profile).  Configurable via
# MAIN_PROTECTIVE_STOP_PCT (e.g. 0.03 = 3%).
PROTECTIVE_STOP_PCT = float(os.environ.get("MAIN_PROTECTIVE_STOP_PCT", "0.06"))
# Stop-placement retry: submitting the stop while the entry order is still
# open makes Alpaca reject it as a "potential wash trade" (opposite-side
# market/stop order exists), which previously left positions running with
# no broker-side protection.  Same policy as the turbo trader.
STOP_PLACEMENT_MAX_ATTEMPTS = 4
STOP_PLACEMENT_INITIAL_DELAY = 2.0

# ── Scalp strategy set (PR #37) ──────────────────────────────────────
# The main trader's DEFAULT strategy is now the ScalpSet — three modular
# scalp models (ICT IFVG / Box Theory / VolProfile+FIB) evaluated per tick
# and arbitrated into ONE signal per symbol.  The old mean-reversion
# strategy stays importable for instant rollback via MAIN_STRATEGY.
MAIN_STRATEGY = os.environ.get("MAIN_STRATEGY", "scalp").strip().lower()
# Per-module on/off switches (env booleans; all on by default).
SCALP_MODULE_IFVG = os.environ.get("SCALP_MODULE_IFVG", "true").lower() != "false"
SCALP_MODULE_BOX = os.environ.get("SCALP_MODULE_BOX", "true").lower() != "false"
SCALP_MODULE_VOLFIB = os.environ.get("SCALP_MODULE_VOLFIB", "true").lower() != "false"
# Signal arbitration: "best_rr" (highest R:R; ties broken in module order
# IFVG > Box > VolFib) or "first" (module order wins regardless of R:R).
SCALP_ARBITRATION = os.environ.get("SCALP_ARBITRATION", "best_rr").strip().lower()
# Per-symbol cooldown after a position closes (measured in 1-minute bars /
# ticks): no NEW entry for the symbol until the window elapses.
SCALP_COOLDOWN_BARS = int(os.environ.get("SCALP_COOLDOWN_BARS", "5"))
# ── Fill anchoring + churn cap (post-go-live fix, 2026-09-16) ────────
# MARKET fills land far from the signal-time reference on fast tape.  Live
# COIN case (2026-09-16): signal "LONG entry=167.96 SL=167.01", filled at
# 162.58 — so the strategy SL sat ABOVE the fill.  Alpaca rejected every
# GTC stop with 42210000 "stop price must be less than current price", the
# retry loop burned 4 attempts, and the in-process SL then treated the
# stale level as breached and closed the trade (then the next signal
# repeated the whole loop).  Strategy SL/TP are now re-anchored to the
# CONFIRMED FILL before the broker stop is submitted — see
# ``anchor_scalp_levels`` for the exact rule.
SCALP_ANCHOR_LEVELS = os.environ.get("SCALP_ANCHOR_LEVELS", "true").lower() != "false"
# Minimum distance the anchored SL/TP must keep from the fill: a fraction
# of the fill, floored at an absolute amount.  Without it a level could sit
# on top of the market price (instantly-triggered / rejected).
SCALP_STOP_MIN_DISTANCE_PCT = float(
    os.environ.get("SCALP_STOP_MIN_DISTANCE_PCT", "0.0005"))   # 0.05% of fill
SCALP_STOP_MIN_DISTANCE_ABS = float(
    os.environ.get("SCALP_STOP_MIN_DISTANCE_ABS", "0.01"))     # or 1 cent
# Cadence (seconds) of the loud "position has NO broker stop" warning.
SCALP_NO_STOP_WARN_SECONDS = float(os.environ.get("SCALP_NO_STOP_WARN_SECONDS", "60"))
# Re-entry churn cap: max ScalpSet ENTRY ORDERS submitted per symbol per
# trading day (<= 0 disables).  Prevents the COIN-style loop where a setup
# re-emits every cooldown and re-enters after every stop-out.
SCALP_MAX_ENTRIES_PER_SYMBOL_PER_SESSION = int(
    os.environ.get("SCALP_MAX_ENTRIES_PER_SYMBOL_PER_SESSION", "3"))
# SMT divergence pair for the IFVG module (best-effort on US equities).
SCALP_PAIR_SYMBOL = os.environ.get("SCALP_PAIR_SYMBOL", "QQQ").upper()
# Frame-cache TTLs (seconds) — higher timeframes are lazy-fetched so we
# don't hammer the data provider every tick.  1m is always fresh.
SCALP_CACHE_1M_SECONDS = float(os.environ.get("SCALP_CACHE_1M_SECONDS", "0"))
SCALP_CACHE_5M_SECONDS = float(os.environ.get("SCALP_CACHE_5M_SECONDS", "15"))
SCALP_CACHE_15M_SECONDS = float(os.environ.get("SCALP_CACHE_15M_SECONDS", "120"))
SCALP_CACHE_30M_SECONDS = float(os.environ.get("SCALP_CACHE_30M_SECONDS", "120"))
SCALP_CACHE_HTF_SECONDS = float(os.environ.get("SCALP_CACHE_HTF_SECONDS", "300"))
# ── Short-side knobs (the lessons from turbo PR #29, ported to main) ──
# Whole-share flooring for short orders (Alpaca rejects fractional short
# sales outright), BP-capped sizing, broker-shortability gate, and a
# per-symbol rejection backoff (N same-session rejections disable that
# symbol's shorts for the rest of the session; longs are unaffected).
MAIN_SHORT_WHOLE_SHARES = os.environ.get("MAIN_SHORT_WHOLE_SHARES", "true").lower() != "false"
MAIN_SHORT_REQUIRE_SHORTABLE = os.environ.get("MAIN_SHORT_REQUIRE_SHORTABLE", "true").lower() != "false"
MAIN_SHORT_DISABLE_AFTER = int(os.environ.get("MAIN_SHORT_DISABLE_AFTER", "3"))
MAIN_BP_USAGE_PCT = float(os.environ.get("MAIN_BP_USAGE_PCT", "0.95"))
# Default day-limit TP acceptance semantics: day TPs die at market close,
# so no overnight limit risk exists by construction.
SCALP_TP_DAY_LIMIT = True

# Module tie-break priority (arbitration order).
SCALP_MODULE_ORDER = ("ict_ifvg", "box_theory", "volprofile_fib")
# Minimum bars per timeframe before a module may emit signals (fail-SAFE
# gate: warn + skip until enough history exists; never crash).
SCALP_MIN_BARS = {
    "ict_ifvg": {"1m": 35, "15m": 10, "30m": 15, "1h": 25, "4h": 12, "1d": 2},
    "box_theory": {"5m": 3, "1d": 2},
    "volprofile_fib": {"5m": 16, "1d": 2},
}
# Timeframe fetch spec: key -> (yfinance interval, lookback days, TTL s).
# "4h" is derived by resampling the "1h" frame (yfinance has no 4h bar).
SCALP_TF_SPEC = {
    "1m": ("1min", 3, None),  # None → fresh every tick (SCALP_CACHE_1M_SECONDS)
    "5m": ("5min", 5, SCALP_CACHE_5M_SECONDS),
    "15m": ("15min", 10, SCALP_CACHE_15M_SECONDS),
    "30m": ("30min", 25, SCALP_CACHE_30M_SECONDS),
    "1h": ("1h", 60, SCALP_CACHE_HTF_SECONDS),
    "4h": ("1h", 60, SCALP_CACHE_HTF_SECONDS),  # resampled from 1h
    "1d": ("1day", 40, SCALP_CACHE_HTF_SECONDS),
}
SCALP_EXCHANGE_TZ = "America/New_York"
# Previous-day session windows used to build the LiquidityMap (exchange
# local clock times; best-effort on RTH-centric US equity bars).
SCALP_LIQUIDITY_WINDOWS = (
    SessionWindow("asia", "20:00", "24:00"),
    SessionWindow("london", "02:00", "05:00"),
)

# ── Logging ────────────────────────────────────────────────────────
log_dir = Path("/home/team/shared/engine/logs")
log_dir.mkdir(parents=True, exist_ok=True)
today = datetime.now().strftime("%Y%m%d")
log_file = log_dir / f"trades_{today}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("live_trader")


# ── Trading logic ──────────────────────────────────────────────────

def _compute_rsi(close: "pd.Series", period: int = 14) -> "pd.Series":
    """Compute RSI (Relative Strength Index) over *period* bars.

    Mirrors the turbo trader's RSI so both engines use identical regime
    math.  RSI < REGIME_RSI_THRESHOLD means the instrument is weak/oversold.
    """
    import pandas as pd
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _regime_gate_allows_long(
    data: "pd.DataFrame",
    ma_period: int = 10,
    rsi_period: int = 14,
    rsi_threshold: float = 40.0,
) -> "tuple[bool, str]":
    """Return ``(allow, reason)`` for a mean-reversion LONG on *data*.

    A long is BLOCKED only in a confirmed downtrend — price below the
    *ma_period*-bar MA AND RSI(*rsi_period*) below *rsi_threshold*.  This is
    the #1 bleeding source on down days: the main trader unconditionally
    buys mean-reversion dips, catching falling knives (e.g. -$351 COIN,
    -$116 AVGO on 2026-08-26).  When either condition is healthy (price
    above the MA, or RSI recovering), the long is allowed — this is a
    filter, not a trend follower.  ``reason`` is human-readable for audit
    logging (empty string when allowed).
    """
    close = data["close"]
    if len(close) < max(ma_period, rsi_period) + 1:
        return True, "insufficient data"
    ma = sma(close, period=ma_period)
    rsi = _compute_rsi(close, period=rsi_period)
    cur = float(close.iloc[-1])
    cur_ma = float(ma.iloc[-1])
    cur_rsi = float(rsi.iloc[-1])
    if pd.isna(cur_ma) or pd.isna(cur_rsi):
        return True, "insufficient indicator history"
    below_ma = cur < cur_ma
    weak_rsi = cur_rsi < rsi_threshold
    if below_ma and weak_rsi:
        return False, (
            f"downtrend: price {cur:.2f} < MA{ma_period} {cur_ma:.2f} "
            f"AND RSI {cur_rsi:.1f} < {rsi_threshold}"
        )
    return True, ""


def _main_stop_client_id(symbol: str) -> str:
    """Generate a unique idempotency key for main protective stop orders.

    Format: ``algoflow_MAIN_{SYMBOL}_STOP_{timestamp_ns}`` — the
    ``algoflow_MAIN_`` prefix keeps stale stops inside the boot-time
    stale-order cancellation window (``cancel_orders_by_client_id_prefix``
    in ``run()``), while the ``_STOP_`` infix distinguishes them from
    entry/exit fill orders.
    """
    return f"algoflow_MAIN_{symbol.upper()}_STOP_{time.monotonic_ns()}"


def _main_order_client_id(symbol: str, side: str, kind: str) -> str:
    """Idempotency key for scalp entry / TP orders.

    ``kind`` is ``ENTRY`` or ``TP`` so the boot-time stale-order
    cancellation and the per-symbol order book are self-describing.
    """
    return f"algoflow_MAIN_{symbol.upper()}_{kind}_{side}_{time.monotonic_ns()}"


def _normalize_order_price(price: float) -> float:
    """Round *price* to an Alpaca-legal equity increment.

    Alpaca rejects sub-penny increments for prices >= $1.00 with 42210000
    ("sub-penny increment does not fulfill minimum pricing criteria"), which
    was observed live on 2026-09-16: strategy levels such as 211.160004 were
    rejected outright, leaving positions with no take-profit order at all.
    Sub-dollar prices keep 4 decimals (Alpaca permits sub-penny ticks there).
    """
    try:
        p = float(price)
    except (TypeError, ValueError):
        return price
    if not math.isfinite(p):
        return p
    return round(p, 2 if abs(p) >= 1.0 else 4)


def _is_invalid_stop_level_error(exc: object) -> bool:
    """True for an Alpaca 42210000 "stop price must be less|greater than
    current price" rejection.

    Those are PERMANENT for the submitted level: the stop sits on the wrong
    side of the live market price, so re-submitting the same price just
    hammers the broker (the live COIN 2026-09-16 loop).  Every other
    rejection class (e.g. wash-trade 40310000) stays retryable.
    """
    msg = str(exc).lower()
    if "42210000" not in msg:
        return False
    return ("stop price must be less than current price" in msg
            or "stop price must be greater than current price" in msg)


@dataclass(frozen=True)
class AnchoredLevels:
    """SL/TP levels re-anchored to the actual fill (see ``anchor_scalp_levels``).

    ``sl_source`` / ``tp_source`` describe how each level was derived:

    * ``strategy``            — the signal's own level was still valid vs fill
    * ``fill_risk``           — re-priced from the fill using the signal risk
    * ``fill_risk_clamped``   — as above, widened to the min-distance floor
    * ``fill_reward`` / ``fill_reward_clamped`` — same for the TP (reward leg)
    * ``no_strategy_sl``      — signal had no SL -> -6% backstop
    * ``backstop``            — strategy SL invalid vs fill AND re-pricing
                                degenerated -> -6% backstop (logged loudly)
    * ``none``                — no TP order (rely on BE/trail + stops)
    * ``unanchored``          — no valid fill price; strategy levels kept
    * ``unusable``            — no derivable level at all (caller logs loudly)
    """

    sl: float | None
    tp: float | None
    sl_source: str
    tp_source: str
    min_distance: float = 0.0

    @property
    def sl_reanchored(self) -> bool:
        return self.sl_source in ("fill_risk", "fill_risk_clamped")

    @property
    def tp_reanchored(self) -> bool:
        return self.tp_source in ("fill_reward", "fill_reward_clamped")

    @property
    def sl_is_backstop(self) -> bool:
        return self.sl_source == "backstop"


def anchor_scalp_levels(
    *,
    is_short: bool,
    fill: float,
    entry_ref: float | None = None,
    sl_ref: float | None = None,
    tp_ref: float | None = None,
    min_distance_pct: float | None = None,
    min_distance_abs: float | None = None,
    backstop_pct: float | None = None,
) -> AnchoredLevels:
    """Anchor a ScalpSet signal's SL/TP to the ACTUAL fill price.

    The broker only validates levels against the live market price, so the
    signal-time reference (the bar close the module used) is NOT a safe base
    for a MARKET entry: the fill can be points away from it (live COIN
    2026-09-16: signal ref 167.96, fill 162.58, strategy SL 167.01 -> every
    stop submission rejected with 42210000).

    Rule (deterministic, documented):

    * ``min_dist = max(fill * min_distance_pct, min_distance_abs)``.
    * SL is kept verbatim when it is still on the correct side of the fill
      AND at least ``min_dist`` away (long: ``sl <= fill - min_dist``;
      short: ``sl >= fill + min_dist``).
    * Otherwise the stop is re-priced from the fill using the signal's own
      risk amount ``|entry_ref - sl_ref|`` shifted to the fill
      (long: ``fill - risk``, short: ``fill + risk``), then widened to
      ``min_dist`` when the risk is smaller than the floor.
    * If that re-priced level is degenerate (<= 0, or still on the wrong
      side of the fill), the existing -6% backstop is used instead — the
      caller logs loudly ("strategy SL invalid vs fill — using backstop").
    * The TP follows the same rule with the signal's reward amount
      (long: ``fill + reward``); when nothing valid can be derived the TP
      is dropped (``None``) and BE/trail plus the stops take over.
    * Levels are rounded to the exchange tick via ``_normalize_order_price``.

    A missing fill (``None``/``<= 0``/NaN) cannot be anchored against, so the
    raw strategy levels are returned unchanged (``unanchored``).
    """
    pct = SCALP_STOP_MIN_DISTANCE_PCT if min_distance_pct is None else min_distance_pct
    abs_min = SCALP_STOP_MIN_DISTANCE_ABS if min_distance_abs is None else min_distance_abs
    back_pct = PROTECTIVE_STOP_PCT if backstop_pct is None else backstop_pct

    def _num(value) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None

    fill_f = _num(fill)
    if fill_f is None or fill_f <= 0:
        return AnchoredLevels(sl=_num(sl_ref), tp=_num(tp_ref),
                              sl_source="unanchored", tp_source="unanchored")
    min_dist = max(fill_f * abs(pct), abs_min)
    sign = 1.0 if is_short else -1.0
    entry_ref_f = _num(entry_ref)
    sl_ref_f = _num(sl_ref)
    tp_ref_f = _num(tp_ref)

    # ── Stop-loss ────────────────────────────────────────────────────
    sl: float | None
    sl_source: str
    if sl_ref_f is not None and (
        sl_ref_f >= fill_f + min_dist if is_short else sl_ref_f <= fill_f - min_dist
    ):
        sl, sl_source = _normalize_order_price(sl_ref_f), "strategy"
    else:
        risk = abs(entry_ref_f - sl_ref_f) if (entry_ref_f is not None and sl_ref_f is not None) else None
        if risk is not None and risk > 0:
            raw = fill_f + sign * risk
            clamped = max(raw, fill_f + min_dist) if is_short else min(raw, fill_f - min_dist)
            level = _normalize_order_price(clamped)
            if level > 0 and ((level > fill_f) if is_short else (level < fill_f)):
                sl = level
                sl_source = "fill_risk" if abs(clamped - raw) <= 1e-9 else "fill_risk_clamped"
            else:
                sl, sl_source = None, "backstop"
        else:
            sl, sl_source = None, ("backstop" if sl_ref_f is not None else "no_strategy_sl")
    if sl is None and sl_source != "unusable":
        level = _normalize_order_price(
            fill_f * (1 + back_pct) if is_short else fill_f * (1 - back_pct))
        if level > 0 and ((level > fill_f) if is_short else (level < fill_f)):
            sl, sl_source = level, ("backstop" if sl_ref_f is not None else "no_strategy_sl")
        else:
            sl, sl_source = None, "unusable"

    # ── Take-profit ──────────────────────────────────────────────────
    tp: float | None
    tp_source: str
    if tp_ref_f is not None and (
        tp_ref_f <= fill_f - min_dist if is_short else tp_ref_f >= fill_f + min_dist
    ):
        tp, tp_source = _normalize_order_price(tp_ref_f), "strategy"
    else:
        reward = abs(tp_ref_f - entry_ref_f) if (tp_ref_f is not None and entry_ref_f is not None) else None
        tp, tp_source = None, "none"
        if reward is not None and reward > 0:
            raw = fill_f - sign * reward
            clamped = min(raw, fill_f - min_dist) if is_short else max(raw, fill_f + min_dist)
            level = _normalize_order_price(clamped)
            if level > 0 and ((level < fill_f) if is_short else (level > fill_f)):
                tp = level
                tp_source = "fill_reward" if abs(clamped - raw) <= 1e-9 else "fill_reward_clamped"
    return AnchoredLevels(sl=sl, tp=tp, sl_source=sl_source, tp_source=tp_source,
                          min_distance=min_dist)


# ── Order-classification helpers (used by stop placement / replacement) ──

def _order_is_stop(o) -> bool:
    """True when an open-order object is a stop order.

    Checks the Alpaca ``type`` field, an explicit numeric ``stop_price``,
    or the ``_STOP_`` client-id marker.  MagicMock-safe: unset attrs are
    never treated as stops (a plain same-side order is not a stop).
    """
    otype = getattr(o, "type", None)
    if isinstance(otype, str) and "stop" in otype.lower():
        return True
    sp = getattr(o, "stop_price", None)
    if isinstance(sp, (int, float)) and not isinstance(sp, bool):
        return True
    cid = getattr(o, "client_order_id", None)
    if isinstance(cid, str) and "_STOP_" in cid.upper():
        return True
    return False


def _order_is_limit(o) -> bool:
    """True when an open-order object is a limit order (entry or TP)."""
    otype = getattr(o, "type", None)
    if isinstance(otype, str) and "limit" in otype.lower():
        return True
    lp = getattr(o, "limit_price", None)
    if isinstance(lp, (int, float)) and not isinstance(lp, bool):
        return True
    return False


def _order_matches_side(o, side: str) -> bool:
    return str(getattr(o, "side", "")).upper().endswith(str(side).upper())


# ── Order liveness: a cancelling order is NOT protection ──────────────────
# 2026-09-22 incident: the boot-time stale-order sweep cancelled the META and
# QQQ protective stops; the broker pinned both PENDING_CANCEL (10s cancel
# timeout each), the coverage check counted those *cancelling* orders as
# protection ("existing BUY stop found; not submitting duplicate") and the two
# shorts then sat with no protective order at the broker for 34 hours.
_DEAD_ORDER_STATUSES = frozenset({
    "CANCELED", "CANCELLED", "PENDING_CANCEL", "REJECTED", "EXPIRED",
    "REPLACED", "DONE_FOR_DAY", "FILLED", "STOPPED", "SUSPENDED",
})

# How many times a stop may be re-anchored onto the live market within one
# placement call before we give up loudly (never silently).
_MAX_MARKET_REANCHORS = 2


def _order_status_token(o) -> str:
    """Normalised status token for an open-order object ("" when absent).

    Accepts Alpaca's ``OrderStatus`` enum (``str()`` == ``OrderStatus.NEW``)
    as well as plain strings.  Anything that is not a textual status (a test
    double, a broker payload without one) yields "" = "unknown".
    """
    raw = getattr(o, "status", None)
    if raw is None:
        return ""
    if isinstance(raw, str):
        token = raw
    else:
        value = getattr(raw, "value", None)
        if isinstance(value, str):
            token = value
        else:
            return ""
    token = str(token).upper()
    if "." in token:
        token = token.rsplit(".", 1)[-1]
    return token.strip()


def _order_is_live_working(o) -> bool:
    """True only when the broker says this order is STILL WORKING.

    PENDING_CANCEL / CANCELED / REJECTED / EXPIRED (every terminal state) are
    not protection: counting them as coverage is exactly how two short
    positions were left naked for 34 hours (2026-09-22).  An order object that
    carries no recognisable status at all is treated as live — the
    conservative direction (never place a duplicate stop) and the only
    behaviour possible for a broker payload that does not report status.
    """
    token = _order_status_token(o)
    if not token:
        return True
    return token not in _DEAD_ORDER_STATUSES


def _order_has_type_metadata(o) -> bool:
    """True when the object states its order type (Alpaca always does)."""
    otype = getattr(o, "type", None)
    if isinstance(otype, str):
        return True
    return isinstance(getattr(otype, "value", None), str)


def _order_is_stop_like(o) -> bool:
    """True for a STOP-like order: recognised stop metadata, or an order whose
    type this code cannot read at all (legacy/partial payloads).

    A KNOWN non-stop type (a market/limit order) is never stop-like: a working
    opposite-side *exit* order must not be mistaken for protection.
    """
    if _order_is_limit(o):
        return False
    return _order_is_stop(o) or not _order_has_type_metadata(o)


def _order_is_protective_for(o, is_short: bool) -> bool:
    """True when *o* is a LIVE, correctly-sided, working protective stop.

    Direction-aware — a BUY stop protects a short, a SELL stop protects a long
    — and stop-only: a resting limit (take-profit / entry) or a market order
    is not protection.
    """
    if not _order_is_live_working(o):
        return False
    want = "BUY" if is_short else "SELL"
    if not _order_matches_side(o, want):
        return False
    return _order_is_stop_like(o)


def _market_anchored_stop(
    market_price, is_short: bool, pct: float | None = None,
) -> float | None:
    """A backstop level anchored to the LIVE MARKET (None when unusable).

    Short -> market * (1 + pct) (above the market), long -> market * (1 - pct)
    (below it).  Never entry-anchored: on a position that has run against us
    the entry anchor lands on the wrong side of the market, the broker rejects
    it (42210000) and the position is left with no stop at all.
    """
    if pct is None:
        pct = PROTECTIVE_STOP_PCT
    if isinstance(market_price, bool) or not isinstance(market_price, (int, float)):
        return None
    if float(market_price) <= 0:
        return None
    level = float(market_price) * (1 + pct) if is_short else float(market_price) * (1 - pct)
    return round(level, 2)


def _stop_on_wrong_side(stop_price, is_short: bool, market_price) -> bool:
    """True when *stop_price* is not on the protective side of the market.

    A short's BUY stop must sit ABOVE the market; a long's SELL stop must sit
    BELOW it.  Without a usable market price the answer is False (unknown, so
    nothing is re-anchored on a guess).
    """
    if isinstance(market_price, bool) or not isinstance(market_price, (int, float)):
        return False
    if float(market_price) <= 0:
        return False
    if isinstance(stop_price, bool) or not isinstance(stop_price, (int, float)):
        return False
    if is_short:
        return float(stop_price) <= float(market_price)
    return float(stop_price) >= float(market_price)


# ── Short-rejection classification + BP-capped sizing (turbo #29 lessons) ──

def _rejection_kind(error_message: str | None) -> str:
    """Classify a broker rejection into a coarse bucket for backoff.

    Buckets: ``"short_not_allowed"`` (not shortable / shorting disabled /
    borrow problems), ``"fractional_short"`` (fractional short sale —
    fixable by whole-share sizing), ``"buying_power"`` (insufficient
    buying power / funds / margin), ``"other"``.  Mirrors the turbo trader.
    """
    msg = (error_message or "").lower()
    if "fractional" in msg and "short" in msg:
        return "fractional_short"
    if any(
        phrase in msg
        for phrase in (
            "cannot be sold short",
            "not shortable",
            "shorting not allowed",
            "shorting is not allowed",
            "shorting is disabled",
            "short sale",
            "no shares available",
            "hard to borrow",
            "not borrowable",
            "borrow",
        )
    ):
        return "short_not_allowed"
    if any(
        phrase in msg
        for phrase in (
            "insufficient buying power",
            "insufficient funds",
            "insufficient margin",
            "insufficient balance",
            "exceeds buying power",
        )
    ):
        return "buying_power"
    return "other"


def _size_entry_qty(
    *,
    equity: float,
    buying_power: float | None,
    price: float,
    size_pct: float,
    bp_usage_pct: float = MAIN_BP_USAGE_PCT,
    whole_shares: bool = False,
) -> tuple[float, bool]:
    """Compute an entry quantity capped by AVAILABLE buying power.

    ``equity * size_pct`` is the desired notional; it is additionally capped
    at ``buying_power * bp_usage_pct`` so the order can never exceed what the
    account can fill.  When ``buying_power`` is ``None`` (unknown — account
    fetch failed), the BP cap cannot be applied: returns ``(0.0, True)``
    (skip entry, BP unknown).  With ``whole_shares=True`` the qty is floored
    to whole shares (shorts — Alpaca rejects fractional short sales).
    Mirrors the turbo trader's proven sizing path (PR #29).
    """
    if price <= 0 or equity <= 0:
        return 0.0, False
    desired = equity * size_pct
    if buying_power is None:
        return 0.0, True
    bp_cap = max(0.0, buying_power) * bp_usage_pct
    capped = desired > bp_cap
    notional = min(desired, bp_cap)
    qty = notional / price
    if whole_shares:
        qty = math.floor(qty)
    return qty, capped


def _account_buying_power(account: dict) -> float | None:
    """Extract buying power from a ``get_account()`` result.

    Returns ``None`` when the account is unavailable (sentinel) or the
    payload lacks a numeric buying power — never a fabricated number.
    """
    if not isinstance(account, dict) or account.get("available") is False:
        return None
    bp = account.get("buying_power")
    if bp is None:
        return None
    try:
        return float(bp)
    except (TypeError, ValueError):
        return None


class LiveTrader:
    def __init__(self):
        self.broker = AlpacaBroker()
        self.provider = YFinanceProvider()
        self.strategy = MeanReversionStrategy(config=STRATEGY_CONFIG)
        self.pm = PositionManager(STRATEGY_CONFIG)
        self._entry_times: dict[str, datetime] = {}  # when each position was opened
        self.day_trades: list[dict] = []
        self.start_equity = 0.0
        self._main_strategy = MAIN_STRATEGY  # "scalp" (default) | "mean_reversion" rollback
        self._scalp_init_state()

    # ── Session baseline (start equity) ──────────────────────────────
    def _restore_start_equity(self) -> bool:
        """Reload today's persisted start baseline (stable across restarts)."""
        equity = load_start_equity()
        if equity is not None and equity > 0:
            self.start_equity = equity
            return True
        return False

    def _set_start_equity(self, equity: float) -> None:
        """Record the day baseline and persist it for watchdog restarts."""
        self.start_equity = float(equity)
        if not save_start_equity(self.start_equity):
            logger.warning("Could not persist session start equity to %s",
                           "logs/session_state.json")

    async def _begin_session(self, assumed: bool) -> bool:
        """Establish the day's P&L baseline; return True when ready to trade."""
        if self.start_equity is None or self.start_equity <= 0:
            if self._restore_start_equity():
                logger.info("✅ %s — resuming session (baseline reloaded)",
                            "Market is OPEN" if not assumed else "Market assumed OPEN")
                return True
            account = await self.broker.get_account()
            equity = account_equity(account)
            if equity is None or equity <= 0:
                logger.warning("Account equity unavailable — cannot establish session baseline")
                return False
            self._set_start_equity(equity)
        logger.info("✅ %s — starting trading",
                    "Market is OPEN" if not assumed else "Market assumed OPEN")
        logger.info("   Starting equity: ${:,.2f}".format(self.start_equity))
        return True

    def _market_gate_status_name(self, market_open: bool | None) -> str:
        """Human-readable status for gate logs: OPEN / CONFIRMED-CLOSED /
        UNKNOWN.  ``None`` is UNKNOWN — an indeterminate clock that must
        NEVER be treated as a confirmed close."""
        if market_open is True:
            return "OPEN"
        if market_open is False:
            return "CONFIRMED-CLOSED"
        return "UNKNOWN"

    async def _market_clock(self, context: str) -> tuple[bool | None, int]:
        """One broker clock check with per-attempt logging.

        ``context`` names the call site (``pre-open`` / ``intraday``) so
        attempt counters and last errors are kept separately.  Returns
        ``(status, attempt_number)``; status ``None`` means indeterminate
        (timeout/exception) and is logged loudly with the last error —
        never a silent retry.
        """
        if not hasattr(self, "_clock_attempts"):
            self._clock_attempts: dict[str, int] = {}
            self._clock_last_error: dict[str, str | None] = {}
        self._clock_attempts[context] = self._clock_attempts.get(context, 0) + 1
        attempt = self._clock_attempts[context]
        try:
            status = await self.broker.is_market_open()
        except Exception as exc:  # network errors, SDK failures — indeterminate
            self._clock_last_error[context] = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Market clock check FAILED (%s — attempt %d): %s — treating as UNKNOWN, will retry",
                context, attempt, self._clock_last_error[context],
            )
            return None, attempt
        self._clock_last_error[context] = None
        return status, attempt

    async def wait_for_market_open(self):
        """Wait for 9:30 ET using local DST-aware time as the primary gate.

        Hardened 2026-09-15 (PR #35): ONLY a CONFIRMED-CLOSED Alpaca clock
        defers.  ``None``/exception is indeterminate and is retried on a
        SHORT backoff with one log line per attempt (attempt N, last
        error) — a silent multi-hour wait is impossible.  Once local ET is
        >= 09:30 + buffer and the last status was NOT confirmed-closed,
        the gate FAILS OPEN on local time.  While RTH per the local DST-
        aware schedule is open but we are still waiting, a loud overdue
        log fires every 60s so a stuck gate is glaring.
        """
        from zoneinfo import ZoneInfo
        from src.watchdog.market_status import in_rth_schedule
        logger.info("Waiting for market to open (9:30 AM ET)...")
        last_heartbeat = time.monotonic()
        last_overdue = time.monotonic()
        retry_seconds = MARKET_GATE_RETRY_SECONDS
        heartbeat_interval = MARKET_GATE_HEARTBEAT_SECONDS
        fail_open_buffer = timedelta(seconds=MARKET_GATE_FAIL_OPEN_BUFFER_SECONDS)
        overdue_cadence = MARKET_GATE_OVERDUE_LOG_SECONDS
        last_status: bool | None = None  # None = unknown (NOT confirmed-closed)
        while True:
            seconds_until = self._seconds_until_open()
            now_et = datetime.now(ZoneInfo("America/New_York"))
            before_open = now_et.weekday() < 5 and (now_et.hour, now_et.minute) < (9, 30)
            if seconds_until > 0 and (before_open or now_et.weekday() >= 5 or now_et.hour >= 16):
                # Sleep until shortly before the DST-aware opening instant;
                # cap it so heartbeat messages remain useful.
                sleep_for = min(max(seconds_until - 60.0, 1.0), heartbeat_interval)
                await asyncio.sleep(sleep_for)
                # Heartbeat: pre-open / weekend waits produce no trading
                # output for hours, so emit a low-frequency marker that
                # keeps the watchdog's staleness check truthful.
                if time.monotonic() - last_heartbeat >= heartbeat_interval:
                    mins_left = max(int(self._seconds_until_open() // 60), 0)
                    logger.info(
                        "heartbeat: waiting for market open, ~%dm remaining",
                        mins_left,
                    )
                    last_heartbeat = time.monotonic()
                continue
            # ── At/past the calculated open: ask the Alpaca clock. ──
            status, attempt = await self._market_clock("pre-open")
            if status is None:
                # Indeterminate clock — loud per-attempt retry, never silent.
                logger.warning(
                    "Market clock UNKNOWN (attempt %d, last error: %s) — "
                    "retrying in %.0fs — NOT a confirmed close",
                    attempt, self._clock_last_error.get("pre-open"), retry_seconds,
                )
            last_status = status
            status_name = self._market_gate_status_name(status)
            # ── Loud overdue log while RTH says open but we still wait. ──
            # This is the missing observability that let the 2026-09-14
            # stall rot for 9 hours: a stuck gate is now impossible to
            # miss.  Fires for ANY status (closed/unknown) — every 60s.
            if in_rth_schedule(now_et):
                mins_past = max(
                    (now_et.hour * 60 + now_et.minute) - (9 * 60 + 30), 0
                )
                if time.monotonic() - last_overdue >= overdue_cadence:
                    open_instant = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
                    fail_in = max(
                        int((open_instant + fail_open_buffer - now_et).total_seconds()), 0
                    )
                    logger.warning(
                        "⚠️ STILL WAITING FOR OPEN %d MINUTES PAST 09:30 ET "
                        "(market_open=%s, attempt=%d) — FAIL-OPEN in %ds",
                        mins_past, status_name, attempt, fail_in,
                    )
                    last_overdue = time.monotonic()
            # ── Confirmed open → begin session. ──
            if status is True:
                if await self._begin_session(assumed=False):
                    return
            # ── FAIL-OPEN: past 09:30+buffer and last status was NOT
            #    confirmed-closed → proceed on local time. ──
            elif status is not False:
                open_instant = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
                if now_et >= open_instant + fail_open_buffer:
                    logger.warning(
                        "⚠️ FAIL-OPEN at %s ET: local time past 09:30+%ds buffer and "
                        "last clock status was %s (NOT confirmed-closed, attempt %d) "
                        "— proceeding on local time",
                        now_et.strftime("%H:%M:%S"),
                        int(MARKET_GATE_FAIL_OPEN_BUFFER_SECONDS),
                        status_name, attempt,
                    )
                    if await self._begin_session(assumed=True):
                        return
            # status False → confirmed closed → keep waiting (heartbeat).
            if time.monotonic() - last_heartbeat >= heartbeat_interval:
                logger.info(
                    "Still waiting for market open, next check in %.0fs (clock confirmation)",
                    retry_seconds,
                )
                last_heartbeat = time.monotonic()
            await asyncio.sleep(retry_seconds)
    @staticmethod
    def _seconds_until_open() -> float:
        """Seconds until next 9:30 AM ET market open (DST-aware).

        Market open is 9:30 AM in the America/New_York timezone, which is
        13:30 UTC in summer (EDT, UTC-4) but 14:30 UTC in winter (EST,
        UTC-5).  The old hardcoded 13:30 UTC target was only correct during
        DST.  Computing the target in the NY zone keeps it right year-round.
        """
        from zoneinfo import ZoneInfo
        ny = ZoneInfo("America/New_York")
        now = datetime.now(ny)
        target = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if now.weekday() >= 5:  # weekend
            days_until_mon = (7 - now.weekday()) % 7
            target += timedelta(days=days_until_mon)
        elif now > target:
            target += timedelta(days=1)
            if target.weekday() >= 5:
                days_until_mon = (7 - target.weekday()) % 7
                target += timedelta(days=days_until_mon)
        return (target - now).total_seconds()

    async def run(self):
        """Main trading loop."""
        logger.info("=" * 60)
        logger.info("AlgoFlow Live Trader — STARTING")
        logger.info(f"Symbols: {SYMBOLS}")
        if self._strategy_mode() == "scalp":
            modules = []
            if SCALP_MODULE_IFVG:
                modules.append("IFVG")
            if SCALP_MODULE_BOX:
                modules.append("Box")
            if SCALP_MODULE_VOLFIB:
                modules.append("VolFib")
            logger.info("Strategy: ScalpSet [%s] (arbitration=%s, cooldown=%dbars, pair=%s)",
                        "|".join(modules) if modules else "NONE - ALL MODULES OFF",
                        SCALP_ARBITRATION, SCALP_COOLDOWN_BARS, SCALP_PAIR_SYMBOL)
            logger.info("   Shorts: %s | out data mins: 1m fresh, 5m %.0fs, "
                        "15m/30m %.0fs, 1h/4h/1d %.0fs",
                        "ON (whole-share, BP-capped, shortability-gated)" if MAIN_SHORT_REQUIRE_SHORTABLE
                        else "ON (broker shortability not required)",
                        SCALP_CACHE_5M_SECONDS, SCALP_CACHE_15M_SECONDS, SCALP_CACHE_HTF_SECONDS)
        else:
            logger.info(f"Strategy: MeanReversion (rollback) | Confidence ≥ {CONFIDENCE_THRESHOLD}")
        logger.info(f"Max positions: {MAX_POSITIONS} | Size: {POSITION_SIZE_PCT*100:.0f}% equity")
        logger.info("=" * 60)

        try:
            await self.broker.startup_health_check()
        except Exception as exc:
            logger.critical("FATAL: broker authentication/account health check failed; refusing to trade: %s", exc)
            return

        # ── Layer 4: Cancel stale orders from prior sessions ─────────
        logger.info("Cancelling any stale orders from prior sessions…")
        cancelled = await self._cancel_stale_orders()
        logger.info(f"Cancelled {cancelled} stale order(s)")
        remaining = await self.broker.get_open_orders()
        # Only a non-stop order we still own counts as an UNCONFIRMED cancel:
        # the protective stops the sweep deliberately keeps must not be
        # mistaken for a stuck cancel (they would defer the position cleanup
        # on every single boot of a protected account).
        stale_cancel_unconfirmed = any(
            str(getattr(o, "client_order_id", "")).startswith("algoflow_MAIN_")
            and not _order_is_stop_like(o)
            for o in remaining
        )
        if stale_cancel_unconfirmed:
            # A leftover order the broker pins as "pending cancel" (e.g. a
            # cancel was in flight when a previous host died) answers every
            # cancel attempt with 42210000 and can stay visible for days.
            # That must NOT kill the boot: the order is already being
            # cancelled at the broker, so we skip only the post-startup
            # position-cleanup step (its intent was to liquidate leftovers
            # whose orders we expected to be gone) and continue the normal
            # boot — sync positions, re-place protective stops, wait for
            # market. Mirror of the turbo trader's tolerant handling.
            logger.warning(
                "Stale order cancellation was not confirmed — deferring post-startup "
                "position cleanup; continuing boot (positions will be synced and "
                "protective stops re-placed)"
            )

        # ── Layer 2: Sync positions from Alpaca at startup ───────────
        logger.info("STEP 1/3: Syncing positions from broker…")
        await self._sync_positions_from_broker()
        logger.info("STEP 1/3: Position sync complete — %d open positions tracked",
                     self.pm.get_open_count())

        # ── Layer 2b: Log inherited position state ────────────────────
        for sym in self.pm.get_open_symbols():
            pos = self.pm.get_positions().get(sym)
            if pos:
                logger.info("  Inherited: %s x %s @ $%.2f", pos.quantity, sym, pos.entry_price)

        # ── Post-startup stale position cleanup ──────────────────────
        if stale_cancel_unconfirmed:
            logger.warning(
                "Post-startup stale position cleanup DEFERRED (stale order "
                "cancellation unconfirmed)"
            )
        else:
            await self._post_startup_cleanup()

        # ── Ensure every held position has a broker-side GTC stop ────
        # Placed right here — immediately after sync — so positions are
        # protected at the broker even if every process on this box dies
        # minutes later (the exact failure mode this trader must survive).
        logger.info("STEP 1/3: Ensuring protective stops on held positions…")
        await self._ensure_protective_stops()

        # Keep the process alive between sessions.  Each iteration waits for
        # the next market open, trades one session, reports its summary, then
        # resets local state before waiting for the following trading day.
        while True:
            await self.wait_for_market_open()
            logger.info("STEP 2/3: Entering main tick loop…")
            tick = 0
            self._tick_loop_last_heartbeat = time.monotonic()
            try:
                while True:
                    tick += 1
                    # ── EOD mandatory liquidation check ──────────────────
                    # ScalpSet (PR #37): main positions MAY hold overnight
                    # (GTC-protected) — mandatory EOD liquidation is turbo's
                    # behaviour only.  The mean-reversion rollback keeps the
                    # legacy EOD liquidation as-is.
                    if self._strategy_mode() != "scalp":
                        try:
                            if self._is_near_close():
                                logger.info(f"⏰ Within {MANDATORY_CLOSE_MINUTES} min of close — "
                                            "triggering mandatory EOD liquidation")
                                await self._eod_liquidate()
                                break
                        except Exception:
                            logger.exception("EOD check/liquidate failed — continuing")
                    # ── Intraday market check (same hardening as the gate) ──
                    # None/exception = indeterminate clock: NEVER treated as
                    # a confirmed close.  Retry loudly with per-attempt
                    # logging on a short backoff — a transient API outage
                    # cannot trigger a false mid-session liquidation, and a
                    # stuck clock cannot be silent.
                    try:
                        market_open, attempt = await self._market_clock("intraday")
                    except Exception:
                        logger.exception("Market-open check failed — treating as unknown, will retry")
                        market_open, attempt = None, self._clock_attempts.get("intraday", 0)
                    if market_open is None:
                        logger.warning(
                            "Market clock UNKNOWN intraday (attempt %d, last error: %s) "
                            "— retrying in %.0fs — never treating unknown as closed",
                            attempt, self._clock_last_error.get("intraday"),
                            MARKET_LOOP_UNKNOWN_RETRY_SECONDS,
                        )
                        await asyncio.sleep(MARKET_LOOP_UNKNOWN_RETRY_SECONDS)
                        continue
                    if not market_open:
                        logger.info("⏹️  Market closed — completing session and waiting for next open")
                        break
                    if self._strategy_mode() == "scalp":
                        await self._safe_tick_scalp(tick)
                    else:
                        await self._safe_tick(tick)
                    # Intraday liveness heartbeat: a healthy session can be
                    # INFO-silent for hours (no signals → no order logs),
                    # which made watchdog staleness useless intraday
                    # (2026-09-14: a 9h freeze produced zero lines during
                    # RTH).  One line per cadence proves the loop is alive.
                    if time.monotonic() - self._tick_loop_last_heartbeat >= MARKET_TICK_HEARTBEAT_SECONDS:
                        logger.info(
                            "❤️ Tick loop alive — tick %d, market OPEN (clock confirmed)",
                            tick,
                        )
                        self._tick_loop_last_heartbeat = time.monotonic()
                    logger.debug("Tick %d: complete — sleeping %ds", tick, CHECK_INTERVAL)
                    await asyncio.sleep(CHECK_INTERVAL)
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                await self.shutdown()
                return
            except Exception as e:
                logger.exception("FATAL: Unhandled exception in main loop — %s", e)

            # Preserve the session summary while keeping the broker connection
            # alive for the next session. Positions should already be flat from
            # EOD liquidation; shutdown remains a safety net if not.
            await self.shutdown(close_broker=False)
            self.pm.reset()
            self._entry_times.clear()
            self.day_trades.clear()
            self._scalp_reset_session_state()
            logger.info("Session state reset — waiting for next market open")

    async def _safe_tick_scalp(self, tick_num: int):
        """Wrapper around _tick_scalp (same contract as _safe_tick)."""
        try:
            await self._tick_scalp(tick_num)
        except Exception:
            logger.exception("Tick %d (scalp) crashed — continuing to next tick", tick_num)
            await asyncio.sleep(CHECK_INTERVAL)

    async def _safe_tick(self, tick_num: int):
        """Wrapper around _tick that catches all exceptions so one bad tick
        never kills the trader.  Logs the full traceback and continues."""
        try:
            await self._tick(tick_num)
        except Exception:
            logger.exception("Tick %d crashed — continuing to next tick", tick_num)
            await asyncio.sleep(CHECK_INTERVAL)

    async def _tick(self, tick_num: int):
        """One polling cycle: fetch → signals → execute."""
        now = datetime.now(timezone.utc)
        lookback = now - timedelta(minutes=30)

        logger.debug("Tick %d: evaluating %d symbols (%d positions open)",
                     tick_num, len(SYMBOLS), self.pm.get_open_count())

        for symbol in SYMBOLS:
            # Skip if at max positions and don't hold this one
            if self.pm.get_open_count() >= MAX_POSITIONS and not self.pm.has_position(symbol):
                continue

            try:
                logger.debug("Tick %d: fetching %s 1m bars…", tick_num, symbol)
                mdf = await self.provider.fetch_bars(
                    symbol, start=lookback, end=now, timeframe="1min"
                )
                data = mdf.df
                if data.empty:
                    continue

                signals = self.strategy.generate_signals(data)
                if not signals:
                    continue

                latest = signals[-1]
                if latest.confidence < CONFIDENCE_THRESHOLD:
                    continue

                current_price = float(data["close"].iloc[-1])

                if latest.signal_type == SignalType.BUY:
                    # ── Regime gate: skip the mean-reversion LONG in a
                    #    confirmed downtrend (price < MA10 AND RSI < 40) so we
                    #    stop catching falling knives.  This is the #1 bleeding
                    #    source on down days. ──
                    if ENABLE_REGIME_GATE:
                        allow, reason = _regime_gate_allows_long(
                            data,
                            ma_period=REGIME_MA_PERIOD,
                            rsi_period=REGIME_RSI_PERIOD,
                            rsi_threshold=REGIME_RSI_THRESHOLD,
                        )
                        if not allow:
                            logger.info(
                                "🚫 REGIME GATE %s: skipping mean-reversion LONG — %s",
                                symbol, reason,
                            )
                            continue
                    await self._handle_buy(symbol, current_price, latest.confidence)
                elif latest.signal_type == SignalType.SELL:
                    await self._handle_sell(symbol, current_price, latest.confidence)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

        # Check stop-loss / take-profit
        await self._check_risk_stops()

    async def _handle_buy(self, symbol: str, price: float, confidence: float):
        if self.pm.has_position(symbol):
            return
        if self.pm.get_open_count() >= MAX_POSITIONS:
            return

        account = await self.broker.get_account()
        equity = account_equity(account)
        if equity is None:
            logger.warning(f"⚠️  {symbol}: account equity unavailable — skipping entry")
            return
        if not self.pm.can_open(symbol, equity):
            return

        value = equity * POSITION_SIZE_PCT
        qty = value / price if price > 0 else 0
        if qty < 1:
            return

        order = Order(symbol=symbol, side=OrderSide.BUY, quantity=qty, order_type=OrderType.MARKET,
                      client_id=f"algoflow_MAIN_{symbol.upper()}_BUY_{time.monotonic_ns()}")
        result = await self.broker.place_order(order)

        if is_order_alive(result.status):
            self.pm.open_position(symbol, qty, price)
            self._entry_times[symbol.upper()] = datetime.now(timezone.utc)  # time-based exit
            logger.info(f"📈 BUY  {symbol}: {qty:.1f} shares @ ${price:.2f} = ${value:,.2f} | "
                        f"conf={confidence:.2f} | order={result.order_id[:8]}")
            # ── Place GTC protective stop at broker ─────────────────
            # The stop-placement retry machinery handles the case where the
            # entry BUY is still open (wash-trade reject) by waiting and
            # retrying; the position is never left broker-unprotected.
            await self._place_protective_stop(symbol, qty, price)
        else:
            logger.warning(f"❌ BUY {symbol} REJECTED: {result.status}")

    async def _handle_sell(self, symbol: str, price: float, confidence: float):
        if not self.pm.has_position(symbol):
            return

        pos = self.pm.get_positions().get(symbol.upper())
        if pos is None:
            return
        abs_qty = abs(pos.quantity)
        is_short = pos.quantity < 0

        # ── Cancel the GTC protective stop before closing ────────────
        # Otherwise the stop stays open (GTC) after the position closes and
        # could later trigger as an accidental short.  If the cancellation
        # is not confirmed, defer the close — the position remains tracked
        # AND protected, the safest state.
        if not await self._cancel_protective_stops(symbol):
            logger.warning(
                "SELL %s deferred: protective-order cancellation was not confirmed",
                symbol,
            )
            return

        outcome = await close_position_verified(
            self.pm,
            self.broker,
            symbol,
            exit_reason="signal",
            client_id=f"algoflow_MAIN_{symbol.upper()}_SELL_{time.monotonic_ns()}",
        )
        if outcome.status == "filled" and outcome.trade is not None:
            self._entry_times.pop(symbol.upper(), None)
            trade = outcome.trade
            logger.info(
                f"📉 SELL {symbol}: {abs(trade.quantity):.1f} shares @ ${outcome.fill_price:.2f} | "
                f"P&L: ${trade.pnl:,.2f} | conf={confidence:.2f} | order={outcome.order_id[:8]}"
            )
        elif outcome.status == "pending":
            # The close order is live but the broker has not confirmed a fill.
            # We keep the position tracked and book NO P&L; the next position
            # sync reconciles the fill from broker fill history.
            logger.info(
                f"⏳ SELL {symbol}: close pending (fill not confirmed) — no P&L booked | "
                f"conf={confidence:.2f}"
            )
        elif outcome.status == "rejected":
            error = (outcome.message or "").lower()
            if "cannot be sold short" in error:
                self.pm.discard_position(symbol, reason="broker says position is not held")
                self._entry_times.pop(symbol.upper(), None)
                logger.warning("SELL %s rejected as phantom position; removed from tracking", symbol)
            else:
                # The protective stop was cancelled above — restore it so the
                # position doesn't run naked because of a rejected exit.
                logger.warning(
                    "SELL %s rejected (%s); keeping position tracked — restoring protective stop",
                    symbol, outcome.message,
                )
                await self._place_protective_stop(symbol, abs_qty, pos.entry_price, is_short=is_short)

    async def _sync_positions_from_broker(self):
        """Reconcile local PositionManager with Alpaca's actual positions.

        - New positions on Alpaca that we don't know about → add to PM.
        - Positions in PM that Alpaca doesn't have → remove from PM (stale).
        """
        logger.info("Syncing positions from broker…")
        try:
            broker_positions = await self.broker.get_positions()
        except Exception as e:
            logger.error(f"Failed to fetch broker positions for sync: {e}")
            return

        broker_symbols = {p["symbol"].upper(): p for p in broker_positions}
        pm_symbols = set(self.pm.get_open_symbols())

        added = 0
        removed = 0

        # Add positions the broker knows about but we don't
        for sym, pos_data in broker_symbols.items():
            if sym not in pm_symbols:
                self.pm.open_position(
                    symbol=sym,
                    quantity=pos_data["qty"],
                    entry_price=pos_data["avg_entry_price"],
                )
                logger.info(
                    "  + Added %s: %s shares @ $%.2f (sync)",
                    sym, pos_data["qty"], pos_data["avg_entry_price"],
                )
                added += 1

        # Remove positions we track but broker doesn't have.  A tracked
        # position that vanishes from the broker (e.g. a pending cleanup
        # MARKET SELL that filled in the background, or an external sale) is
        # booked at the broker's true last fill price.  If no fill price can
        # be determined, the position is DROPPED without booking P&L — never
        # fabricate a mark-based loss.
        for sym in pm_symbols - set(broker_symbols):
            fill_price = await self.broker.get_last_fill_price(sym)
            if fill_price is not None:
                self.pm.close_position(sym, exit_price=fill_price, exit_reason="sync_removed")
                logger.info(
                    "  - Removed %s (not on broker) — P&L booked at broker fill $%.2f",
                    sym, fill_price,
                )
            else:
                self.pm.discard_position(sym, reason="sync_removed_no_fill")
                logger.info(
                    "  - Removed stale %s (not on broker) — no fill price available; "
                    "dropped WITHOUT booking P&L",
                    sym,
                )
            removed += 1

        logger.info(
            "Synced %d positions from broker (%d added, %d removed)",
            len(broker_symbols), added, removed,
        )

    async def _cancel_stale_orders(self) -> int:
        """Boot sweep of OUR stale orders — never a live protective stop.

        Structural fix for the 2026-09-22 incident: the sweep used to cancel
        every ``algoflow_MAIN_`` order, which included the protective stops of
        positions we had just inherited.  The two cancelled stops got pinned
        PENDING_CANCEL, the protection check then counted those cancelling
        orders as coverage, and the positions sat naked for 34 hours.  The
        sweep and the protection check must never disagree about the same
        order, so the sweep simply does not touch an order that is a
        protective stop for a position the broker still holds.

        Returns the number of orders cancelled.  When the held positions
        cannot be read, NO order that looks like a protective stop is
        cancelled (fail safe: leaving a stale stop is recoverable, leaving a
        position naked is not).
        """
        prefix = "algoflow_MAIN_"
        try:
            open_orders = await self.broker.get_open_orders()
        except Exception as exc:
            logger.warning(
                "Stale-order sweep: cannot read open orders (%s) — nothing cancelled", exc,
            )
            return 0
        held: set[str] = set()
        positions_known = True
        try:
            held = {
                str(p.get("symbol", "")).upper()
                for p in await self.broker.get_positions()
            }
        except Exception as exc:
            positions_known = False
            logger.warning(
                "Stale-order sweep: cannot read broker positions (%s) — no order that "
                "looks like a protective stop will be cancelled on this boot", exc,
            )
        cancelled, kept = 0, []
        for o in open_orders:
            cid = str(getattr(o, "client_order_id", "") or "")
            if not cid.startswith(prefix):
                continue
            sym = str(getattr(o, "symbol", "")).upper()
            # Stop-like (see _order_is_stop_like): never cancel one while the
            # position it protects is still held.
            protective = _order_is_stop_like(o)
            if protective and (not positions_known or sym in held):
                kept.append(sym)
                logger.info(
                    "🛡️  stale-order sweep: KEEPING protective order %s for held position "
                    "%s (a stop protecting a live position is never stale)",
                    str(getattr(o, "id", ""))[:8], sym,
                )
                continue
            try:
                if await self.broker.cancel_order_and_wait(str(o.id)):
                    cancelled += 1
            except Exception as exc:
                logger.warning(
                    "stale-order sweep: cancel of order %s failed: %s",
                    str(getattr(o, "id", ""))[:8], exc,
                )
        if kept:
            logger.info(
                "🛡️  stale-order sweep kept %d protective stop(s) for held positions: %s",
                len(kept), ", ".join(sorted(set(kept))),
            )
        return cancelled

    # ── Post-startup stale position cleanup ────────────────────────────

    async def _post_startup_cleanup(self):
        """Liquidate stale positions if trader starts while market is closed.

        If the market is closed (e.g. after a crash/restart/sandbox cycle
        that missed the regular EOD window) and there are open positions
        at the broker, liquidate them immediately so nothing hangs overnight.
        """
        try:
            try:
                market_open = await self.broker.is_market_open()
            except Exception as exc:
                # Indeterminate clock (timeout/error) — do NOT liquidate on
                # an unknown clock.  Only a CONFIRMED close may trigger
                # cleanup, or we could sell into a live tape.
                logger.warning(
                    "🧹 Post-close cleanup skipped: market clock UNKNOWN (%s) — "
                    "only a confirmed close may liquidate", exc,
                )
                return
            if market_open is not False:
                if market_open is None:
                    logger.info(
                        "🧹 Post-close cleanup skipped: market clock UNKNOWN — "
                        "only a confirmed close may liquidate",
                    )
                return  # Market is open (True) or unknown (None) — no cleanup

            positions = await self.broker.get_positions()
            if not positions:
                return

            logger.info(
                "🧹 Post-close cleanup: liquidating %d stale position(s) from previous session",
                len(positions),
            )
            liquidated = 0
            for p in positions:
                sym = p.get("symbol")
                qty = float(p.get("qty", 0))
                if qty > 0:
                    # Cancel the GTC protective stop before liquidating so it
                    # can't outlive the position.
                    if not await self._cancel_protective_stops(sym):
                        logger.warning("🧹 Cleanup SELL %s deferred: cancellation not confirmed", sym)
                        continue
                    logger.info("🧹 Cleanup SELL %s: %s shares", sym, qty)
                    outcome = await close_position_verified(
                        self.pm, self.broker, sym,
                        exit_reason="post_close_cleanup",
                    )
                    if outcome.status == "filled":
                        liquidated += 1
                    elif outcome.status == "pending":
                        logger.info(
                            "🧹 Cleanup SELL %s: close pending (fill not confirmed) — "
                            "no P&L booked, will reconcile at next sync",
                            sym,
                        )
                    else:
                        logger.warning(
                            "🧹 Cleanup SELL %s rejected (%s); restoring protective stop and keeping position",
                            sym, outcome.message,
                        )
                        await self._place_protective_stop(
                            sym, qty, float(p.get("avg_entry_price", 0)) or 0,
                        )
            logger.info(
                "🧹 Post-close cleanup complete — %d position(s) liquidated",
                liquidated,
            )
        except Exception:
            logger.exception("🧹 Post-close cleanup failed — continuing startup")

    # ── EOD mandatory liquidation ─────────────────────────────────────

    @staticmethod
    def _is_near_close() -> bool:
        """Return True if we are within MANDATORY_CLOSE_MINUTES of market close.

        Market close is 20:00 UTC (4 PM ET).  We trigger mandatory liquidation
        MANDATORY_CLOSE_MINUTES before that.
        """
        now = datetime.now(timezone.utc)
        close_time = now.replace(
            hour=MARKET_CLOSE_UTC_HOUR,
            minute=MARKET_CLOSE_UTC_MINUTE,
            second=0,
            microsecond=0,
        )
        seconds_until_close = (close_time - now).total_seconds()
        return 0 <= seconds_until_close <= (MANDATORY_CLOSE_MINUTES * 60)

    async def _eod_liquidate(self):
        """Liquidate ALL broker positions for mandatory end-of-day close."""
        positions = await self.broker.get_positions()
        count = len(positions)
        if count == 0:
            logger.info("⏰ Mandatory EOD liquidation — no positions to close")
            return

        liquidated = 0
        for p in positions:
            sym = p.get("symbol")
            qty = float(p.get("qty", 0))
            if qty > 0:
                # ── Cancel the GTC protective stop before liquidating ──
                # If the cancellation is not confirmed, defer: the position
                # stays tracked AND protected overnight — safer than a naked
                # position or an orphaned stop.
                if not await self._cancel_protective_stops(sym):
                    logger.warning("⏰ EOD SELL %s deferred: cancellation not confirmed", sym)
                    continue
                logger.info(f"⏰ EOD closing {sym}: {qty} shares...")
                outcome = await close_position_verified(
                    self.pm, self.broker, sym, exit_reason="eod",
                )
                if outcome.status == "filled":
                    liquidated += 1
                elif outcome.status == "pending":
                    logger.info(
                        "⏰ EOD SELL %s: close pending (fill not confirmed) — no P&L booked",
                        sym,
                    )
                else:
                    logger.warning(
                        "⏰ EOD SELL %s rejected (%s); restoring protective stop and keeping position",
                        sym, outcome.message,
                    )
                    await self._place_protective_stop(
                        sym, qty, float(p.get("avg_entry_price", 0)) or 0,
                    )

        logger.info(f"⏰ Mandatory EOD liquidation — {liquidated} positions closed.")

    # ── Broker-level protective stops ────────────────────────────────

    async def _market_reference_price(self, symbol: str) -> float | None:
        """Best-effort LIVE market price for *symbol* (None when unknown).

        Source: the broker's own position mark (``current_price``), which is
        refreshed on every call.  Deliberately NOT the entry price and NOT a
        historical fill: an entry-anchored level is exactly what leaves a
        losing inherited position unprotected (2026-09-22 AVGO).
        """
        sym = symbol.upper()
        try:
            positions = await self.broker.get_positions()
        except Exception as exc:
            logger.warning("Market reference for %s: position fetch failed (%s)", sym, exc)
            return None
        for p in positions or []:
            try:
                if str(p.get("symbol", "")).upper() != sym:
                    continue
                price = p.get("current_price")
            except Exception:
                continue
            if isinstance(price, bool) or not isinstance(price, (int, float)):
                continue
            if float(price) > 0:
                return float(price)
        return None

    def _unprotected_store(self) -> set:
        store = getattr(self, "_unprotected_symbols", None)
        if store is None:
            store = set()
            self._unprotected_symbols = store
        return store

    def _mark_unprotected(self, symbol: str, detail: str) -> None:
        """Record + shout about a held position with NO live broker stop."""
        sym = symbol.upper()
        self._unprotected_store().add(sym)
        logger.error(
            "🚨 UNPROTECTED %s: no live protective stop at the broker — %s",
            sym, detail,
        )

    def _mark_protected(self, symbol: str, stop_price: float | None = None) -> None:
        """Clear the unprotected marker for a symbol whose stop is live."""
        self._unprotected_store().discard(symbol.upper())
        if stop_price is not None:
            logger.debug("🛡️  %s: protection restored at $%.2f", symbol.upper(), stop_price)

    async def _place_protective_stop(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        max_attempts: int = STOP_PLACEMENT_MAX_ATTEMPTS,
        initial_delay: float = STOP_PLACEMENT_INITIAL_DELAY,
        is_short: bool = False,
        stop_price: float | None = None,
        market_price: float | None = None,
    ) -> bool:
        """Place a GTC protective stop-loss order at the broker.

        This order survives process death and sandbox cycling — Alpaca holds
        it until triggered or cancelled.  For LONG positions the stop is a
        SELL at ``entry_price * (1 - PROTECTIVE_STOP_PCT)`` (6% below entry);
        for SHORT positions it is a BUY at ``entry_price * (1 + pct)`` (6%
        ABOVE entry — a short loses money when price rises).

        ``stop_price`` (scalp integration, PR #37): when given, the stop is
        placed EXACTLY at that level — the strategy's risk-defined SL — and
        the -6% backstop is used only when it is ``None``.  Passed through
        for every scalp entry so teardown stops reconcile with strategy risk.

        ScalpSet callers pass the level ALREADY ANCHORED TO THE FILL
        (``_finalize_open_bundle`` → ``anchor_scalp_levels``), including the
        backstop price itself when the strategy SL was invalid vs the fill.
        An INVALID-LEVEL rejection is not retried (see
        ``_is_invalid_stop_level_error``): the level is permanently wrong for
        the current market, so the loop fast-fails and logs once.

        Duplicate detection is TYPE-AWARE: an existing same-side STOP order
        means the stop is already there (skip), while a same-side LIMIT (the
        scalp take-profit day-limit order) or a market order never counts as
        a stop — otherwise a resting TP would suppress stop placement.

        Retries with backoff: submitting the stop while the entry order is
        still open makes Alpaca reject it as a "potential wash trade"
        (``opposite side market/stop order exists``).  Each attempt
        re-checks the symbol's open orders so the stop is never submitted
        while an opposite-side order is live.
        """
        sym = symbol.upper()
        qty = int(qty)
        if qty <= 0:
            logger.warning("🛡️  STOP %s: position too small for protective stop (qty < 1 share)", sym)
            return False

        stop_side = "BUY" if is_short else "SELL"
        entry_side = "SELL" if is_short else "BUY"
        entry_anchored = stop_price is None
        if stop_price is not None:
            origin = "strategy SL"
        else:
            stop_price = round(
                entry_price * (1 + PROTECTIVE_STOP_PCT) if is_short
                else entry_price * (1 - PROTECTIVE_STOP_PCT),
                2,
            )
            origin = f"backstop entry {('+' if is_short else '-')}{PROTECTIVE_STOP_PCT * 100:.0f}%"
        stop_price = round(float(stop_price), 2)

        # ── Side-aware, MARKET-anchored backstop (2026-09-22) ──────────
        # A position re-synced after a restart can be far past its entry:
        # the live AVGO short had entry 337.75 and market 361.10, so the
        # entry-anchored backstop (337.75 * 1.06 = 358.02) sat BELOW the
        # market, Alpaca answered 42210000, the fast-fail path returned and
        # the short was left with no protective order at all.  Any
        # entry-derived level that lands on the wrong side of the live market
        # is re-anchored to the MARKET, never kept.  A strategy-provided SL is
        # left alone here (its geometry is the strategy's business); it is
        # re-anchored only if the broker actually rejects it as invalid.
        if entry_anchored and market_price is None:
            market_price = await self._market_reference_price(sym)
        if _stop_on_wrong_side(stop_price, is_short, market_price):
            anchored = _market_anchored_stop(market_price, is_short)
            if anchored is not None:
                logger.warning(
                    "🛡️  STOP %s: level $%.2f (%s) is on the WRONG side of the live "
                    "market $%.2f for a %s — re-anchoring to the market %s%.0f%% = "
                    "$%.2f so the position is not left unprotected",
                    sym, stop_price, origin, float(market_price),
                    "SHORT" if is_short else "LONG",
                    "+" if is_short else "-", PROTECTIVE_STOP_PCT * 100, anchored,
                )
                stop_price = anchored
                origin = ("market-anchored backstop "
                          f"{('+' if is_short else '-')}{PROTECTIVE_STOP_PCT * 100:.0f}%")
        reanchors_used = 0

        for attempt in range(1, max_attempts + 1):
            # ── Re-check the symbol's open orders before each attempt ──
            # Another process may have placed a stop order during the
            # entry-to-stop window, and an open opposite-side order would
            # make the stop bounce off Alpaca's wash-trade filter.
            try:
                existing = await self.broker.get_open_orders(symbol=sym)
                # A same-side STOP (or, for legacy order objects that don't
                # expose stop metadata, a plain same-side order) means the
                # stop is already there.  A same-side LIMIT (scalp TP day
                # limit) never counts as a stop, and neither does a
                # cancelling/terminal order: a PENDING_CANCEL stop is NOT
                # protection (2026-09-22 — two shorts naked for 34h).
                stop_present = any(
                    _order_is_protective_for(o, is_short) for o in existing
                )
                if stop_present:
                    logger.info("🛡️  STOP %s: existing %s stop found; not submitting duplicate",
                                sym, stop_side)
                    return True
                entry_open = any(
                    _order_is_live_working(o)
                    and _order_matches_side(o, entry_side)
                    and not _order_is_stop(o)
                    for o in existing
                )
                if entry_open:
                    if attempt < max_attempts:
                        logger.info(
                            "🛡️  STOP %s: entry %s still open — waiting %.0fs before retry (%d/%d)",
                            sym, entry_side, initial_delay, attempt, max_attempts,
                        )
                        await asyncio.sleep(initial_delay)
                        continue
            except Exception as exc:
                logger.warning(
                    "🛡️  STOP %s: cannot verify open orders (attempt %d/%d): %s",
                    sym, attempt, max_attempts, exc,
                )

            client_id = _main_stop_client_id(sym)
            try:
                await self.broker.place_stop_order(
                    symbol=sym,
                    qty=qty,
                    stop_price=stop_price,
                    client_id=client_id,
                    side=stop_side,
                )
                logger.info(
                    "🛡️  STOP %s: GTC %s stop-loss at $%.2f (%s, entry=%.2f%s)",
                    sym, stop_side, stop_price, origin, entry_price,
                    " [SHORT]" if is_short else "",
                )
                self._mark_protected(sym, stop_price)
                return True
            except Exception as exc:
                if _is_invalid_stop_level_error(exc):
                    # The level sits on the wrong side of the live market
                    # (the live COIN 2026-09-16 loop, and the AVGO short after
                    # the 2026-09-22 restart).  Fast-failing here is what left
                    # the position with NO broker stop, so: re-anchor onto the
                    # market and try again; only give up — loudly, and with the
                    # position explicitly marked unprotected — when even that
                    # is impossible.
                    fresh = await self._market_reference_price(sym)
                    anchored = _market_anchored_stop(fresh, is_short)
                    if (
                        anchored is not None
                        and anchored != stop_price
                        and reanchors_used < _MAX_MARKET_REANCHORS
                        and attempt < max_attempts
                    ):
                        reanchors_used += 1
                        logger.error(
                            "🛡️  STOP %s: INVALID-LEVEL rejection (attempt %d/%d) — %s. "
                            "Level $%.2f is on the wrong side of the live market %s; "
                            "re-anchoring to $%.2f and trying again — this position "
                            "must not stay unprotected.",
                            sym, attempt, max_attempts, exc, stop_price,
                            f"${float(fresh):.2f}" if fresh is not None else "UNKNOWN",
                            anchored,
                        )
                        stop_price = anchored
                        origin = ("market-anchored backstop "
                                  f"{('+' if is_short else '-')}{PROTECTIVE_STOP_PCT * 100:.0f}%")
                        continue
                    logger.error(
                        "🛡️  STOP %s: INVALID-LEVEL rejection (attempt %d/%d) — %s. "
                        "Level $%.2f is on the wrong side of the market and no "
                        "market-referenced level could be derived (market=%s). NOT "
                        "retrying this level (permanent for this price). Position has "
                        "NO broker-level stop; in-process risk checks stay active.",
                        sym, attempt, max_attempts, exc, stop_price,
                        f"${float(fresh):.2f}" if fresh is not None else "UNKNOWN",
                    )
                    self._mark_unprotected(
                        sym,
                        "level $%.2f was rejected as invalid and no "
                        "market-referenced level could be derived "
                        "(market reference unavailable)" % stop_price,
                    )
                    return False
                if attempt < max_attempts:
                    delay = initial_delay * attempt
                    logger.warning(
                        "🛡️  STOP %s: placement rejected (attempt %d/%d) — %s; retrying in %.0fs",
                        sym, attempt, max_attempts, exc, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "🛡️  STOP %s: FAILED after %d attempts — %s. Position has NO "
                        "broker-level stop; in-process risk checks still active.",
                        sym, max_attempts, exc,
                    )
        self._mark_unprotected(
            sym, f"stop placement failed after {max_attempts} attempts",
        )
        return False

    async def _cancel_protective_stops(self, symbol: str) -> bool:
        """Cancel all open orders for *symbol* (protective stops).

        Called before closing a position so the GTC stop doesn't remain
        open after the position is gone (an orphaned SELL stop could later
        trigger as an accidental short).  Returns True only when every
        previously-open order for the symbol is confirmed gone.
        """
        sym = symbol.upper()
        try:
            open_orders = await self.broker.get_open_orders(symbol=sym)
        except Exception as exc:
            logger.warning("Failed to fetch open orders for %s: %s", sym, exc)
            return False

        cancelled = 0
        for o in open_orders:
            try:
                if await self.broker.cancel_order_and_wait(str(o.id)):
                    cancelled += 1
                    logger.debug("  Cancelled order %s for %s", str(o.id)[:8], sym)
                else:
                    logger.warning("  Cancellation not confirmed for order %s", str(o.id)[:8])
            except Exception as exc:
                logger.warning("  Failed to cancel order %s: %s", str(o.id)[:8], exc)

        if cancelled:
            logger.info("🗑️  Cancelled %d protective order(s) for %s", cancelled, sym)
        remaining = await self.broker.get_open_orders()
        remaining_ids = {str(getattr(order, "id", "")) for order in remaining}
        return not any(str(o.id) in remaining_ids for o in open_orders)

    async def _ensure_protective_stops(self):
        """Ensure every inherited main position has a LIVE protective stop.

        Called right after ``_sync_positions_from_broker()`` at startup, so
        positions that survived a process-group kill / sandbox cycle are
        protected within seconds of boot.  Only symbols in the main trader's
        ``SYMBOLS`` set are touched: the account is shared with the turbo
        trader, and we must never place our own stops on its leveraged-ETF
        positions.

        Coverage is defined strictly: an order counts only when the broker
        says it is still WORKING, it is a stop (not a resting limit) and it is
        on the protective side for this position's direction.  A
        PENDING_CANCEL / cancelled / rejected / expired order is NOT
        protection — 2026-09-22: two shorts sat naked for 34 hours because a
        cancelling order satisfied this check.  The "no stop found" warning is
        emitted only when a placement is actually attempted, so a healthy sync
        is never a false alarm.
        """
        main_set = {s.upper() for s in SYMBOLS}
        if not self.pm.get_open_symbols():
            logger.info("🛡️  No inherited positions — skipping protective stop check")
            return

        # Fetch all open orders once so we can check live stop coverage
        try:
            open_orders = await self.broker.get_open_orders()
        except Exception as exc:
            logger.warning("Cannot verify protective stops — order fetch failed: %s", exc)
            logger.error(
                "🚨 PROTECTION UNVERIFIED (startup): %d inherited position(s) could not "
                "be checked for a broker stop because the order fetch failed — verify "
                "at the broker before trusting this session",
                len(self.pm.get_open_symbols()),
            )
            return

        unprotected: list[str] = []
        for sym in list(self.pm.get_open_symbols()):
            if sym not in main_set:
                continue
            pos = self.pm.get_positions().get(sym)
            if pos is None:
                continue
            is_short = pos.quantity < 0
            stop_side = "BUY" if is_short else "SELL"

            live_stops = [
                o for o in open_orders
                if str(getattr(o, "symbol", "")).upper() == sym
                and _order_is_protective_for(o, is_short)
            ]
            if live_stops:
                logger.info(
                    "🛡️  %s: already protected by live %s stop %s — no action",
                    sym, stop_side,
                    ", ".join(str(getattr(o, "id", ""))[:8] for o in live_stops),
                )
                self._mark_protected(sym)
                continue

            dead = [
                o for o in open_orders
                if str(getattr(o, "symbol", "")).upper() == sym
                and not _order_is_live_working(o)
            ]
            if dead:
                logger.warning(
                    "🛡️  %s: %d order(s) for this symbol are NOT working "
                    "(cancelling/terminal: %s) — they are not protection; "
                    "placing a live stop now",
                    sym, len(dead),
                    ", ".join(
                        f"{str(getattr(o, 'id', ''))[:8]}={_order_status_token(o) or 'unknown'}"
                        for o in dead
                    ),
                )
            logger.warning(
                "🛡️  %s: NO live protective stop for inherited position "
                "(%s shares @ $%.2f%s) — attempting placement now",
                sym, abs(pos.quantity), pos.entry_price,
                " [short]" if is_short else "",
            )
            placed = await self._place_protective_stop(
                sym, int(abs(pos.quantity)), pos.entry_price, is_short=is_short,
            )
            if placed:
                logger.info(
                    "🛡️  %s: protective stop CONFIRMED live at the broker after placement",
                    sym,
                )
            else:
                unprotected.append(sym)

        if unprotected:
            logger.error(
                "🚨 PROTECTION GAP after startup sync: %s still have NO live broker "
                "stop despite placement attempts — these positions are exposed to a "
                "gap move; verify at the broker NOW",
                ", ".join(sorted(unprotected)),
            )
        else:
            logger.info("🛡️  Protective stop check complete — every held position covered")

    async def _audit_position_protection(self, *, context: str, heal: bool = True) -> list[str]:
        """Verify — and by default restore — protection for every held position.

        Runs at the end of every position-sync pass.  A sync that ends with a
        held position lacking a LIVE broker stop is an incident, not a log
        line in passing (2026-09-22: a restart left two shorts naked for 34
        hours and nothing said so).  Returns the symbols still unprotected.

        ``heal`` re-places a missing stop (throttled per symbol) unless
        another non-stop working order for that symbol is in flight, where a
        new stop would only bounce off the broker's wash-trade filter.
        """
        self._scalp_init_state()
        main_set = {s.upper() for s in SYMBOLS}
        held = [s for s in self.pm.get_open_symbols() if s in main_set]
        if not held:
            return []
        try:
            open_orders = await self.broker.get_open_orders()
        except Exception as exc:
            logger.error(
                "🚨 PROTECTION UNVERIFIED (%s): cannot fetch open orders (%s) — %d held "
                "position(s) (%s) are of UNKNOWN protection state",
                context, exc, len(held), ", ".join(sorted(held)),
            )
            return []
        throttles = getattr(self, "_protection_retry_ts", None)
        if throttles is None:
            throttles = {}
            self._protection_retry_ts = throttles
        unprotected: list[str] = []
        for sym in sorted(held):
            pos = self.pm.get_positions().get(sym)
            if pos is None:
                continue
            is_short = pos.quantity < 0
            if any(
                str(getattr(o, "symbol", "")).upper() == sym
                and _order_is_protective_for(o, is_short)
                for o in open_orders
            ):
                self._mark_protected(sym)
                continue
            busy = any(
                str(getattr(o, "symbol", "")).upper() == sym
                and _order_is_live_working(o)
                and not _order_is_protective_for(o, is_short)
                and not _order_is_limit(o)
                for o in open_orders
            )
            logger.warning(
                "🚨 PROTECTION GAP (%s): %s (%s %s @ $%.2f) has NO live broker stop; "
                "%s",
                context, sym, "short" if is_short else "long", abs(pos.quantity),
                pos.entry_price,
                "another order is in flight — re-placing on the next pass"
                if busy else ("attempting placement now" if heal else "no re-placement attempted"),
            )
            if busy or not heal:
                unprotected.append(sym)
                continue
            now = time.monotonic()
            if now - float(throttles.get(sym) or 0.0) < SCALP_NO_STOP_WARN_SECONDS:
                unprotected.append(sym)
                continue
            throttles[sym] = now
            ok = await self._place_protective_stop(
                sym, int(abs(pos.quantity)), pos.entry_price, is_short=is_short,
            )
            if not ok:
                unprotected.append(sym)
                logger.error(
                    "🚨 PROTECTION GAP (%s): %s is STILL unprotected after a placement "
                    "attempt — a gap move would be unhedged", context, sym,
                )
        return unprotected

# ══════════════════════════════════════════════════════════════════
    # Scalp strategy set — data assembly, evaluation, execution (PR #37)
    # ══════════════════════════════════════════════════════════════════

    def _strategy_mode(self) -> str:
        """Active strategy mode for THIS instance.

        Production instances are built through __init__ (which records the
        module MAIN_STRATEGY) and therefore run the scalp set by default.
        Instances created without __init__ (hermetic test doubles via
        object.__new__) have no attribute and keep the legacy
        mean-reversion flow so the pre-existing run()-level test contracts
        (e.g. market-gate ticks; regime-gate ticks) remain exactly intact.
        """
        return getattr(self, "_main_strategy", "mean_reversion")

    def _scalp_init_state(self) -> None:
        """Idempotently (re)initialise all scalp-side session state.

        Called from ``__init__`` and lazily by every scalp method so
        ``object.__new__``-built test instances work without ``__init__``.
        """
        if not hasattr(self, "_frame_cache"):
            self._frame_cache: dict[tuple[str, str], tuple[float, pd.DataFrame | None]] = {}
        if not hasattr(self, "_scalp_bundles"):
            # sym -> pending LIMIT-entry bundle awaiting a broker fill:
            # {"qty", "direction", "sl", "tp", "strategy", "entry_price"}
            self._scalp_bundles: dict[str, dict] = {}
        if not hasattr(self, "_scalp_positions"):
            # sym -> live-position management state:
            # {"entry", "sl", "tp", "direction", "stop_placed", "be_done",
            #  "be_trigger_r", "be_buffer", "trailing", "trail_r",
            #  "trail_trigger_r", "stop_order_id"}
            self._scalp_positions: dict[str, dict] = {}
        if not hasattr(self, "_cooldown_until"):
            self._cooldown_until: dict[str, float] = {}
        if not hasattr(self, "_last_signal_key"):
            self._last_signal_key: dict[str, str] = {}
        if not hasattr(self, "_shorts_disabled"):
            self._shorts_disabled: set[str] = set()
        if not hasattr(self, "_short_rejections"):
            self._short_rejections: dict[str, int] = {}
        if not hasattr(self, "_shortable_cache"):
            self._shortable_cache: dict[str, bool | None] = {}
        if not hasattr(self, "_scalp_data_warned"):
            self._scalp_data_warned: set[str] = set()
        if not hasattr(self, "_scalp_data_day"):
            self._scalp_data_day: str = ""
        if not hasattr(self, "_scalp_ready_logged"):
            self._scalp_ready_logged: set[str] = set()
        if not hasattr(self, "_scalp_entries"):
            # sym -> ScalpSet entry orders submitted today (churn cap)
            self._scalp_entries: dict[str, int] = {}
        if not hasattr(self, "_scalp_entries_day"):
            self._scalp_entries_day: str = ""
        if not hasattr(self, "_scalp_cap_warned"):
            self._scalp_cap_warned: set[str] = set()

    # ── Entry accounting / fill resolution (2026-09-16 fix) ──────────
    def _scalp_entries_today(self, sym: str) -> int:
        """Number of ScalpSet entries submitted for *sym* today.

        The counter is keyed to the UTC trading day (the session runs
        13:30-20:00 UTC, so a UTC date rollover never lands mid-session) and
        is cleared lazily on the first call of a new day.
        """
        self._scalp_init_state()
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._scalp_entries_day != day:
            self._scalp_entries_day = day
            self._scalp_entries.clear()
            self._scalp_cap_warned.clear()
        return int(self._scalp_entries.get(sym.upper(), 0))

    def _scalp_entry_cap_reached(self, sym: str) -> bool:
        """True when *sym* has used its per-session ScalpSet entry budget."""
        cap = SCALP_MAX_ENTRIES_PER_SYMBOL_PER_SESSION
        if cap is None or cap <= 0:
            return False   # cap disabled
        return self._scalp_entries_today(sym) >= cap

    def _log_entry_cap_reached(self, sym: str) -> None:
        """Log the churn-cap skip once per symbol/day (debug afterwards)."""
        self._scalp_init_state()
        sym = sym.upper()
        if sym in self._scalp_cap_warned:
            logger.debug("🚧 SCALP %s: entry cap reached — signal skipped", sym)
            return
        self._scalp_cap_warned.add(sym)
        logger.info(
            "🚧 SCALP %s: session entry cap reached (%d/%d entries today) — "
            "skipping further signals for this symbol until the next session",
            sym, self._scalp_entries_today(sym),
            SCALP_MAX_ENTRIES_PER_SYMBOL_PER_SESSION,
        )

    def _record_scalp_entry(self, sym: str) -> None:
        """Count a submitted entry against the per-session churn cap."""
        self._scalp_init_state()
        sym = sym.upper()
        self._scalp_entries[sym] = self._scalp_entries_today(sym) + 1
        logger.info(
            "🧮 SCALP %s: entry %d/%d this session%s",
            sym, self._scalp_entries[sym], SCALP_MAX_ENTRIES_PER_SYMBOL_PER_SESSION,
            " (cap reached — no further entries today)"
            if self._scalp_entry_cap_reached(sym) else "",
        )

    async def _resolve_fill_price(self, sym: str, result, fallback: float) -> float:
        """Return the price the entry ACTUALLY filled at (2026-09-16 fix).

        Anchoring SL/TP to a signal-time reference is what produced stops on
        the wrong side of the market, so the entry's fill is resolved from
        the broker, in order of truth:

        1. ``OrderResult.filled_avg_price`` — the broker's own fill report;
        2. the open position's ``avg_entry_price`` for the symbol;
        3. *fallback* (the signal-time price) as a last resort, logged loudly
           because it is exactly the stale reference this fix guards against.
        """
        sym = sym.upper()
        filled = getattr(result, "filled_avg_price", None)
        if not isinstance(filled, bool) and isinstance(filled, (int, float)) and filled > 0:
            return float(filled)
        try:
            positions = await self.broker.get_positions()
        except Exception as exc:
            logger.warning("Fill resolution %s: position fetch failed (%s)", sym, exc)
            positions = []
        for p in positions or []:
            try:
                if str(p.get("symbol", "")).upper() != sym:
                    continue
                avg = p.get("avg_entry_price")
            except Exception:
                continue
            if not isinstance(avg, bool) and isinstance(avg, (int, float)) and float(avg) > 0:
                return float(avg)
        logger.warning(
            "⚠️  SCALP %s: broker fill price unknown — anchoring SL/TP to the "
            "signal-time price %.2f (last resort)", sym, float(fallback),
        )
        return float(fallback)

    def _warn_no_broker_stop(self, sym: str, state: dict, entry: float) -> None:
        """Loud, rate-limited (default: per-minute) "no broker stop" warning.

        A position without a broker-level stop is protected only by the
        in-process SL, which exists only while THIS process lives — it must
        never pass silently.  The level reported here is the same one the
        broker stop would have used: anchored to the FILL, never the stale
        signal reference.
        """
        now = time.monotonic()
        last = float(state.get("no_stop_warn_ts") or 0.0)
        if now - last < SCALP_NO_STOP_WARN_SECONDS:
            return
        state["no_stop_warn_ts"] = now
        self._scalp_positions[sym] = state
        sl = state.get("sl")
        logger.warning(
            "🚨 SCALP %s: NO BROKER STOP — position entered at %.2f is protected "
            "ONLY by the in-process SL %s (process-local). Anchored level in use; "
            "see the stop-placement log lines above.",
            sym, float(entry), f"${float(sl):.2f}" if sl is not None else "NONE",
        )

    # ── Data assembly ────────────────────────────────────────────────

    async def _fetch_tf(self, symbol: str, tf_key: str) -> pd.DataFrame | None:
        """Fetch (or return cached) bars for *symbol*/*tf_key*.

        Timeframes are lazy-fetched with per-TF TTLs from ``SCALP_TF_SPEC``
        so the data provider is not hammered every tick; ``1m`` is always
        fresh.  Indexes are localised to the exchange timezone (naive ET)
        so session windows line up with clock times —
        ``CandleFrame.from_dataframe`` converts tz-aware indexes to UTC,
        which would shift every session window by -4/-5 hours.
        """
        self._scalp_init_state()
        interval, lookback_days, ttl = SCALP_TF_SPEC[tf_key]
        if ttl is None:
            ttl = SCALP_CACHE_1M_SECONDS
        key = (symbol.upper(), tf_key)
        now_mono = time.monotonic()
        cached = self._frame_cache.get(key)
        if cached is not None and (now_mono - cached[0]) < ttl:
            return cached[1]

        now = datetime.now(timezone.utc)
        try:
            mdf = await self.provider.fetch_bars(
                symbol, start=now - timedelta(days=lookback_days), end=now,
                timeframe=interval,
            )
        except Exception as exc:
            logger.warning("SCALP DATA %s %s fetch failed: %s", symbol, tf_key, exc)
            self._frame_cache[key] = (now_mono, None)
            return None
        df = mdf.df
        if df is None or df.empty:
            self._frame_cache[key] = (now_mono, None)
            return None
        df = df.copy()
        if df.index.tz is not None:
            try:
                df.index = df.index.tz_convert(SCALP_EXCHANGE_TZ).tz_localize(None)
            except Exception:
                df.index = df.index.tz_localize(None)
        self._frame_cache[key] = (now_mono, df)
        logger.log(
            logging.DEBUG, "SCALP DATA %s %s: %d bars (latest %s)",
            symbol, tf_key, len(df), df.index[-1],
        )
        return df

    def _resample_4h(self, df1h: pd.DataFrame) -> pd.DataFrame | None:
        """Derive a 4H frame by resampling a 1H frame (yfinance has no 4h).

        The 4H frame is only consumed by the IFVG module's HTF-bias EMA(10),
        so a deterministic resample is a faithful approximation.
        """
        if df1h is None or len(df1h) < 4:
            return None
        out = df1h.resample("4h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        return out if not out.empty else None

    def _build_liquidity(self, symbol: str, frames: dict) -> LiquidityMap:
        """Build the LiquidityMap from the day/week frames.

        PDH/PDL and previous-day high/low come from the previous trading
        day's 1D bar; Asia/London extremes come from the previous day's 1m
        bars in the configured clock windows (RTH-centric US bars often
        leave these ``None`` — the modules degrade gracefully); today's
        in-progress RTH session high/low come from the current day's 1m
        bars.
        """
        liq = LiquidityMap()
        f1d = frames.get("1d")
        if f1d is not None and len(f1d) >= 2:
            idx = len(f1d) - 2  # last-but-one daily bar = previous session
            liq = LiquidityMap(
                pdh=float(f1d.high[idx]),
                pdl=float(f1d.low[idx]),
                prev_day_high=float(f1d.high[idx]),
                prev_day_low=float(f1d.low[idx]),
            )
        f1m = frames.get("1m")
        if f1m is not None and len(f1m) > 0:
            today = pd.Timestamp(f1m.ts[-1]).normalize()
            mask = f1m.ts >= np.datetime64(today)
            if mask.any():
                liq = LiquidityMap(
                    **{**liq.__dict__,
                       "session_high": float(f1m.high[mask].max()),
                       "session_low": float(f1m.low[mask].min())},
                )
            prev_mask = f1m.ts < np.datetime64(today)
            if prev_mask.any():
                prev_day_frame = CandleFrame(
                    symbol=symbol, timeframe="1m",
                    ts=f1m.ts[prev_mask], open=f1m.open[prev_mask],
                    high=f1m.high[prev_mask], low=f1m.low[prev_mask],
                    close=f1m.close[prev_mask], volume=f1m.volume[prev_mask],
                )
                for w in SCALP_LIQUIDITY_WINDOWS:
                    hi, lo = session_extremes(prev_day_frame, w)
                    if hi is None or lo is None:
                        continue
                    if w.name == "asia":
                        liq = LiquidityMap(**{**liq.__dict__, "asia_high": hi, "asia_low": lo})
                    elif w.name == "london":
                        liq = LiquidityMap(**{**liq.__dict__, "london_high": hi, "london_low": lo})
        return liq

    async def _build_scalp_context(self, symbol: str) -> ScalpContext | None:
        """Assemble the ScalpContext for *symbol* (frames + liquidity + pair)."""
        self._scalp_init_state()
        frames: dict[str, CandleFrame] = {}
        for tf_key in ("1m", "5m", "15m", "30m", "1h", "4h", "1d"):
            df = await self._fetch_tf(symbol, tf_key)
            if df is None or df.empty:
                continue
            if tf_key == "4h":
                df = self._resample_4h(df)
                if df is None:
                    continue
            frames[tf_key] = CandleFrame.from_dataframe(symbol.upper(), tf_key, df)
        if "1m" not in frames:
            return None
        pair_frames: dict[str, CandleFrame] = {}
        pair = SCALP_PAIR_SYMBOL
        if pair != symbol.upper():
            pdf = await self._fetch_tf(pair, "1m")
            if pdf is not None and not pdf.empty:
                pair_frames["1m"] = CandleFrame.from_dataframe(pair, "1m", pdf)
        else:
            pair_frames["1m"] = frames["1m"]
        liq = self._build_liquidity(symbol, frames)
        return ScalpContext(
            symbol=symbol.upper(),
            frames=frames,
            liquidity=liq,
            tz=SCALP_EXCHANGE_TZ,
            pair_frames=pair_frames,
            pair_symbol=pair,
        )

    # ── Data-availability gate ───────────────────────────────────────

    def _scalp_data_ready(self, symbol: str, ctx: ScalpContext) -> bool:
        """True when every ENABLED module has enough bars for its lookbacks.

        Fail-safe by design: on insufficient data we log ONE loud warning
        per symbol per trading day and skip signals — we never crash, and
        modules resume as their bars accumulate intraday.
        """
        self._scalp_init_state()
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo(SCALP_EXCHANGE_TZ))
        day = now_et.strftime("%Y-%m-%d")
        if day != self._scalp_data_day:
            self._scalp_data_day = day
            self._scalp_data_warned = set()
            self._scalp_ready_logged = set()
        specs = []
        if SCALP_MODULE_IFVG:
            specs.append("ict_ifvg")
        if SCALP_MODULE_BOX:
            specs.append("box_theory")
        if SCALP_MODULE_VOLFIB:
            specs.append("volprofile_fib")
        shortfall: list[str] = []
        for key in specs:
            for tf_key, need in SCALP_MIN_BARS[key].items():
                frame = ctx.frames.get(tf_key)
                have = len(frame) if frame is not None else 0
                if have < need:
                    shortfall.append(f"{tf_key}={have}<{need}")
        if not shortfall:
            if symbol.upper() not in self._scalp_ready_logged:
                self._scalp_ready_logged.add(symbol.upper())
                logger.info(
                    "SCALP DATA %s: enough bars for all enabled modules — signals ON",
                    symbol,
                )
            return True
        sym = symbol.upper()
        if sym not in self._scalp_data_warned:
            self._scalp_data_warned.add(sym)
            logger.warning(
                "⚠️  SCALP DATA %s: insufficient bars — %s — skipping signals "
                "until enough history exists (never crashes; resumes intraday)",
                symbol, ", ".join(shortfall),
            )
        return False

    # ── Evaluation + arbitration ─────────────────────────────────────

    def _evaluate_scalp_modules(self, ctx: ScalpContext) -> list[ScalpSignal]:
        """Run every enabled module against *ctx* and return all signals."""
        out: list[ScalpSignal] = []
        if SCALP_MODULE_IFVG:
            try:
                out.extend(scalp_ict_ifvg.evaluate(ctx))
            except Exception as exc:
                logger.error("SCALP %s: ict_ifvg evaluation crashed: %s", ctx.symbol, exc)
        if SCALP_MODULE_BOX:
            try:
                out.extend(scalp_box_theory.evaluate(ctx))
            except Exception as exc:
                logger.error("SCALP %s: box_theory evaluation crashed: %s", ctx.symbol, exc)
        if SCALP_MODULE_VOLFIB:
            try:
                out.extend(scalp_volprofile_fib.evaluate(ctx))
            except Exception as exc:
                logger.error("SCALP %s: volprofile_fib evaluation crashed: %s", ctx.symbol, exc)
        return out

    def _arbitrate_scalp_signal(self, symbol: str, signals: list[ScalpSignal]) -> ScalpSignal | None:
        """Pick ONE signal per symbol per tick via the arbitration rule.

        ``best_rr`` (default): highest R:R, ties broken in module order
        (IFVG > Box > VolFib).  ``first``: module order wins regardless of
        R:R.  Logs the arbitration when multiple modules agree/disagree.
        """
        if not signals:
            return None
        if len(signals) == 1:
            return signals[0]
        mode = SCALP_ARBITRATION if SCALP_ARBITRATION in ("best_rr", "first") else "best_rr"
        if mode == "first":
            chosen = min(signals, key=lambda s: SCALP_MODULE_ORDER.index(s.strategy))
        else:
            chosen = min(signals, key=lambda s: (-s.rr, SCALP_MODULE_ORDER.index(s.strategy)))
        summary = ", ".join(
            f"{s.strategy}:{s.direction.value}@{s.rr:.2f}R"
            for s in sorted(signals, key=lambda s: SCALP_MODULE_ORDER.index(s.strategy))
        )
        logger.info(
            "⚖️  ARBITRATION %s: %d module(s) signaled [%s] — picked %s (%s %.2fR)",
            symbol, len(signals), summary, chosen.strategy, chosen.direction.value, chosen.rr,
        )
        return chosen

    def _signal_key(self, s: ScalpSignal) -> str:
        """Identity of a signal setup (dedupe key across ticks)."""
        return (
            f"{s.strategy}|{s.direction.value}|{s.entry_type.value}|"
            f"{s.entry_price:.4f}|{s.stop_loss:.4f}|{s.take_profit:.4f}"
        )

    def _log_signal(self, s: ScalpSignal) -> None:
        """One-line per-signal log: module, symbol, side, entry, SL, TP, R:R."""
        logger.info(
            "📡 SIGNAL %s %s: %s %s @ %s entry=%.2f SL=%.2f TP=%.2f R:R=%.2f",
            s.symbol, s.strategy, s.direction.value, s.entry_type.value,
            "market" if s.entry_type == EntryType.MARKET else "limit",
            s.entry_price, s.stop_loss, s.take_profit, s.rr,
        )

    # ── Short capability / rejection backoff (turbo #29 lessons) ─────

    async def _short_capability_allows(self, symbol: str, account: dict) -> bool:
        """Gate a short attempt behind account + instrument capability.

        False (with a clear log) when the account reports shorting disabled
        or the broker flags the symbol NOT shortable — longs continue for
        that symbol either way.  A failed lookup (``None``) is unknown →
        allowed, so a transient failure never permanently blocks.  Results
        are cached per symbol for the session (False sticks, None retries).
        """
        self._scalp_init_state()
        sym = symbol.upper()
        account_flag = account.get("shorting_enabled") if isinstance(account, dict) else None
        if account_flag is False:
            logger.warning("SHORT %s skipped — account reports shorting disabled", symbol)
            return False
        if self._shortable_cache.get(sym) is False:
            logger.info("SHORT %s skipped — broker flags symbol NOT shortable (cached)", symbol)
            return False
        if not MAIN_SHORT_REQUIRE_SHORTABLE:
            return True
        is_shortable = getattr(self.broker, "is_shortable", None)
        if is_shortable is None:
            return True
        try:
            shortable = await is_shortable(symbol)
        except Exception as exc:
            logger.debug("SHORT %s: shortability lookup errored (%s) — attempting anyway", symbol, exc)
            return True
        if shortable is False:
            self._shortable_cache[sym] = False
            logger.warning(
                "SHORT %s skipped — broker flags symbol NOT SHORTABLE "
                "(Alpaca shortable=False; e.g. leveraged ETFs SOXL/TZA/LABD). "
                "Longs for %s continue unaffected.",
                symbol, symbol,
            )
            return False
        if shortable is True:
            self._shortable_cache[sym] = True
        return True

    def _record_short_rejection(self, symbol: str, error_message: str | None) -> None:
        """Count a short rejection; disable the symbol's shorts past the limit."""
        self._scalp_init_state()
        sym = symbol.upper()
        kind = _rejection_kind(error_message)
        count = self._short_rejections.get(sym, 0) + 1
        self._short_rejections[sym] = count
        logger.warning(
            "❌ SHORT %s REJECTED (%s — rejection %d/%d this session) — %s",
            symbol, kind, count, MAIN_SHORT_DISABLE_AFTER,
            error_message or "no broker message",
        )
        if kind == "short_not_allowed":
            self._shortable_cache[sym] = False  # definitive verdict sticks
        if count >= MAIN_SHORT_DISABLE_AFTER and sym not in self._shorts_disabled:
            self._shorts_disabled.add(sym)
            logger.warning(
                "⛔ SHORT %s disabled for the rest of the session after %d rejections "
                "(avoids tight retry loop); longs unaffected",
                symbol, count,
            )

    # ── Entry execution ──────────────────────────────────────────────

    async def _scalp_enter(self, signal: ScalpSignal, current_price: float) -> bool:
        """Execute one arbitrated ScalpSignal (MARKET or LIMIT).

        MARKET → order now at the market, track the position, attach the
        strategy-SL GTC stop + day-limit TP.  LIMIT → rest a DAY limit at
        ``signal.entry_price`` (dies at close — no overnight limit risk) and
        record a pending bundle; the per-tick position sync detects the fill
        and attaches the stop/TP then.  Returns True when an order was
        placed (not necessarily filled).
        """
        self._scalp_init_state()
        sym = signal.symbol.upper()
        if self.pm.has_position(sym) or sym in self._scalp_bundles:
            return False
        if time.monotonic() < self._cooldown_until.get(sym, 0.0):
            logger.debug("🕐 SCALP %s: cooldown active — skipping new entry", sym)
            return False
        # Re-entry churn cap: a setup that keeps re-emitting after every
        # stop-out must not loop all session (the live COIN 2026-09-16 case).
        if self._scalp_entry_cap_reached(sym):
            self._log_entry_cap_reached(sym)
            return False
        account = await self.broker.get_account()
        equity = account_equity(account)
        if equity is None or equity <= 0:
            logger.warning("⚠️  %s: account equity unavailable — skipping scalp entry", sym)
            return False
        if not self.pm.can_open(sym, equity):
            return False
        bp = _account_buying_power(account)
        is_short = signal.direction == Direction.SHORT
        if is_short:
            if not await self._short_capability_allows(sym, account):
                return False
            if sym in self._shorts_disabled:
                logger.info(
                    "SHORT %s skipped — shorts disabled this session after %d rejection(s)",
                    sym, self._short_rejections.get(sym, 0),
                )
                return False
        if bp is None:
            logger.warning("⚠️  %s: buying power unavailable — skipping scalp entry", sym)
            return False
        entry_ref = current_price if signal.entry_type == EntryType.MARKET else signal.entry_price
        if entry_ref <= 0:
            return False
        qty, bp_capped = _size_entry_qty(
            equity=equity, buying_power=bp, price=entry_ref,
            size_pct=POSITION_SIZE_PCT,
            whole_shares=(is_short and MAIN_SHORT_WHOLE_SHARES),
        )
        if qty < 1:
            logger.warning(
                "⚠️  %s: scalp qty %.2f < 1 share — skipping entry%s",
                sym, qty, " (BP-capped)" if bp_capped else "",
            )
            return False
        side = OrderSide.SELL if is_short else OrderSide.BUY
        client_id = _main_order_client_id(sym, side.value, "ENTRY")
        order_type = OrderType.MARKET if signal.entry_type == EntryType.MARKET else OrderType.LIMIT
        limit_price = (None if order_type == OrderType.MARKET
                       else _normalize_order_price(signal.entry_price))
        order = Order(
            symbol=sym, side=side, quantity=qty, order_type=order_type,
            limit_price=limit_price, client_id=client_id,
        )
        self._log_signal(signal)
        try:
            result = await self.broker.place_order(order)
        except Exception as exc:
            logger.error("SCALP %s entry submission failed: %s", sym, exc)
            if is_short:
                self._record_short_rejection(sym, str(exc))
            return False
        if not is_order_alive(result.status):
            err = getattr(result, "error_message", None) or result.status
            logger.warning("❌ SCALP %s %s REJECTED: %s", sym, side.value, err)
            if is_short:
                self._record_short_rejection(sym, str(err))
            return False
        self._last_signal_key[sym] = self._signal_key(signal)
        self._record_scalp_entry(sym)
        if order_type == OrderType.MARKET:
            # Anchor every SL/TP level to the ACTUAL fill, never the
            # signal-time reference — MARKET fills land points away from the
            # signal price on fast tape, which is how the stop ended up on
            # the wrong side of the market (live COIN 2026-09-16).
            fill_price = await self._resolve_fill_price(
                sym, result,
                current_price if current_price > 0 else signal.entry_price,
            )
            self.pm.open_position(
                sym, -qty if is_short else qty, fill_price,
                stop_loss_price=signal.stop_loss,
                take_profit_price=signal.take_profit,
            )
            self._entry_times[sym] = datetime.now(timezone.utc)
            logger.info(
                "📈 %s %s: %s @ $%.2f | qty=%.2f (%s%d%% equity%s) | strategy=%s",
                "SHORT" if is_short else "BUY", sym,
                "MARKET" if order_type == OrderType.MARKET else "LIMIT",
                fill_price, qty, "-" if is_short else "", POSITION_SIZE_PCT * 100,
                " BP-capped" if bp_capped else "", signal.strategy,
            )
            await self._finalize_open_bundle(sym, qty, fill_price, signal)
        else:
            self._scalp_bundles[sym] = {
                "qty": qty, "direction": "SHORT" if is_short else "LONG",
                "sl": signal.stop_loss, "tp": signal.take_profit,
                "strategy": signal.strategy, "entry_price": signal.entry_price,
            }
            logger.info(
                "⏳ %s %s: day-limit %s at $%.2f resting (dies at close) | qty=%.2f%s | strategy=%s",
                "SHORT" if is_short else "BUY", sym,
                "SHORT" if is_short else "LONG", signal.entry_price, qty,
                " BP-capped" if bp_capped else "", signal.strategy,
            )
        return True

    async def _finalize_open_bundle(
        self, sym: str, qty: float, fill_price: float, sig: ScalpSignal,
    ) -> None:
        """Attach the ANCHORED-SL GTC stop + day-limit TP to a live position.

        Caller has already recorded the position in the PositionManager with
        NEGATIVE qty for shorts.

        SL/TP ANCHORING (post-go-live fix, 2026-09-16): the broker validates a
        stop against the LIVE market price, so the signal-time levels are
        re-anchored to the CONFIRMED fill (``pos.entry_price``) here.  A
        strategy level that is still valid relative to the fill is kept
        verbatim; otherwise it is re-priced from the fill using the signal's
        risk/reward distance, clamped to a minimum distance, and — when even
        that is degenerate — replaced by the -6% backstop with a loud log
        ("strategy SL invalid vs fill — using backstop").  The SAME levels are
        stored in the position state, so the in-process SL evaluates the
        strategy stop relative to the FILL, not the stale reference.
        """
        sym = sym.upper()
        pos = self.pm.get_positions().get(sym)
        if pos is None:
            return
        is_short = sig.direction == Direction.SHORT
        fill = float(pos.entry_price) or float(fill_price or 0.0)
        if SCALP_ANCHOR_LEVELS:
            levels = anchor_scalp_levels(
                is_short=is_short, fill=fill, entry_ref=sig.entry_price,
                sl_ref=sig.stop_loss, tp_ref=sig.take_profit,
            )
        else:  # env kill-switch: pre-fix behaviour (raw signal levels)
            levels = AnchoredLevels(
                sl=sig.stop_loss, tp=sig.take_profit,
                sl_source="strategy", tp_source="strategy",
            )
        if levels.sl_reanchored or levels.tp_reanchored:
            logger.warning(
                "⚓ SCALP %s: signal levels stale vs fill %.2f "
                "(entry_ref=%.2f SL_ref=%.2f TP_ref=%.2f) — re-anchored to "
                "SL=%s TP=%s (min distance %.4f)",
                sym, fill, float(sig.entry_price), float(sig.stop_loss),
                float(sig.take_profit),
                f"{levels.sl:.2f}" if levels.sl is not None else "BACKSTOP",
                f"{levels.tp:.2f}" if levels.tp is not None else "none",
                levels.min_distance,
            )
        if levels.sl_is_backstop:
            logger.error(
                "🚨 SCALP %s: strategy SL invalid vs fill — using backstop "
                "(fill=%.2f SL_ref=%.2f TP_ref=%.2f entry_ref=%.2f, backstop "
                "%.0f%% -> $%s). Strategy risk NOT honoured on this position.",
                sym, fill, float(sig.stop_loss), float(sig.take_profit),
                float(sig.entry_price), PROTECTIVE_STOP_PCT * 100,
                f"{levels.sl:.2f}" if levels.sl is not None else "NONE",
            )
        elif levels.sl_source == "no_strategy_sl":
            logger.warning(
                "🛡️  SCALP %s: signal carries no SL — using the %.0f%% backstop "
                "from the fill (%.2f -> %s)",
                sym, PROTECTIVE_STOP_PCT * 100, fill,
                f"${levels.sl:.2f}" if levels.sl is not None else "NONE",
            )
        if levels.sl is None:
            logger.error(
                "🚨 SCALP %s: no valid stop level derivable from fill %.2f — "
                "position runs on the in-process risk pass only", sym, fill,
            )
        # Keep the PositionManager record in step with the broker levels: the
        # in-process risk pass reads the scalp state, while this record feeds
        # reports/teardown, and both must reflect the anchored geometry.
        pos.stop_loss_price = levels.sl
        pos.take_profit_price = levels.tp
        stop_ok = await self._place_protective_stop(
            sym, abs(pos.quantity), pos.entry_price,
            is_short=is_short, stop_price=levels.sl,
        )
        state = self._scalp_positions.get(sym, {})
        state.update({
            "entry": pos.entry_price, "sl": levels.sl, "tp": levels.tp,
            "sl_source": levels.sl_source, "tp_source": levels.tp_source,
            "direction": "SHORT" if is_short else "LONG",
            "be_done": False,
            "be_trigger_r": sig.breakeven_trigger_r,
            "be_buffer": sig.breakeven_buffer,
            "trailing": sig.trailing,
            "trail_r": sig.trail_distance_r,
            "trail_trigger_r": sig.trail_trigger_r,
            "stop_order_id": None,
            "stop_placed": stop_ok,
            "tp_placed": False,
            "no_stop_warn_ts": None,
        })
        self._scalp_positions[sym] = state
        # ── Exit structure (2026-09-22) ──────────────────────────────
        # A ground-up fix for the live defect where NO upside exit existed:
        # the GTC stop reserves the position's whole shares, Alpaca rejects
        # any second order for the same shares (40310000), and the rejection
        # path used to NULL the TP level out of the state — leaving the
        # position with a stop and nothing else.  The level is now ALWAYS
        # kept: the plan below decides whether the upside exit rests at the
        # broker or is owned by the trader-side monitor.
        plan = plan_exit_structure(
            abs(pos.quantity), stop_placed=stop_ok, tp=levels.tp,
        )
        state["stop_qty"] = plan.stop_qty
        state["tp_monitored"] = plan.monitored_tp
        state["tp_plan_reason"] = plan.reason
        self._scalp_positions[sym] = state
        logger.info("🧩 SCALP %s: exit structure — %s", sym, format_exit_plan(plan))
        if (SCALP_TP_DAY_LIMIT and levels.tp is not None
                and plan.broker_tp_qty > 0):
            tp_ok = await self._place_tp_limit(
                sym, plan.broker_tp_qty, levels.tp, is_short=is_short,
            )
            state["tp_placed"] = tp_ok
            if tp_ok:
                state["tp_monitored"] = False
            else:
                logger.warning(
                    "🎯 SCALP %s: broker TP order REJECTED — the upside exit "
                    "at $%.2f is now TRADER-SIDE MONITORED for the full %.6f "
                    "shares (the GTC stop stays at the broker)",
                    sym, float(levels.tp), abs(pos.quantity),
                )
        elif levels.tp is not None:
            logger.warning(
                "🎯 SCALP %s: upside exit at $%.2f is TRADER-SIDE MONITORED "
                "for the full %.6f shares — %s (the GTC stop stays at the "
                "broker; monitor cancels it, closes, and re-protects on "
                "failure)",
                sym, float(levels.tp), abs(pos.quantity), plan.reason,
            )
        self._scalp_positions[sym] = state

    async def _place_tp_limit(
        self, symbol: str, qty: float, tp_price: float, is_short: bool = False,
    ) -> bool:
        """Place a DAY take-profit limit at *tp_price* (dies at market close).

        No overnight limit risk by construction.  Booked P&L happens ONLY
        through the verified-close machinery: when the limit fills, the
        per-tick position sync sees the position vanish and books the P&L
        at the broker's fill price.
        """
        sym = symbol.upper()
        side = OrderSide.BUY if is_short else OrderSide.SELL
        order = Order(
            symbol=sym, side=side, quantity=qty, order_type=OrderType.LIMIT,
            # Tick-normalised: a raw 211.160004 is rejected outright by
            # Alpaca ("sub-penny increment", 42210000) and the position then
            # runs with no take-profit order at all (live 2026-09-16).
            limit_price=_normalize_order_price(tp_price),
            client_id=_main_order_client_id(sym, side.value, "TP"),
        )
        try:
            result = await self.broker.place_order(order)
        except Exception as exc:
            logger.warning("🎯  TP %s placement failed: %s", sym, exc)
            return False
        if not is_order_alive(result.status):
            msg = getattr(result, "error_message", None) or result.status
            logger.warning("🎯  TP %s REJECTED: %s", sym, msg)
            return False
        logger.info(
            "🎯  TP %s: day-limit %s at $%.2f x %.2f (order=%s)",
            sym, side.value, tp_price, qty, str(result.order_id)[:8],
        )
        return True

    async def _cancel_stop_only(self, symbol: str) -> bool:
        """Cancel ONLY the GTC stop for *symbol* (TP/entry limits untouched).

        Used by BE/trail replacement so the resting day-limit TP survives a
        stop move.  Returns True only when every open STOP order for the
        symbol is confirmed gone (never leave double stops: the replacement
        is only placed after this confirms).
        """
        sym = symbol.upper()
        try:
            open_orders = await self.broker.get_open_orders(symbol=sym)
        except Exception as exc:
            logger.warning("Failed to fetch open orders for %s: %s", sym, exc)
            return False
        stops = [o for o in open_orders if _order_is_stop(o)]
        if not stops:
            return True
        for o in stops:
            try:
                if not await self.broker.cancel_order_and_wait(str(o.id)):
                    logger.warning("Cancellation not confirmed for stop order %s", str(o.id)[:8])
                    return False
            except Exception as exc:
                logger.warning("Failed to cancel stop order %s: %s", str(o.id)[:8], exc)
                return False
        logger.info("🗑️  Cancelled %d stop order(s) for %s (TP untouched)", len(stops), sym)
        return True

    async def _replace_stop(self, sym: str, new_stop: float, reason: str) -> bool:
        """Cancel the symbol's GTC stop and place it at *new_stop*.

        Cancel-then-place with confirmation: a failed cancel leaves the OLD
        stop in place (still protecting) and the replacement is deferred —
        never two stops at once.
        """
        sym = sym.upper()
        pos = self.pm.get_positions().get(sym)
        if pos is None:
            return False
        is_short = pos.quantity < 0
        if not await self._cancel_stop_only(sym):
            logger.warning(
                "🛡️  STOP %s: %s replacement deferred — cancel not confirmed; old stop stays",
                sym, reason,
            )
            return False
        new_stop = round(float(new_stop), 2)
        ok = await self._place_protective_stop(
            sym, abs(pos.quantity), pos.entry_price,
            is_short=is_short, stop_price=new_stop,
        )
        if ok:
            state = self._scalp_positions.get(sym, {})
            state["sl"] = new_stop
            self._scalp_positions[sym] = state
            logger.info("🎗️  STOP %s moved to $%.2f (%s)", sym, new_stop, reason)
        return ok

    async def _scalp_risk_pass(self) -> None:
        """Per-tick risk management for live scalp positions.

        1. In-process fallback: if no broker stop is confirmed for a
           position, close via the verified-close path when price breaches
           the strategy SL (or TP when the TP order is absent).
        2. BE/trail: when the signal's rules trigger, cancel+replace the GTC
           stop to break-even / trailed level (never double stops), using
           the verified stop machinery.
        """
        self._scalp_init_state()
        main_set = {s.upper() for s in SYMBOLS}
        for sym in list(self.pm.get_open_symbols()):
            if sym not in main_set:
                continue  # never manage non-main symbols (turbo's positions)
            if not self.pm.has_position(sym):
                continue
            state = self._scalp_positions.get(sym)
            if state is None:
                continue
            try:
                mdf = await self.provider.fetch_bars(
                    sym,
                    start=datetime.now(timezone.utc) - timedelta(minutes=5),
                    end=datetime.now(timezone.utc),
                    timeframe="1min",
                )
                if mdf.df.empty:
                    continue
                price = float(mdf.df["close"].iloc[-1])
            except Exception:
                continue
            pos = self.pm.get_positions().get(sym)
            if pos is None:
                continue
            is_short = pos.quantity < 0
            entry = pos.entry_price
            # A position without a broker-level stop must never pass
            # silently: the in-process SL is the only protection left and it
            # lives only as long as this process.
            if not state.get("stop_placed", False):
                self._warn_no_broker_stop(sym, state, entry)
            cur_sl = state.get("sl")
            if cur_sl is None:
                continue
            risk = abs(entry - cur_sl)
            if risk <= 0:
                continue
            profit_r = ((entry - price) / risk) if is_short else ((price - entry) / risk)
            # ── In-process SL / TP fallback (broker stop/TP missing) ──
            if not state.get("stop_placed", False):
                breached = (price <= cur_sl) if not is_short else (price >= cur_sl)
                if breached:
                    logger.warning(
                        "🚨 SCALP %s: in-process SL $%.2f breached (no broker stop) — closing",
                        sym, cur_sl,
                    )
                    await self._scalp_close_position(sym, price, "scalp_sl_inproc")
                    continue
            tp = state.get("tp")
            if tp is not None and not state.get("tp_placed", False):
                hit = (price >= tp) if not is_short else (price <= tp)
                if hit:
                    logger.info(
                        "🎯 SCALP %s: MONITORED TP $%.2f hit (price %.2f, no "
                        "resting TP order — the GTC stop reserves the shares) "
                        "— cancelling the stop and closing the full position",
                        sym, tp, price,
                    )
                    await self._scalp_close_position(sym, price, "scalp_tp_inproc")
                    continue
            # ── BE / trail ──
            be_done = state.get("be_done", False)
            be_trigger = float(state.get("be_trigger_r", 0.0) or 0.0)
            trailing = bool(state.get("trailing"))
            trail_r = float(state.get("trail_r", 0.0) or 0.0)
            trail_trigger = float(state.get("trail_trigger_r", 0.0) or 0.0)
            if not be_done and be_trigger > 0 and profit_r >= be_trigger:
                buffer = float(state.get("be_buffer", 0.0) or 0.0)
                be = (entry - buffer) if is_short else (entry + buffer)
                if (not is_short and be > cur_sl) or (is_short and be < cur_sl):
                    if await self._replace_stop(sym, be, "break-even"):
                        state = self._scalp_positions[sym]
                        state["be_done"] = True
                        self._scalp_positions[sym] = state
                        continue
            if trailing and trail_r > 0 and profit_r >= trail_trigger:
                new_sl = (price + trail_r * risk) if is_short else (price - trail_r * risk)
                cur_sl = state.get("sl")
                better = (not is_short and new_sl > cur_sl) or (is_short and new_sl < cur_sl)
                if better and abs(new_sl - cur_sl) / max(cur_sl, 1e-9) >= 0.001:
                    await self._replace_stop(sym, new_sl, f"trail (R={profit_r:.2f})")

    async def _scalp_close_position(self, sym: str, price: float, reason: str) -> bool:
        """Cancel the symbol's orders and close via the verified-close path.

        The ONLY way scalp P&L is booked is through close_position_verified
        (or a broker fill detected by the position sync).  Sets the entry
        cooldown after a confirmed close.
        """
        sym = sym.upper()
        if not self.pm.has_position(sym):
            self._scalp_cleanup_state(sym)
            return True
        # Cancel ALL of the symbol's orders (stop + TP) before closing so
        # no GTC stop outlives the position (orphaned-stop protection).
        cancels = await self._cancel_protective_stops(sym)
        if not cancels:
            logger.warning(
                "SCALP %s: %s close deferred — order cancellation not confirmed", sym, reason,
            )
            return False
        outcome = await close_position_verified(
            self.pm, self.broker, sym,
            exit_reason=reason,
            client_id=_main_order_client_id(sym, "SELL", "EXIT"),
        )
        if outcome.status == "filled" and outcome.trade is not None:
            self._entry_times.pop(sym, None)
            self._cooldown_until[sym] = time.monotonic() + SCALP_COOLDOWN_BARS * 60.0
            logger.info(
                "📉 EXIT %s (%s): %.2f → %.2f P&L=$%.2f | cooldown %dmin",
                sym, reason, outcome.trade.entry_price, outcome.fill_price,
                outcome.trade.pnl, SCALP_COOLDOWN_BARS,
            )
            self._scalp_cleanup_state(sym)
            return True
        if outcome.status == "pending":
            logger.info("⏳ EXIT %s (%s): close pending — P&L deferred to position sync", sym, reason)
            # The stop was cancelled to free the shares for this close.  If
            # the close only PARTIALLY filled, what is still held is naked —
            # re-protect the residual before returning (idempotent).
            await self._reprotect_residual(sym, f"{reason} close pending")
            return True
        logger.warning("EXIT %s (%s) rejected (%s) — restoring protective stop", sym, reason, outcome.message)
        pos = self.pm.get_positions().get(sym)
        if pos is not None:
            state = self._scalp_positions.get(sym, {})
            await self._place_protective_stop(
                sym, abs(pos.quantity), pos.entry_price,
                is_short=pos.quantity < 0, stop_price=state.get("sl"),
            )
        return False

    async def _reprotect_residual(self, sym: str, reason: str) -> bool:
        """Re-place the protective stop for a PARTIAL-exit residual.

        Every exit path cancels the resting stop before closing (the broker
        will not let a close order co-exist with a stop on the same shares).
        If the close then only fills part of the position, the shares left
        behind are unprotected — this restores protection for whatever the
        broker still holds, at the same anchored level.

        Idempotent: ``_place_protective_stop`` returns early when a same-side
        stop already rests, so calling this after any exit attempt is safe
        (never two stops, never a naked residual).

        Returns True when the broker holds nothing (nothing to protect) or a
        stop is in place.
        """
        sym = sym.upper()
        try:
            positions = await self.broker.get_positions()
        except Exception as exc:
            logger.warning("Cannot verify residual protection for %s: %s", sym, exc)
            return False
        row = next(
            (p for p in positions if str(p.get("symbol", "")).upper() == sym), None,
        )
        if row is None:
            return True
        qty = float(row.get("qty", 0) or 0)
        if qty == 0:
            return True
        state = self._scalp_positions.get(sym, {})
        entry = float(row.get("avg_entry_price", 0) or 0) or float(state.get("entry") or 0)
        logger.warning(
            "🛡️  SCALP %s: residual %.6f shares still held after %s — the stop "
            "was cancelled for that exit, re-placing it now",
            sym, qty, reason,
        )
        return await self._place_protective_stop(
            sym, abs(qty), entry, is_short=qty < 0, stop_price=state.get("sl"),
        )

    def _scalp_cleanup_state(self, sym: str) -> None:
        self._scalp_init_state()
        sym = sym.upper()
        self._scalp_positions.pop(sym, None)
        self._scalp_bundles.pop(sym, None)
        self._last_signal_key.pop(sym, None)

    # ── Per-tick position sync (limit fills, TP fills, orphan cleanup) ─

    async def _scalp_sync_positions(self) -> None:
        """Reconcile broker positions for MAIN symbols with local state.

        Handles three events that only a broker sync can see:

        * a resting day-limit ENTRY filled → adopt the position, attach the
          bundle's strategy-SL stop + day-limit TP at the broker's actual
          fill qty/price;
        * a tracked position vanished (day-limit TP filled / external
          close) → cancel leftover orders (never leave an orphaned GTC
          stop), book P&L at the broker's fill price, start the cooldown;
        * an inherited position (no bundle) → adopt + default protective
          stop.

        Only MAIN symbols are touched — the account is shared with turbo and
        its leveraged-ETF positions are never managed here.
        """
        self._scalp_init_state()
        main_set = {s.upper() for s in SYMBOLS}
        try:
            broker_positions = await self.broker.get_positions()
        except Exception as exc:
            logger.warning("SCALP position sync failed: %s", exc)
            return
        by_sym = {str(p["symbol"]).upper(): p for p in broker_positions}
        pm_syms = set(self.pm.get_open_symbols())
        # ── Removed: closed somewhere (TP fill / external) ──
        for sym in list(pm_syms):
            if sym not in main_set:
                continue
            if sym in by_sym:
                continue
            fill_price = await self.broker.get_last_fill_price(sym)
            if fill_price is not None:
                self.pm.close_position(sym, exit_price=fill_price, exit_reason="scalp_sync_removed")
            else:
                self.pm.discard_position(sym, reason="scalp_sync_removed_no_fill")
            # Never leave an orphaned GTC stop after the position is gone.
            try:
                await self._cancel_protective_stops(sym)
            except Exception as exc:
                logger.warning("Could not cancel leftover orders for %s: %s", sym, exc)
            self._entry_times.pop(sym, None)
            self._cooldown_until[sym] = time.monotonic() + SCALP_COOLDOWN_BARS * 60.0
            self._scalp_cleanup_state(sym)
            logger.info(
                "👻 SCALP %s: position vanished from broker (fill=%s) — booked/cleaned, cooldown %dmin",
                sym, f"{fill_price:.2f}" if fill_price is not None else "UNKNOWN",
                SCALP_COOLDOWN_BARS,
            )
        # ── Added: limit-entry filled or inherited ──
        for sym, p in by_sym.items():
            if sym not in main_set:
                continue
            if sym in pm_syms:
                continue
            qty = float(p.get("qty", 0))
            price = float(p.get("avg_entry_price", 0)) or 0.0
            if qty == 0:
                continue
            bundle = self._scalp_bundles.pop(sym, None)
            if bundle is not None:
                if price <= 0:
                    price = bundle.get("entry_price") or 0.0
                signed = -qty if bundle["direction"] == "SHORT" else qty
                self.pm.open_position(
                    sym, signed, price,
                    stop_loss_price=bundle.get("sl"), take_profit_price=bundle.get("tp"),
                )
                self._entry_times[sym] = datetime.now(timezone.utc)
                logger.info(
                    "✅ SCALP %s: day-LIMIT %s filled — %s x %.2f @ $%.2f (bundle finalized)",
                    sym, bundle["direction"], bundle["direction"], qty, price,
                )
                sig = ScalpSignal(
                    symbol=sym, timestamp=pd.Timestamp(datetime.now(timezone.utc)),
                    direction=Direction.LONG if bundle["direction"] == "LONG" else Direction.SHORT,
                    entry_type=EntryType.LIMIT, entry_price=bundle["entry_price"],
                    stop_loss=bundle["sl"], take_profit=bundle["tp"],
                    risk=abs(bundle["entry_price"] - bundle["sl"]),
                    reward=abs(bundle["tp"] - bundle["entry_price"]),
                    rr=abs(bundle["tp"] - bundle["entry_price"]) / max(
                        abs(bundle["entry_price"] - bundle["sl"]), 1e-9),
                    strategy=bundle.get("strategy", "scalp"),
                )
                await self._finalize_open_bundle(sym, qty, price, sig)
            else:
                self.pm.open_position(sym, qty, price)
                self._entry_times[sym] = datetime.now(timezone.utc)
                logger.warning(
                    "🛡️  SCALP %s: inherited position %s x %.2f @ $%.2f — placing default protective stop",
                    sym, sym, qty, price,
                )
                await self._place_protective_stop(sym, int(qty), price, is_short=False)

        # ── Shrunk: a partial exit left a residual ────────────────────
        # A tracked position that lost shares on the broker (a partial fill
        # of a close order, or an external reduction) must not sit naked.  No
        # P&L is booked here — the exit accounting stays with the verified
        # close / vanish path — but protection is restored immediately, and
        # loudly, because a residual with no stop is exactly the failure this
        # trader must never allow.
        for sym in sorted(pm_syms & set(by_sym)):
            if sym not in main_set:
                continue
            tracked = float(self.pm.get_positions()[sym].quantity)
            broker_qty = float(by_sym[sym].get("qty", 0) or 0)
            if abs(broker_qty) >= abs(tracked) - 1e-9:
                continue
            logger.warning(
                "🚨 SCALP %s: broker holds %.6f but %.6f is tracked — a PARTIAL "
                "exit left a residual; re-protecting it before the next tick",
                sym, broker_qty, tracked,
            )
            await self._reprotect_residual(sym, "partial exit (position sync)")

        # ── Protection audit: a sync pass must never END with a naked
        # position (2026-09-22).  Warns loudly and attempts re-placement.
        await self._audit_position_protection(context="position sync")

    # ── Scalp tick loop ──────────────────────────────────────────────

    async def _tick_scalp(self, tick_num: int) -> None:
        """One polling cycle for the ScalpSet: frames → signals → execute."""
        self._scalp_init_state()
        now = datetime.now(timezone.utc)
        logger.debug("Tick %d: scalp evaluating %d symbols (%d positions open)",
                     tick_num, len(SYMBOLS), self.pm.get_open_count())

        for symbol in SYMBOLS:
            sym = symbol.upper()
            # One active setup per symbol: a live position or a resting
            # day-limit entry bundle blocks new signals.
            if self.pm.has_position(sym) or sym in self._scalp_bundles:
                continue
            if time.monotonic() < self._cooldown_until.get(sym, 0.0):
                continue
            if self._scalp_entry_cap_reached(sym):
                self._log_entry_cap_reached(sym)
                continue
            if self.pm.get_open_count() >= MAX_POSITIONS:
                continue
            try:
                ctx = await self._build_scalp_context(symbol)
                if ctx is None:
                    logger.debug("SCALP %s: no 1m frame — skipping tick", symbol)
                    continue
                if not self._scalp_data_ready(symbol, ctx):
                    continue
                signals = self._evaluate_scalp_modules(ctx)
                chosen = self._arbitrate_scalp_signal(symbol, signals)
                if chosen is None:
                    continue
                # Dedupe: the same setup re-emits every tick while its
                # conditions persist — place it only once.
                key = self._signal_key(chosen)
                if self._last_signal_key.get(sym) == key:
                    continue
                current_price = float(ctx.frames["1m"].close[-1])
                await self._scalp_enter(chosen, current_price)
            except Exception as exc:
                logger.error("SCALP %s: tick error — %s", symbol, exc)

        # Broker reconciliation (limit fills / TP fills / orphan cleanup)
        await self._scalp_sync_positions()
        # Risk management: BE/trail + in-process SL/TP fallback
        await self._scalp_risk_pass()

    def _scalp_reset_session_state(self) -> None:
        """Reset per-session scalp state (called at session end)."""
        self._scalp_init_state()
        self._frame_cache.clear()
        self._scalp_bundles.clear()
        self._scalp_positions.clear()
        self._cooldown_until.clear()
        self._last_signal_key.clear()
        self._shorts_disabled.clear()
        self._short_rejections.clear()
        self._shortable_cache.clear()
        self._scalp_data_warned.clear()
        self._scalp_ready_logged.clear()
        self._scalp_entries.clear()
        self._scalp_entries_day = ""
        self._scalp_cap_warned.clear()
    async def _check_risk_stops(self):
        """Check stop-loss / take-profit for open positions."""
        for symbol in list(self.pm.get_open_symbols()):
            if not self.pm.has_position(symbol):
                continue
            try:
                mdf = await self.provider.fetch_bars(
                    symbol,
                    start=datetime.now(timezone.utc) - timedelta(minutes=5),
                    end=datetime.now(timezone.utc),
                    timeframe="1min",
                )
                if mdf.df.empty:
                    continue
                price = float(mdf.df["close"].iloc[-1])
                pos = self.pm.get_positions().get(symbol.upper())
                if pos is None:
                    continue
                entry = pos.entry_price
                if entry == 0:
                    continue

                change_pct = (price - entry) / entry

                # ── Time-based exit: recycle capital after MAX_HOLD_MINUTES ──
                entry_time = self._entry_times.get(symbol.upper())
                held_minutes = 0.0
                if entry_time is not None:
                    held_minutes = (datetime.now(timezone.utc) - entry_time).total_seconds() / 60

                if held_minutes >= MAX_HOLD_MINUTES:
                    await self._handle_sell(symbol, price, 1.0)
                    logger.info(f"⏰ TIME-EXIT {symbol}: held {held_minutes:.0f}min, P&L={change_pct*100:+.1f}%")
                    continue

                if change_pct <= -STRATEGY_CONFIG.stop_loss_pct:
                    await self._handle_sell(symbol, price, 1.0)
                    logger.warning(f"🛑 STOP-LOSS {symbol}: -{abs(change_pct)*100:.1f}%")
                elif change_pct >= STRATEGY_CONFIG.take_profit_pct:
                    await self._handle_sell(symbol, price, 1.0)
                    logger.info(f"🎯 TAKE-PROFIT {symbol}: +{change_pct*100:.1f}%")
            except Exception as e:
                logger.error(f"Risk check error {symbol}: {e}")

    async def shutdown(self, close_broker: bool = True):
        """Close positions and report P&L; optionally retain broker for next session."""
        logger.info("Shutting down..." )

        # Close all positions
        positions = await self.broker.get_positions()
        for p in positions:
            sym = p.get("symbol")
            qty = float(p.get("qty", 0))
            if qty > 0:
                logger.info(f"Closing {sym}: {qty} shares...")
                # Cancel the GTC protective stop so it can't outlive the
                # position (best-effort on this path — shutdown is a last
                # resort; positions should already be flat from EOD).
                try:
                    await self._cancel_protective_stops(sym)
                except Exception:
                    logger.exception("Failed to cancel protective stop for %s during shutdown", sym)
                order = Order(symbol=sym, side=OrderSide.SELL, quantity=qty, order_type=OrderType.MARKET)
                await self.broker.place_order(order)

        # Final account state — NEVER fabricate an end equity.  A timed-out
        # fetch returns the unavailable sentinel (equity None); we then print
        # "UNKNOWN" instead of computing a bogus -100% from 0.0.
        account = await self.broker.get_account()
        end_equity = account_equity(account)
        start_equity = self.start_equity
        if start_equity is None or start_equity <= 0:
            persisted = load_start_equity()
            if persisted is not None:
                start_equity = persisted
        logger.info("=" * 60)
        logger.info(f"SESSION COMPLETE")
        if start_equity is None or start_equity <= 0:
            logger.info("Start: UNKNOWN (no baseline captured)")
        else:
            logger.info(f"Start: ${start_equity:,.2f}")
        if end_equity is None:
            logger.info("End:   UNKNOWN (account fetch failed)")
            logger.info("P&L:   UNKNOWN (account fetch timed out)")
        elif start_equity is None or start_equity <= 0:
            logger.info(f"End:   ${end_equity:,.2f}")
            logger.info("P&L:   UNKNOWN (no session start baseline)")
        else:
            pnl = end_equity - start_equity
            pnl_pct = (pnl / start_equity * 100)
            logger.info(f"End:   ${end_equity:,.2f}")
            logger.info(f"P&L:   ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
        logger.info(f"Log:   {log_file}")
        logger.info("=" * 60)
        # Reset the day baseline so the next session re-captures it.  The
        # persisted state file is date-stamped, so it is ignored on a new day
        # and re-used on a same-day restart (stable day P&L).
        self.start_equity = 0.0
        if close_broker:
            await self.broker.close()


# ── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    trader = LiveTrader()
    asyncio.run(trader.run())
