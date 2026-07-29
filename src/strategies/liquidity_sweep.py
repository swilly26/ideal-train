"""Liquidity Sweep trading strategy.

Captures reversals at key market structure levels by detecting when price
breaches a prior high/low and then reverses back through it — a "liquidity
sweep" — confirmed by a Fair Value Gap (FVG) in the reversal direction.

Levels
------
- PDH / PDL : Prior day high and low (yesterday's full session).
- PMH / PML : Pre-market high and low (4:00 AM – 9:30 AM ET).

Signal flow
-----------
1. Sweep detection — price breaches a key level and reverses through it
   within ``SWEEP_MAX_CANDLES`` candles.
2. FVG confirmation — a Fair Value Gap forms in the reversal direction.
3. Entry at the FVG mid-point, stop at the swept level, target at 2:1 R:R.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.strategies.base import Signal, SignalType

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

SWEEP_MAX_CANDLES = 3          # max candles for a valid reversal after breach
FVG_MIN_SIZE_PCT = 0.0005      # minimum FVG size as % of price (0.05%)
CONFIDENCE_THRESHOLD = 0.5     # minimum confidence to trade


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class FVG:
    """A Fair Value Gap between two non-overlapping candles."""
    top: float       # upper boundary of the gap
    bottom: float    # lower boundary of the gap
    mid: float       # midpoint = ideal entry
    size_pct: float  # gap size as % of price


# ── Strategy ─────────────────────────────────────────────────────────

class LiquiditySweepStrategy:
    """Detect liquidity sweeps at key market structure levels.

    Parameters
    ----------
    symbols : list[str]
        Ticker symbols to trade.
    config : dict | None
        Optional overrides for SWEEP_MAX_CANDLES, FVG_MIN_SIZE_PCT, etc.
    """

    def __init__(self, symbols: list[str], config: dict | None = None):
        self.symbols = [s.upper() for s in symbols]
        self.config = config or {}
        self.levels: dict[str, dict[str, float | None]] = {}

        # Allow per-instance overrides of the module-level constants
        self.sweep_max_candles: int = int(
            self.config.get("SWEEP_MAX_CANDLES", SWEEP_MAX_CANDLES)
        )
        self.fvg_min_size_pct: float = float(
            self.config.get("FVG_MIN_SIZE_PCT", FVG_MIN_SIZE_PCT)
        )
        self.confidence_threshold: float = float(
            self.config.get("CONFIDENCE_THRESHOLD", CONFIDENCE_THRESHOLD)
        )

    # ── Level calculation ────────────────────────────────────────────

    async def calculate_levels(self, provider) -> None:
        """Fetch PDH/PDL and PMH/PML for every symbol.

        Called once per session after market open.  Stores results in
        ``self.levels``.  Symbols that fail data fetch are skipped with
        a warning rather than crashing.
        """
        today = datetime.now(timezone.utc).date()

        for symbol in self.symbols:
            try:
                levels: dict[str, float | None] = {
                    "pdh": None, "pdl": None,
                    "pmh": None, "pml": None,
                }

                # ── PDH / PDL: yesterday's daily bar ─────────────────
                yesterday = today - timedelta(days=1)
                y_start = datetime(yesterday.year, yesterday.month, yesterday.day,
                                   tzinfo=timezone.utc)
                y_end = y_start + timedelta(days=1) - timedelta(seconds=1)

                try:
                    mdf = await provider.fetch_bars(
                        symbol, start=y_start, end=y_end, timeframe="1day"
                    )
                    day_df = mdf.df
                    if not day_df.empty:
                        levels["pdh"] = float(day_df["high"].iloc[-1])
                        levels["pdl"] = float(day_df["low"].iloc[-1])
                except Exception:
                    logger.warning(
                        "LiquiditySweep %s: could not fetch daily bar for PDH/PDL, "
                        "skipping day levels", symbol
                    )

                # ── PMH / PML: pre-market 4:00–9:30 AM ET ────────────
                # 4:00 AM ET = 08:00 UTC, 9:30 AM ET = 13:30 UTC
                pm_start = datetime(today.year, today.month, today.day,
                                    8, 0, 0, tzinfo=timezone.utc)
                pm_end = datetime(today.year, today.month, today.day,
                                  13, 30, 0, tzinfo=timezone.utc)

                try:
                    mdf = await provider.fetch_bars(
                        symbol, start=pm_start, end=pm_end, timeframe="1min"
                    )
                    pm_df = mdf.df
                    if not pm_df.empty:
                        levels["pmh"] = float(pm_df["high"].max())
                        levels["pml"] = float(pm_df["low"].min())
                except Exception:
                    logger.warning(
                        "LiquiditySweep %s: could not fetch pre-market bars for "
                        "PMH/PML, skipping pre-market levels", symbol
                    )

                # Only store if we got at least one level
                if any(v is not None for v in levels.values()):
                    self.levels[symbol] = levels
                    logger.info(
                        "LiquiditySweep %s levels: PDH=%s PDL=%s PMH=%s PML=%s",
                        symbol,
                        f"{levels['pdh']:.2f}" if levels["pdh"] else "N/A",
                        f"{levels['pdl']:.2f}" if levels["pdl"] else "N/A",
                        f"{levels['pmh']:.2f}" if levels["pmh"] else "N/A",
                        f"{levels['pml']:.2f}" if levels["pml"] else "N/A",
                    )
                else:
                    logger.warning(
                        "LiquiditySweep %s: no levels available, skipping symbol", symbol
                    )

            except Exception as exc:
                logger.warning(
                    "LiquiditySweep %s: level calculation failed — %s", symbol, exc
                )

        if not self.levels:
            logger.warning("LiquiditySweep: no levels calculated for any symbol")

    # ── Sweep detection ──────────────────────────────────────────────

    def detect_sweep(
        self, df_1min: pd.DataFrame, levels: dict[str, float | None]
    ) -> str | None:
        """Check last *sweep_max_candles* candles for level breaches + reversals.

        Parameters
        ----------
        df_1min : pd.DataFrame
            Recent 1-min OHLCV bars (must have at least 2 rows).
        levels : dict
            Symbol levels dict with pdh/pdl/pmh/pml keys.

        Returns
        -------
        str | None
            ``"bullish"``, ``"bearish"``, or ``None`` if no sweep detected.
        """
        if len(df_1min) < 2:
            return None

        close = df_1min["close"]
        high = df_1min["high"]
        low = df_1min["low"]

        # Build list of (level_name, level_value) for levels that exist.
        support_levels: list[tuple[str, float]] = []
        resistance_levels: list[tuple[str, float]] = []

        for key, name in [("pdl", "PDL"), ("pml", "PML")]:
            val = levels.get(key)
            if val is not None:
                support_levels.append((name, val))

        for key, name in [("pdh", "PDH"), ("pmh", "PMH")]:
            val = levels.get(key)
            if val is not None:
                resistance_levels.append((name, val))

        n = min(self.sweep_max_candles + 1, len(df_1min))
        recent = df_1min.iloc[-n:]  # last n candles (chronological)

        # ── Bullish sweep: low breaches support → close recovers above ─
        for name, level in support_levels:
            for i in range(len(recent) - 1):
                # Candle i drops below the level
                if recent["low"].iloc[i] < level:
                    # Look for a subsequent candle that closes back above
                    for j in range(i + 1, min(i + 1 + self.sweep_max_candles, len(recent))):
                        if recent["close"].iloc[j] > level:
                            logger.debug(
                                "Bullish sweep detected at %s=$%.2f "
                                "(breach candle %d, recovery candle %d)",
                                name, level, i, j,
                            )
                            return "bullish"
        # ── Bearish sweep: high breaches resistance → close drops below ─
        for name, level in resistance_levels:
            for i in range(len(recent) - 1):
                if recent["high"].iloc[i] > level:
                    for j in range(i + 1, min(i + 1 + self.sweep_max_candles, len(recent))):
                        if recent["close"].iloc[j] < level:
                            logger.debug(
                                "Bearish sweep detected at %s=$%.2f "
                                "(breach candle %d, recovery candle %d)",
                                name, level, i, j,
                            )
                            return "bearish"

        return None

    # ── FVG detection ────────────────────────────────────────────────

    def detect_fvg(
        self, df_1min: pd.DataFrame, direction: str
    ) -> FVG | None:
        """Check the last 3 candles for a Fair Value Gap in *direction*.

        - Bullish FVG: ``candle[0].high < candle[2].low`` — a gap up.
          Gap is between ``candle[0].high`` and ``candle[2].low``.
        - Bearish FVG: ``candle[0].low > candle[2].high`` — a gap down.
          Gap is between ``candle[2].high`` and ``candle[0].low``.

        Candle indexing: candle[0] = most recent, candle[2] = 3rd most recent.

        Parameters
        ----------
        df_1min : pd.DataFrame
            Recent 1-min OHLCV bars (needs at least 3 rows).
        direction : str
            ``"bullish"`` or ``"bearish"``.

        Returns
        -------
        FVG | None
            The detected FVG with top/bottom/mid, or None.
        """
        if len(df_1min) < 3:
            return None

        # Recent 3 candles: most recent = iloc[-1]
        c0 = df_1min.iloc[-1]
        c1 = df_1min.iloc[-2]
        c2 = df_1min.iloc[-3]

        price = float(c0["close"])
        if price <= 0:
            return None

        if direction == "bullish":
            # FVG: candle[0].high < candle[2].low → unfilled gap up
            if float(c0["high"]) < float(c2["low"]):
                gap_top = float(c2["low"])
                gap_bottom = float(c0["high"])
                gap_size = gap_top - gap_bottom
                size_pct = gap_size / price
                if size_pct >= self.fvg_min_size_pct:
                    mid = (gap_top + gap_bottom) / 2.0
                    return FVG(top=gap_top, bottom=gap_bottom, mid=mid, size_pct=size_pct)

        elif direction == "bearish":
            # FVG: candle[0].low > candle[2].high → unfilled gap down
            if float(c0["low"]) > float(c2["high"]):
                gap_top = float(c0["low"])
                gap_bottom = float(c2["high"])
                gap_size = gap_top - gap_bottom
                size_pct = gap_size / price
                if size_pct >= self.fvg_min_size_pct:
                    mid = (gap_top + gap_bottom) / 2.0
                    return FVG(top=gap_top, bottom=gap_bottom, mid=mid, size_pct=size_pct)

        return None

    # ── Confidence scoring ───────────────────────────────────────────

    def _score_confidence(
        self,
        direction: str,
        fvg: FVG,
        sweep_immediacy: bool,
        swept_level_name: str,
    ) -> float:
        """Compute a confidence score for the sweep + FVG combination.

        Parameters
        ----------
        direction : str
            ``"bullish"`` or ``"bearish"``.
        fvg : FVG
            The detected Fair Value Gap.
        sweep_immediacy : bool
            True if the reversal happened on the very next candle after the breach.
        swept_level_name : str
            Name of the breached level (PDH, PDL, PMH, PML).

        Returns
        -------
        float
            Confidence between 0.0 and 1.0.
        """
        base = 0.3
        confidence = base

        # Sweep was immediate (next-candle reversal)
        if sweep_immediacy:
            confidence += 0.3

        # FVG is large (>2x minimum size)
        if fvg.size_pct > (self.fvg_min_size_pct * 2):
            confidence += 0.2

        # Level was a prior-day level (PDH/PDL), not just pre-market
        if swept_level_name in ("PDH", "PDL"):
            confidence += 0.2

        return min(1.0, round(confidence, 4))

    # ── Signal generation ────────────────────────────────────────────

    def generate_signals(
        self, data: pd.DataFrame, levels: dict[str, float | None]
    ) -> list[Signal]:
        """Produce trading signals from 1-min OHLCV data and pre-computed levels.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV bars indexed by timestamp (1-min timeframe).
        levels : dict
            Symbol levels dict with pdh/pdl/pmh/pml keys.

        Returns
        -------
        list[Signal]
            Zero or one signal per call (liquidity sweeps are sparse).
        """
        if not levels:
            return []

        # At minimum we need enough bars for sweep detection + FVG (at least 5)
        if len(data) < max(self.sweep_max_candles + 1, 3) + 2:
            return []

        # 1. Detect sweep
        direction = self.detect_sweep(data, levels)
        if direction is None:
            return []

        # 2. Confirm with FVG
        fvg = self.detect_fvg(data, direction)
        if fvg is None:
            return []

        # 3. Determine which level was swept (for stop placement & confidence)
        swept_level_name, swept_level = self._find_swept_level(data, levels, direction)

        # 4. Build the signal
        current_price = float(data["close"].iloc[-1])
        timestamp = data.index[-1]

        # Re-check sweep immediacy for confidence scoring
        sweep_immediacy = self._check_sweep_immediacy(data, levels, direction)

        confidence = self._score_confidence(direction, fvg, sweep_immediacy, swept_level_name)

        if confidence < self.confidence_threshold:
            return []

        # Stop-loss at the swept level (natural invalidation point)
        stop_loss = swept_level if swept_level else current_price
        stop_distance = abs(current_price - stop_loss)

        # Take-profit at 2x stop distance (1:2 risk-reward)
        if direction == "bullish":
            take_profit = current_price + 2.0 * stop_distance
            signal_type = SignalType.BUY
        else:
            take_profit = current_price - 2.0 * stop_distance
            signal_type = SignalType.SELL

        # Determine the symbol from levels metadata (the levels dict may carry
        # symbol info, or we rely on the caller to know the symbol context).
        symbol = str(levels.get("_symbol", "SYM"))

        return [Signal(
            symbol=symbol,
            timestamp=timestamp,
            signal_type=signal_type,
            confidence=confidence,
            metadata={
                "strategy": "liquidity_sweep",
                "direction": direction,
                "swept_level": swept_level_name,
                "swept_price": round(swept_level, 4) if swept_level else None,
                "fvg_top": round(fvg.top, 4),
                "fvg_bottom": round(fvg.bottom, 4),
                "fvg_mid": round(fvg.mid, 4),
                "fvg_size_pct": round(fvg.size_pct, 6),
                "stop_loss": round(stop_loss, 4),
                "take_profit": round(take_profit, 4),
                "sweep_immediacy": sweep_immediacy,
            },
        )]

    # ── Internal helpers ─────────────────────────────────────────────

    def _find_swept_level(
        self,
        df_1min: pd.DataFrame,
        levels: dict[str, float | None],
        direction: str,
    ) -> tuple[str, float | None]:
        """Find which specific level was breached in the sweep.

        Returns (level_name, level_price) for the first breached level found.
        """
        if direction == "bullish":
            candidates = [("PDL", levels.get("pdl")), ("PML", levels.get("pml"))]
        else:
            candidates = [("PDH", levels.get("pdh")), ("PMH", levels.get("pmh"))]

        recent = df_1min.iloc[-self.sweep_max_candles - 1:]

        for name, price in candidates:
            if price is None:
                continue
            for i in range(len(recent) - 1):
                if direction == "bullish" and recent["low"].iloc[i] < price:
                    return (name, price)
                if direction == "bearish" and recent["high"].iloc[i] > price:
                    return (name, price)

        return ("UNKNOWN", None)

    def _check_sweep_immediacy(
        self,
        df_1min: pd.DataFrame,
        levels: dict[str, float | None],
        direction: str,
    ) -> bool:
        """Check if the sweep reversed on the very next candle (immediate)."""
        if len(df_1min) < 3:
            return False

        n = min(self.sweep_max_candles + 1, len(df_1min))
        recent = df_1min.iloc[-n:]

        if direction == "bullish":
            candidates = [levels.get("pdl"), levels.get("pml")]
            for i in range(len(recent) - 1):
                lvl = None
                for c in candidates:
                    if c is not None and recent["low"].iloc[i] < c:
                        lvl = c
                        break
                if lvl is not None:
                    # Check if the very next candle closed above
                    if i + 1 < len(recent) and recent["close"].iloc[i + 1] > lvl:
                        return True
        else:
            candidates = [levels.get("pdh"), levels.get("pmh")]
            for i in range(len(recent) - 1):
                lvl = None
                for c in candidates:
                    if c is not None and recent["high"].iloc[i] > c:
                        lvl = c
                        break
                if lvl is not None:
                    if i + 1 < len(recent) and recent["close"].iloc[i + 1] < lvl:
                        return True

        return False
