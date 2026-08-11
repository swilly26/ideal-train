"""Tests for the TURBO "VIOLENCE" high-volatility tier.

Covers:
1. Config — ``VIOLENCE_SYMBOLS`` list, ``ENABLE_VIOLENCE_TIER`` flag, and the
   effective symbol pool (base 4 + violence 7 = 9, falling back to the base 4
   when the flag is False).
2. Tier-aware risk routing — wider stops/TP (9%/13%) and smaller sizing (40%)
   for violence symbols; base symbols keep 6%/8% and 50%.
3. Trend extension — a violence-tier position that is profitable and still
   trending at the 30-min mark extends to 60 min; base tier always hard-exits
   at 30 min.
4. UVXY routing — mean-reversion signals use the short-lookback (10-bar)
   instance.
5. Short capability on the violence tier — momentum SELL shorts use violence
   sizing and the inverted (BUY-above-entry) protective stop.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

import turbo_trader
from src.execution.broker import OrderResult, OrderSide
from src.execution.position_manager import PositionManager
from src.strategies.base import Signal, SignalType


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _ohlcv(closes):
    idx = pd.date_range("2026-01-01 09:30", periods=len(closes), freq="1min")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.2 for c in closes],
            "low": [c - 0.2 for c in closes],
            "close": closes,
            "volume": [1000] * len(closes),
        },
        index=idx,
    )


def _rising_df(n=10, start=50.0, step=0.5):
    return _ohlcv([start + i * step for i in range(n)])


def _falling_df(n=10, start=50.0, step=-0.5):
    return _ohlcv([start + i * step for i in range(n)])


def _flat_df(n=10, price=50.0):
    return _ohlcv([price] * n)


class FakeBroker:
    """In-memory broker stand-in (same shape as the turbo short tests)."""

    def __init__(self, fill_result=None, status="accepted", positions=None):
        self.orders = []
        self.stop_requests = []  # (symbol, qty, stop_price, client_id, side)
        self._open_orders = []
        self.fill_result = fill_result
        self.status = status
        self.positions = list(positions or [])

    async def get_account(self):
        return {"equity": 200_000.0, "buying_power": 400_000.0,
                "cash": 100_000.0, "portfolio_value": 200_000.0}

    async def get_open_orders(self, symbol=None):
        return [o for o in self._open_orders if o.symbol == symbol]

    async def place_order(self, order):
        self.orders.append(order)
        return OrderResult(
            order_id="order-1",
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            filled_quantity=0.0,
            filled_avg_price=None,
            status=self.status,
            created_at=datetime.now(timezone.utc),
        )

    async def place_stop_order(self, symbol, qty, stop_price, client_id=None, side="SELL"):
        self.stop_requests.append((symbol, qty, stop_price, client_id, side))
        return MagicMock(status="new")

    async def wait_for_order_fill(self, order_id, timeout=8.0, poll_interval=0.25):
        return self.fill_result

    async def cancel_order(self, order_id):
        return True

    async def cancel_order_and_wait(self, order_id):
        self._open_orders = [o for o in self._open_orders if str(o.id) != str(order_id)]
        return True

    async def get_positions(self):
        return self.positions


def _make_trader(broker: FakeBroker):
    trader = object.__new__(turbo_trader.TurboTrader)
    trader.broker = broker
    trader.pm = PositionManager(turbo_trader.STRATEGY_CONFIG)
    trader._entry_times = {}
    trader._ls_symbols = set()
    trader._extended_holds = set()
    trader.levels_cache = {}
    trader.mean_reversion = MagicMock()
    trader.mean_reversion.generate_signals.return_value = []
    trader.uvxy_mean_reversion = MagicMock()
    trader.uvxy_mean_reversion.generate_signals.return_value = []
    return trader


class _Pos:
    """Minimal position stand-in for _trend_extension_qualifies."""

    def __init__(self, quantity, entry_price):
        self.quantity = quantity
        self.entry_price = entry_price


# ---------------------------------------------------------------------------
# 1. Config: symbol list, toggle, effective pool
# ---------------------------------------------------------------------------
class TestViolenceConfig:
    def test_violence_symbols_list(self):
        assert turbo_trader.VIOLENCE_SYMBOLS == [
            "TNA", "TZA", "LABU", "LABD", "UVXY", "NVDL", "TSLR",
        ]

    def test_enable_flag_default_true(self):
        assert turbo_trader.ENABLE_VIOLENCE_TIER is True

    def test_effective_pool_includes_base_plus_violence(self):
        pool = turbo_trader._effective_symbols()
        assert pool == turbo_trader.TURBO_SYMBOLS + turbo_trader.VIOLENCE_SYMBOLS
        # 4 base + 7 violence = 11 (the brief's "9" was an arithmetic slip)
        assert len(pool) == 11
        assert turbo_trader.SYMBOLS == pool

    def test_effective_pool_falls_back_when_disabled(self, monkeypatch):
        monkeypatch.setattr(turbo_trader, "ENABLE_VIOLENCE_TIER", False)
        assert turbo_trader._effective_symbols() == turbo_trader.TURBO_SYMBOLS
        assert len(turbo_trader._effective_symbols()) == 4

    def test_symbols_module_var_matches_effective_pool(self):
        assert turbo_trader.SYMBOLS == turbo_trader._effective_symbols()

    def test_base_turbo_symbols_unchanged(self):
        assert turbo_trader.TURBO_SYMBOLS == ["SOXL", "TQQQ", "FNGU", "SPXL"]
        # Existing turbo behavior is untouched: base risk params and sizing
        assert turbo_trader._risk_params_for("SOXL") == (0.06, 0.08)
        assert turbo_trader._position_size_pct_for("SOXL") == 0.50
        assert turbo_trader.MAX_POSITIONS == 2
        assert turbo_trader.MAX_HOLD_MINUTES == 30

    def test_violence_risk_bands(self):
        # Stop 8-10%, TP 12-15%, size 35-40% — per the owner's brief
        assert 0.08 <= turbo_trader.VIOLENCE_STOP_LOSS_PCT <= 0.10
        assert 0.12 <= turbo_trader.VIOLENCE_TAKE_PROFIT_PCT <= 0.15
        assert 0.35 <= turbo_trader.VIOLENCE_POSITION_SIZE_PCT <= 0.40


# ---------------------------------------------------------------------------
# 2. Tier-aware routing helpers
# ---------------------------------------------------------------------------
class TestTierRouting:
    @pytest.mark.parametrize("sym", ["TNA", "TZA", "LABU", "LABD", "UVXY", "NVDL", "TSLR"])
    def test_violence_symbols_recognized(self, sym):
        assert turbo_trader._is_violence_symbol(sym) is True
        assert turbo_trader._is_violence_symbol(sym.lower()) is True

    @pytest.mark.parametrize("sym", ["SOXL", "TQQQ", "FNGU", "SPXL"])
    def test_base_symbols_not_violence(self, sym):
        assert turbo_trader._is_violence_symbol(sym) is False

    def test_risk_params_tier_aware(self):
        assert turbo_trader._risk_params_for("TNA") == (0.09, 0.13)
        assert turbo_trader._risk_params_for("UVXY") == (0.09, 0.13)
        assert turbo_trader._risk_params_for("SOXL") == (
            turbo_trader.STRATEGY_CONFIG.stop_loss_pct,
            turbo_trader.STRATEGY_CONFIG.take_profit_pct,
        )

    def test_position_size_tier_aware(self):
        assert turbo_trader._position_size_pct_for("TNA") == 0.40
        assert turbo_trader._position_size_pct_for("SOXL") == 0.50


# ---------------------------------------------------------------------------
# 3. Trend extension qualification
# ---------------------------------------------------------------------------
class TestTrendExtension:
    def test_long_profitable_and_rising(self):
        pos = _Pos(quantity=100, entry_price=50.0)
        assert turbo_trader._trend_extension_qualifies(_rising_df(start=50.5), pos) is True

    def test_long_profitable_but_falling(self):
        pos = _Pos(quantity=100, entry_price=50.0)
        # last close (48.5) < first close (52.0) → not trending up
        assert turbo_trader._trend_extension_qualifies(_falling_df(start=52.0), pos) is False

    def test_long_losing(self):
        pos = _Pos(quantity=100, entry_price=60.0)
        assert turbo_trader._trend_extension_qualifies(_rising_df(start=50.0), pos) is False

    def test_short_profitable_and_falling(self):
        pos = _Pos(quantity=-100, entry_price=55.0)
        assert turbo_trader._trend_extension_qualifies(_falling_df(start=54.0), pos) is True

    def test_short_profitable_but_rising(self):
        pos = _Pos(quantity=-100, entry_price=55.0)
        assert turbo_trader._trend_extension_qualifies(_rising_df(start=54.0), pos) is False

    def test_short_losing(self):
        pos = _Pos(quantity=-100, entry_price=45.0)
        assert turbo_trader._trend_extension_qualifies(_falling_df(start=50.0), pos) is False

    def test_insufficient_bars(self):
        pos = _Pos(quantity=100, entry_price=50.0)
        assert turbo_trader._trend_extension_qualifies(_ohlcv([50.0]), pos) is False

    def test_flat_price_not_trending(self):
        pos = _Pos(quantity=100, entry_price=49.0)
        assert turbo_trader._trend_extension_qualifies(_flat_df(price=50.0), pos) is False


# ---------------------------------------------------------------------------
# 4. Tier-aware stop-loss / take-profit in _check_risk_stops
# ---------------------------------------------------------------------------
class TestRiskStopsTierAware:
    @pytest.mark.asyncio
    async def test_violence_long_stop_loss_wider(self):
        """TNA -10% → beyond the 9% violence stop → closed."""
        broker = FakeBroker()
        trader = _make_trader(broker)
        trader.pm.open_position("TNA", 100, 50.0)
        trader._entry_times["TNA"] = datetime.now(timezone.utc)

        class FakeProvider:
            async def fetch_bars(self, *a, **k):
                return type("MDF", (), {"df": _ohlcv([45.0, 45.0])})()  # -10%

        trader.provider = FakeProvider()
        await trader._check_risk_stops()
        assert len(broker.orders) == 1
        assert broker.orders[0].side == OrderSide.SELL
        assert trader.pm.get_positions().get("TNA") is None

    @pytest.mark.asyncio
    async def test_violence_long_within_band_does_not_exit(self):
        """TNA -7% → inside the 9% stop and below 13% TP → hold."""
        broker = FakeBroker()
        trader = _make_trader(broker)
        trader.pm.open_position("TNA", 100, 50.0)
        trader._entry_times["TNA"] = datetime.now(timezone.utc)

        class FakeProvider:
            async def fetch_bars(self, *a, **k):
                return type("MDF", (), {"df": _ohlcv([46.5, 46.5])})()  # -7%

        trader.provider = FakeProvider()
        await trader._check_risk_stops()
        assert broker.orders == []
        assert trader.pm.has_position("TNA")

    @pytest.mark.asyncio
    async def test_base_symbol_keeps_base_stop(self):
        """SOXL -7% → beyond the 6% base stop → closed (base params intact)."""
        broker = FakeBroker()
        trader = _make_trader(broker)
        trader.pm.open_position("SOXL", 100, 50.0)
        trader._entry_times["SOXL"] = datetime.now(timezone.utc)

        class FakeProvider:
            async def fetch_bars(self, *a, **k):
                return type("MDF", (), {"df": _ohlcv([46.5, 46.5])})()  # -7%

        trader.provider = FakeProvider()
        await trader._check_risk_stops()
        assert len(broker.orders) == 1
        assert trader.pm.get_positions().get("SOXL") is None

    @pytest.mark.asyncio
    async def test_violence_long_take_profit_wider(self):
        """TNA +14% → beyond the 13% violence TP → closed."""
        broker = FakeBroker()
        trader = _make_trader(broker)
        trader.pm.open_position("TNA", 100, 50.0)
        trader._entry_times["TNA"] = datetime.now(timezone.utc)

        class FakeProvider:
            async def fetch_bars(self, *a, **k):
                return type("MDF", (), {"df": _ohlcv([57.0, 57.0])})()  # +14%

        trader.provider = FakeProvider()
        await trader._check_risk_stops()
        assert len(broker.orders) == 1
        assert broker.orders[0].side == OrderSide.SELL


# ---------------------------------------------------------------------------
# 5. Time-based exit + trend extension
# ---------------------------------------------------------------------------
class TestTrendExtensionHold:
    @pytest.mark.asyncio
    async def test_trending_violence_position_extends_to_60min(self):
        """TNA profitable & rising at 31 min → no exit, extension marked."""
        broker = FakeBroker()
        trader = _make_trader(broker)
        trader.pm.open_position("TNA", 100, 50.0)
        trader._entry_times["TNA"] = datetime.now(timezone.utc) - timedelta(minutes=31)

        class FakeProvider:
            async def fetch_bars(self, *a, **k):
                return type("MDF", (), {"df": _rising_df(start=50.5)})()  # +6%, rising

        trader.provider = FakeProvider()
        await trader._check_risk_stops()
        assert broker.orders == []  # not time-exited
        assert "TNA" in trader._extended_holds

    @pytest.mark.asyncio
    async def test_extended_position_exits_at_60min(self):
        broker = FakeBroker()
        trader = _make_trader(broker)
        trader.pm.open_position("TNA", 100, 50.0)
        trader._extended_holds.add("TNA")
        trader._entry_times["TNA"] = datetime.now(timezone.utc) - timedelta(minutes=61)

        class FakeProvider:
            async def fetch_bars(self, *a, **k):
                return type("MDF", (), {"df": _rising_df(start=50.5)})()

        trader.provider = FakeProvider()
        await trader._check_risk_stops()
        assert len(broker.orders) == 1
        assert "TNA" not in trader._extended_holds

    @pytest.mark.asyncio
    async def test_violence_position_not_trending_still_exits_at_30min(self):
        """TNA profitable but flattening at 31 min → hard 30-min exit."""
        broker = FakeBroker()
        trader = _make_trader(broker)
        trader.pm.open_position("TNA", 100, 50.0)
        trader._entry_times["TNA"] = datetime.now(timezone.utc) - timedelta(minutes=31)

        class FakeProvider:
            async def fetch_bars(self, *a, **k):
                return type("MDF", (), {"df": _flat_df(price=51.0)})()  # +2% but flat

        trader.provider = FakeProvider()
        await trader._check_risk_stops()
        assert len(broker.orders) == 1
        assert "TNA" not in trader._extended_holds

    @pytest.mark.asyncio
    async def test_base_symbol_never_extended(self):
        """SOXL profitable & rising at 31 min → base tier still hard-exits."""
        broker = FakeBroker()
        trader = _make_trader(broker)
        trader.pm.open_position("SOXL", 100, 50.0)
        trader._entry_times["SOXL"] = datetime.now(timezone.utc) - timedelta(minutes=31)

        class FakeProvider:
            async def fetch_bars(self, *a, **k):
                return type("MDF", (), {"df": _rising_df(start=50.5)})()  # +6%, rising

        trader.provider = FakeProvider()
        await trader._check_risk_stops()
        assert len(broker.orders) == 1
        assert "SOXL" not in trader._extended_holds


# ---------------------------------------------------------------------------
# 6. Violence-tier entry sizing + protective stop placement
# ---------------------------------------------------------------------------
class TestViolenceEntrySizing:
    @pytest.mark.asyncio
    async def test_handle_buy_uses_40pct_sizing_and_9pct_stop(self):
        broker = FakeBroker()
        trader = _make_trader(broker)
        await trader._handle_buy("TNA", 20.0, 1.0, strategy="mean_reversion")
        # 200_000 equity × 0.40 / $20 = 4,000 shares
        assert broker.orders[0].quantity == 4000.0
        assert broker.orders[0].side == OrderSide.BUY
        # Protective stop at 9% below entry
        assert broker.stop_requests[0][0] == "TNA"
        assert broker.stop_requests[0][1] == 4000
        assert broker.stop_requests[0][2] == round(20.0 * (1 - 0.09), 2)
        assert broker.stop_requests[0][4] == "SELL"

    @pytest.mark.asyncio
    async def test_handle_short_sell_uses_40pct_sizing_and_buy_stop_above(self):
        """Short capability applies to the violence tier: SELL entry + BUY
        stop ABOVE entry at the 9% violence distance."""
        broker = FakeBroker()
        trader = _make_trader(broker)
        await trader._handle_short_sell("TNA", 20.0, 0.9, strategy="momentum")
        assert broker.orders[0].side == OrderSide.SELL
        assert broker.orders[0].quantity == 4000.0
        assert trader.pm.get_positions().get("TNA").quantity == -4000.0
        assert broker.stop_requests[0][2] == round(20.0 * (1 + 0.09), 2)
        assert broker.stop_requests[0][4] == "BUY"

    @pytest.mark.asyncio
    async def test_base_buy_keeps_50pct_sizing(self):
        broker = FakeBroker()
        trader = _make_trader(broker)
        await trader._handle_buy("SOXL", 20.0, 1.0, strategy="mean_reversion")
        # 200_000 × 0.50 / $20 = 5,000 shares — unchanged from before the tier
        assert broker.orders[0].quantity == 5000.0
        assert broker.stop_requests[0][2] == round(20.0 * (1 - 0.06), 2)


# ---------------------------------------------------------------------------
# 7. UVXY mean-reversion routing (10-bar lookback)
# ---------------------------------------------------------------------------
class TestUvxyRouting:
    @pytest.mark.asyncio
    async def test_tick_routes_uvxy_to_short_lookback_mr(self, monkeypatch):
        """Only the UVXY short-lookback MR instance produces the entry; the
        base MR instance must NOT fire for UVXY."""
        broker = FakeBroker()
        trader = _make_trader(broker)
        bought = []

        async def fake_handle_buy(symbol, price, confidence, strategy=""):
            bought.append(symbol)

        async def fake_check_risk_stops():
            pass

        trader._handle_buy = fake_handle_buy
        trader._check_risk_stops = fake_check_risk_stops
        # Base MR silent for every symbol, including UVXY
        trader.mean_reversion.generate_signals.return_value = []
        trader.liquidity_sweep = MagicMock()
        trader.liquidity_sweep.generate_signals.return_value = []
        monkeypatch.setattr(turbo_trader, "_generate_momentum_signals",
                            lambda *a, **k: [])

        class FakeProvider:
            async def fetch_bars(self, *a, **k):
                return type("MDF", (), {"df": _ohlcv(list(range(30, 70)))})()  # uptrend

        trader.provider = FakeProvider()
        # UVXY-only BUY from the short-lookback instance
        trader.uvxy_mean_reversion.generate_signals.return_value = [
            Signal(symbol="UVXY", timestamp=pd.Timestamp("2026-01-01 10:00"),
                   signal_type=SignalType.BUY, confidence=0.95,
                   metadata={"z_score": -2.5, "lookback": 10}),
        ]
        await trader._tick(1)
        assert bought == ["UVXY"]  # base MR never fired; UVXY bought first
        assert broker.orders == []

    def test_uvxy_mr_config_uses_10_bar_lookback(self):
        assert turbo_trader.UVXY_MR_CONFIG.extra["lookback"] == 10
        assert turbo_trader.STRATEGY_CONFIG.extra["lookback"] == 20
