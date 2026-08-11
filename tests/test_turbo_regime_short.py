"""Tests for Recommendation #1: trend-gated directional trading with
short-side capability in the turbo trader.

Covers:
1. Regime gate — mean-reversion LONGs are skipped in a confirmed downtrend
   (price below 10-bar MA AND RSI(14) < 40), and allowed otherwise.
2. Momentum SELL confidence — rescaled so realistic 3x ETF moves (1-3% from
   the MA) actually cross the 0.4 activation threshold; BUY confidence is
   untouched.
3. Short entry/exit — momentum SELL opens a SHORT (SELL order, negative-qty
   position, BUY stop ABOVE entry), closing buys to cover with correct P&L,
   and the EOD liquidation flattens shorts too.
"""
from datetime import datetime, timezone
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


def _downtrend_df(n=40, start=100.0, step=-0.5):
    """Monotonic decline → price below MA10 and RSI ≈ 0 at the last bar."""
    return _ohlcv([start + i * step for i in range(n)])


def _uptrend_df(n=40, start=90.0, step=0.3):
    """Monotonic rise → price above MA10 and RSI ≈ 100 at the last bar."""
    return _ohlcv([start + i * step for i in range(n)])


class FakeBroker:
    """In-memory broker stand-in for short entry/exit paths."""

    def __init__(self, fill_result=None, status="accepted", positions=None):
        self.orders = []          # Order objects submitted via place_order
        self.stop_requests = []   # (symbol, qty, stop_price, client_id, side)
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


def _filled_order_mock(qty="100", price="32.50"):
    o = MagicMock()
    o.status = "filled"
    o.qty = qty
    o.filled_avg_price = price
    return o


def _make_trader(broker: FakeBroker):
    trader = object.__new__(turbo_trader.TurboTrader)
    trader.broker = broker
    trader.pm = PositionManager(turbo_trader.STRATEGY_CONFIG)
    trader._entry_times = {}
    trader._ls_symbols = set()
    trader._extended_holds = set()
    trader.levels_cache = {}
    # Mean-reversion instances: base + UVXY short-lookback (violence tier).
    # MagicMock keeps _tick's UVXY routing from raising on symbols it visits.
    trader.mean_reversion = MagicMock()
    trader.mean_reversion.generate_signals.return_value = []
    trader.uvxy_mean_reversion = MagicMock()
    trader.uvxy_mean_reversion.generate_signals.return_value = []
    return trader


# ---------------------------------------------------------------------------
# 1. Regime gate
# ---------------------------------------------------------------------------
class TestRegimeGate:
    def test_blocks_long_in_downtrend(self):
        data = _downtrend_df(step=-0.5)  # price 80.5 < MA10 (~83), RSI ~0
        allow, reason = turbo_trader._regime_gate_allows_long(data)
        assert allow is False
        assert "downtrend" in reason

    def test_allows_long_in_uptrend(self):
        data = _uptrend_df()  # price above MA10, RSI high
        allow, reason = turbo_trader._regime_gate_allows_long(data)
        assert allow is True

    def test_allows_when_rsi_healthy_despite_price_below_ma(self):
        # Price dips below the MA for one bar but RSI stays strong → allowed.
        closes = [100.0] * 30 + [98.0] * 10   # recent chop below, RSI ~40-50
        data = _ohlcv(closes)
        allow, _ = turbo_trader._regime_gate_allows_long(data)
        assert allow is True

    @pytest.mark.asyncio
    async def test_tick_skips_mean_reversion_long_when_gate_blocks(self, monkeypatch):
        """End-to-end: a mean-reversion BUY signal in a downtrend must NOT
        reach _handle_buy, and must be logged as a regime-gate skip."""
        broker = FakeBroker()
        trader = _make_trader(broker)
        bought = []

        async def fake_handle_buy(symbol, price, confidence, strategy=""):
            bought.append((symbol, price, confidence, strategy))

        async def fake_check_risk_stops():
            pass

        trader._handle_buy = fake_handle_buy
        trader._check_risk_stops = fake_check_risk_stops
        trader.mean_reversion = MagicMock()
        trader.mean_reversion.generate_signals.return_value = [
            Signal(symbol="SOXL", timestamp=pd.Timestamp("2026-01-01 10:00"),
                   signal_type=SignalType.BUY, confidence=1.0,
                   metadata={"z_score": -3.0}),
        ]
        trader.liquidity_sweep = MagicMock()
        trader.liquidity_sweep.generate_signals.return_value = []
        monkeypatch.setattr(turbo_trader, "_generate_momentum_signals",
                            lambda *a, **k: [])

        class FakeProvider:
            async def fetch_bars(self, *a, **k):
                return type("MDF", (), {"df": _downtrend_df()})()

        trader.provider = FakeProvider()
        assert turbo_trader.ENABLE_REGIME_GATE is True
        await trader._tick(1)
        assert bought == []  # gate blocked the long

    @pytest.mark.asyncio
    async def test_tick_enters_long_when_gate_disabled_or_uptrend(self, monkeypatch):
        broker = FakeBroker()
        trader = _make_trader(broker)
        bought = []

        async def fake_handle_buy(symbol, price, confidence, strategy=""):
            bought.append((symbol, price, confidence, strategy))

        async def fake_check_risk_stops():
            pass

        trader._handle_buy = fake_handle_buy
        trader._check_risk_stops = fake_check_risk_stops
        trader.mean_reversion = MagicMock()
        trader.mean_reversion.generate_signals.return_value = [
            Signal(symbol="SOXL", timestamp=pd.Timestamp("2026-01-01 10:00"),
                   signal_type=SignalType.BUY, confidence=1.0,
                   metadata={"z_score": -3.0}),
        ]
        trader.liquidity_sweep = MagicMock()
        trader.liquidity_sweep.generate_signals.return_value = []
        monkeypatch.setattr(turbo_trader, "_generate_momentum_signals",
                            lambda *a, **k: [])

        class FakeProvider:
            async def fetch_bars(self, *a, **k):
                return type("MDF", (), {"df": _uptrend_df()})()

        trader.provider = FakeProvider()
        await trader._tick(1)
        assert len(bought) == 1
        assert bought[0][0] == "SOXL"


# ---------------------------------------------------------------------------
# 2. Momentum confidence fix
# ---------------------------------------------------------------------------
class TestMomentumConfidence:
    def test_sell_confidence_reaches_threshold_on_realistic_move(self):
        """1-3% below the MA must produce SELL confidence ≥ 0.4 (the old
        formula capped around 0.1-0.3 and could never fire)."""
        data = _downtrend_df(step=-0.5)  # last bar ~3% below MA10, RSI ~0
        signals = turbo_trader._generate_momentum_signals(data, "SOXL")
        sells = [s for s in signals if s.signal_type == SignalType.SELL]
        assert sells, "expected at least one SELL signal"
        assert max(s.confidence for s in sells) >= 0.4
        # A 3%+ move should reach the 0.5-1.0 range per the task spec
        assert max(s.confidence for s in sells) >= 0.5

    def test_sell_confidence_high_for_strong_breakdown(self):
        data = _downtrend_df(n=60, start=120.0, step=-1.0)  # ~8% below MA10
        signals = turbo_trader._generate_momentum_signals(data, "TQQQ")
        sells = [s for s in signals if s.signal_type == SignalType.SELL]
        assert max(s.confidence for s in sells) > 0.7

    def test_buy_confidence_unchanged(self):
        """The BUY branch must keep producing its original 0.3-1.0 range."""
        data = _uptrend_df()
        signals = turbo_trader._generate_momentum_signals(data, "SOXL")
        buys = [s for s in signals if s.signal_type == SignalType.BUY]
        assert buys, "expected at least one BUY signal"
        for s in buys:
            assert 0.3 <= s.confidence <= 1.0


# ---------------------------------------------------------------------------
# 3. Short entry / exit / EOD flow
# ---------------------------------------------------------------------------
class TestShortFlow:
    @pytest.mark.asyncio
    async def test_handle_short_sell_opens_short_with_buy_stop_above_entry(self):
        broker = FakeBroker(fill_result=_filled_order_mock(qty="100", price="32.50"))
        trader = _make_trader(broker)
        await trader._handle_short_sell("FNGU", 32.0, 0.8, strategy="momentum")

        # Entry: a SELL market order (Alpaca opens a short when flat)
        assert len(broker.orders) == 1
        assert broker.orders[0].side == OrderSide.SELL

        # Position recorded with NEGATIVE quantity (short)
        pos = trader.pm.get_positions().get("FNGU")
        assert pos is not None
        assert pos.quantity == -100.0
        assert pos.entry_price == 32.50

        # Protective stop: BUY side, ABOVE entry (6% up)
        assert len(broker.stop_requests) == 1
        symbol, qty, stop_price, _, stop_side = broker.stop_requests[0]
        assert symbol == "FNGU"
        assert qty == 100
        assert stop_side == "BUY"
        assert stop_price == round(32.50 * (1 + turbo_trader.STRATEGY_CONFIG.stop_loss_pct), 2)

    @pytest.mark.asyncio
    async def test_handle_short_sell_respects_max_positions(self):
        broker = FakeBroker(fill_result=_filled_order_mock())
        trader = _make_trader(broker)
        trader.pm.open_position("SOXL", 100.0, 30.0)
        trader.pm.open_position("TQQQ", 100.0, 40.0)  # MAX_POSITIONS = 2
        await trader._handle_short_sell("FNGU", 32.0, 0.8)
        assert broker.orders == []  # never submitted

    @pytest.mark.asyncio
    async def test_handle_sell_covers_short_with_buy_and_profit(self):
        broker = FakeBroker()
        trader = _make_trader(broker)
        trader.pm.open_position("FNGU", -100.0, 32.50)
        trader._entry_times["FNGU"] = datetime.now(timezone.utc)

        await trader._handle_sell("FNGU", 30.0, 1.0, strategy="risk_stop")

        # Closing a short = BUY-to-cover order for the abs quantity
        assert len(broker.orders) == 1
        assert broker.orders[0].side == OrderSide.BUY
        assert broker.orders[0].quantity == 100.0
        # Position closed
        assert trader.pm.get_positions().get("FNGU") is None
        # P&L on a short that fell 32.50 → 30.00 = +$250
        assert trader.pm.get_realized_pnl() == pytest.approx(250.0)

    @pytest.mark.asyncio
    async def test_tick_opens_short_on_momentum_sell(self, monkeypatch):
        """A momentum SELL with RSI < 40 and no position routes to
        _handle_short_sell instead of doing nothing."""
        broker = FakeBroker()
        trader = _make_trader(broker)
        shorts = []

        async def fake_short_sell(symbol, price, confidence, strategy=""):
            shorts.append((symbol, price, confidence, strategy))

        async def fake_check_risk_stops():
            pass

        trader._handle_short_sell = fake_short_sell
        trader._check_risk_stops = fake_check_risk_stops
        trader.mean_reversion = MagicMock()
        trader.mean_reversion.generate_signals.return_value = []
        trader.liquidity_sweep = MagicMock()
        trader.liquidity_sweep.generate_signals.return_value = []
        monkeypatch.setattr(
            turbo_trader, "_generate_momentum_signals",
            lambda data, symbol, **k: [
                Signal(symbol=symbol, timestamp=pd.Timestamp("2026-01-01 10:00"),
                       signal_type=SignalType.SELL, confidence=0.9,
                       metadata={"strategy": "momentum", "rsi": 30.0}),
            ],
        )

        class FakeProvider:
            async def fetch_bars(self, *a, **k):
                return type("MDF", (), {"df": _downtrend_df()})()

        trader.provider = FakeProvider()
        assert turbo_trader.ENABLE_SHORT_SELLING is True
        await trader._tick(1)
        assert len(shorts) == 1
        assert shorts[0][0] == "SOXL"

    @pytest.mark.asyncio
    async def test_tick_does_not_short_when_flag_disabled(self, monkeypatch):
        broker = FakeBroker()
        trader = _make_trader(broker)
        shorts = []

        async def fake_short_sell(symbol, price, confidence, strategy=""):
            shorts.append(symbol)

        async def fake_check_risk_stops():
            pass

        trader._handle_short_sell = fake_short_sell
        trader._check_risk_stops = fake_check_risk_stops
        trader.mean_reversion = MagicMock()
        trader.mean_reversion.generate_signals.return_value = []
        trader.liquidity_sweep = MagicMock()
        trader.liquidity_sweep.generate_signals.return_value = []
        monkeypatch.setattr(
            turbo_trader, "_generate_momentum_signals",
            lambda data, symbol, **k: [
                Signal(symbol=symbol, timestamp=pd.Timestamp("2026-01-01 10:00"),
                       signal_type=SignalType.SELL, confidence=0.9,
                       metadata={"strategy": "momentum", "rsi": 30.0}),
            ],
        )

        class FakeProvider:
            async def fetch_bars(self, *a, **k):
                return type("MDF", (), {"df": _downtrend_df()})()

        trader.provider = FakeProvider()
        monkeypatch.setattr(turbo_trader, "ENABLE_SHORT_SELLING", False)
        await trader._tick(1)
        assert shorts == []

    @pytest.mark.asyncio
    async def test_eod_liquidation_covers_shorts(self):
        """EOD must buy-to-cover negative-qty broker positions."""
        broker = FakeBroker(positions=[{
            "symbol": "SOXL", "qty": -50.0, "avg_entry_price": 40.0,
            "current_price": 38.0, "market_value": -1900.0, "unrealized_pl": 100.0,
        }])
        trader = _make_trader(broker)
        trader.pm.open_position("SOXL", -50.0, 40.0)
        await trader._eod_liquidate()
        assert len(broker.orders) == 1
        assert broker.orders[0].side == OrderSide.BUY
        assert broker.orders[0].quantity == 50.0
        assert trader.pm.get_positions().get("SOXL") is None

    @pytest.mark.asyncio
    async def test_risk_stops_short_stop_loss_on_rise(self):
        broker = FakeBroker()
        trader = _make_trader(broker)
        trader.pm.open_position("SOXL", -100.0, 40.0)
        trader._entry_times["SOXL"] = datetime.now(timezone.utc)

        class FakeProvider:
            async def fetch_bars(self, *a, **k):
                price = 43.0  # +7.5% above entry → exceeds 6% short stop-loss
                df = _ohlcv([price, price])
                return type("MDF", (), {"df": df})()

        trader.provider = FakeProvider()
        await trader._check_risk_stops()
        assert len(broker.orders) == 1
        assert broker.orders[0].side == OrderSide.BUY  # covered the short

    @pytest.mark.asyncio
    async def test_risk_stops_short_take_profit_on_fall(self):
        broker = FakeBroker()
        trader = _make_trader(broker)
        trader.pm.open_position("SOXL", -100.0, 40.0)
        trader._entry_times["SOXL"] = datetime.now(timezone.utc)

        class FakeProvider:
            async def fetch_bars(self, *a, **k):
                price = 36.0  # -10% below entry → exceeds 8% short take-profit
                df = _ohlcv([price, price])
                return type("MDF", (), {"df": df})()

        trader.provider = FakeProvider()
        await trader._check_risk_stops()
        assert len(broker.orders) == 1
        assert broker.orders[0].side == OrderSide.BUY
        assert trader.pm.get_positions().get("SOXL") is None
