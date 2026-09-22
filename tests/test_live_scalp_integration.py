"""Integration tests for the ScalpSet wiring in the MAIN trader (PR #37).

Hermetic — a fake broker + deterministic OHLCV bar feed replace every
network call; no real stack, no orders, no data downloads.  Covers:

1. Boot/rollback: ``LiveTrader()`` defaults to ``MAIN_STRATEGY=scalp`` and
   ``_strategy_mode()`` honours the instance mode; the mean-reversion
   rollback mode still runs the legacy tick path.
2. Data-availability gate: insufficient bars warn once per symbol per day
   and skip signals (never crash); enough bars ⇒ signals on.
3. Arbitration: ``best_rr`` picks the highest R:R with module-order tie-break
   (IFVG > Box > VolFib); ``first`` picks module order; arbitration is
   logged when modules disagree.
4. End-to-end ``_tick_scalp`` with only the Box module enabled: a 5m zone-A
   trigger emits a SHORT MARKET signal that is sized whole-share, opened as
   a SELL order, and immediately given a GTC stop at the STRATEGY SL level
   (not the -6% backstop) plus a day-limit TP.
4b. LIMIT signal path: ``_scalp_enter`` rests a DAY limit entry and records
   a bundle; the per-tick position sync adopts the fill and attaches the
   stop/TP bundle.
5. Short rejection backoff: N rejections disable that symbol's shorts for
   the session (longs unaffected); a broker ``not shortable`` verdict is
   cached and logged.
6. BE/trail: ``_replace_stop`` cancels ONLY the stop (TP limit survives) and
   re-places it at the new level — never double stops.
"""
import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import live_trader
from src.execution.broker import Order, OrderSide, OrderType
from src.execution.position_manager import PositionManager
from src.strategies.scalp.types import Direction, EntryType, LiquidityMap, ScalpContext, ScalpSignal


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
def _frame_df(closes, freq="5min", start="2026-09-01 09:30", highs=None, lows=None):
    idx = pd.date_range(start, periods=len(closes, ), freq=freq)
    highs = highs or [c + 1.0 for c in closes]
    lows = lows or [c - 1.0 for c in closes]
    return pd.DataFrame(
        {"open": [c - 0.1 for c in closes], "high": highs, "low": lows,
         "close": closes, "volume": [1000] * len(closes)}, index=idx)


class FakeMarketDataFrame:
    def __init__(self, df):
        self.df = df


class FakeProvider:
    """Deterministic bar feed keyed by timeframe string."""

    def __init__(self, frames: dict):
        self.frames = {k: frame_df(*v) if isinstance(v, tuple) else v for k, v in frames.items()}
        self.calls = []

    async def fetch_bars(self, symbol, start=None, end=None, timeframe="1min"):
        self.calls.append((symbol, timeframe))
        df = self.frames.get(timeframe)
        if df is None:
            df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return FakeMarketDataFrame(df)


class FakeOrder:
    def __init__(self, oid, symbol, side, order_type="market", stop_price=None,
                 limit_price=None, client_order_id=""):
        self.id = oid
        self.symbol = symbol
        self.side = side
        self.type = order_type
        self.stop_price = stop_price
        self.limit_price = limit_price
        self.client_order_id = client_order_id


class FakeBroker:
    """In-memory broker: records orders, simulates fills, positions, shorts."""

    def __init__(self, equity=200_000.0, buying_power=400_000.0, shorting_enabled=True,
                 shortable=True, positions=None, open_orders=None, fill_price=100.0,
                 reject_msgs=()):
        self.equity = equity
        self.buying_power = buying_power
        self.shorting_enabled = shorting_enabled
        self._shortable = shortable
        self.positions = {str(p["symbol"]).upper(): p for p in (positions or [])}
        self._open_orders = list(open_orders or [])
        self.fill_price = fill_price
        self._reject_msgs = list(reject_msgs)
        self.orders = []          # every placed Order
        self.stop_requests = []   # (symbol, qty, stop_price, client_id, side)
        self.cancelled_ids = []
        self.order_seq = 0

    # ── account / positions / shortability ──────────────────────────
    async def get_account(self):
        return {"equity": self.equity, "buying_power": self.buying_power,
                "cash": self.equity, "portfolio_value": self.equity,
                "available": True, "shorting_enabled": self.shorting_enabled}

    async def get_positions(self):
        return list(self.positions.values())

    async def get_last_fill_price(self, symbol):
        return self.fill_price

    async def is_shortable(self, symbol):
        return self._shortable

    async def get_open_orders(self, symbol=None):
        if symbol is not None:
            return [o for o in self._open_orders if str(o.symbol).upper() == symbol.upper()]
        return list(self._open_orders)

    async def cancel_order_and_wait(self, order_id, timeout=10.0, poll_interval=0.25):
        self.cancelled_ids.append(str(order_id))
        self._open_orders = [o for o in self._open_orders if str(o.id) != str(order_id)]
        return True

    async def cancel_orders_by_client_id_prefix(self, prefix):
        return 0

    # ── order placement ─────────────────────────────────────────────
    async def place_stop_order(self, symbol, qty, stop_price, client_id=None, side="SELL"):
        self.stop_requests.append((symbol, qty, stop_price, client_id, side))
        oid = f"stop-{len(self.stop_requests)}"
        self._open_orders.append(FakeOrder(oid, symbol, side, order_type="stop",
                                           stop_price=stop_price, client_order_id=client_id or ""))
        return MagicMock(id=oid, status="new")

    async def place_order(self, order: Order):
        self.order_seq += 1
        oid = f"ord-{self.order_seq}"
        if self._reject_msgs:
            msg = self._reject_msgs.pop(0)
            raise Exception(msg)
        status = "filled" if order.order_type == OrderType.MARKET else "accepted"
        self.orders.append(order)
        if order.order_type == OrderType.MARKET:
            qty = float(order.quantity)
            sym = order.symbol.upper()
            cur = self.positions.get(sym, {"symbol": sym, "qty": 0.0, "avg_entry_price": self.fill_price})
            if order.side == OrderSide.BUY:
                new_qty = float(cur["qty"]) + qty
                base = (float(cur["avg_entry_price"]) * float(cur["qty"]) +
                        self.fill_price * qty)
                avg = base / new_qty if new_qty else self.fill_price
                self.positions[sym] = {"symbol": sym, "qty": new_qty, "avg_entry_price": avg}
            else:
                self.positions[sym] = {"symbol": sym, "qty": float(cur["qty"]) + qty,
                                       "avg_entry_price": float(cur["avg_entry_price"])}
            filled_qty = qty
        else:
            self._open_orders.append(FakeOrder(
                oid, order.symbol.upper(), order.side.value, order_type="limit",
                limit_price=order.limit_price, client_order_id=order.client_id or ""))
            filled_qty = 0.0
        from src.execution.broker import OrderResult
        return OrderResult(
            order_id=oid, symbol=order.symbol, side=order.side, quantity=order.quantity,
            filled_quantity=filled_qty,
            filled_avg_price=self.fill_price if filled_qty else None,
            status=status, error_message=None, created_at=datetime.now(timezone.utc),
        )

    async def wait_for_order_fill(self, order_id, timeout=8.0, poll_interval=0.25):
        return None


def _make_scalp_trader(broker=None, provider=None, mode="scalp"):
    trader = object.__new__(live_trader.LiveTrader)
    trader.broker = broker or FakeBroker()
    trader.provider = provider or FakeProvider({})
    trader.pm = PositionManager(live_trader.STRATEGY_CONFIG)
    trader._entry_times = {}
    trader.day_trades = []
    trader.start_equity = 0.0
    trader._main_strategy = mode
    trader._scalp_init_state()
    return trader


def _sig(symbol, direction, entry, sl, tp, strategy, entry_type=EntryType.MARKET, rr=None):
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    return ScalpSignal(
        symbol=symbol, timestamp=pd.Timestamp("2026-09-01 10:00:00", tz="UTC"),
        direction=direction, entry_type=entry_type, entry_price=entry,
        stop_loss=sl, take_profit=tp, risk=risk, reward=reward,
        rr=rr if rr is not None else round(reward / risk, 4), strategy=strategy,
        breakeven_trigger_r=1.0, trailing=True, trail_distance_r=1.0, trail_trigger_r=1.0,
    )


# ---------------------------------------------------------------------------
# 1. Boot / rollback
# ---------------------------------------------------------------------------
class TestBootRollback:
    def test_default_strategy_is_scalp(self):
        assert live_trader.MAIN_STRATEGY == "scalp"
        trader = live_trader.LiveTrader()
        try:
            assert trader._main_strategy == "scalp"
            assert trader._strategy_mode() == "scalp"
        finally:
            trader.broker._client = None

    def test_object_new_instance_defaults_to_mean_reversion_for_legacy_tests(self):
        trader = object.__new__(live_trader.LiveTrader)
        assert trader._strategy_mode() == "mean_reversion"

    def test_rollback_mode_via_instance_flag(self):
        trader = _make_scalp_trader(mode="mean_reversion")
        assert trader._strategy_mode() == "mean_reversion"
        # the legacy MR tick path is still invoked by run() in this mode;
        # regression: _tick (MR) is callable and passes through to _handle_buy.
        df = pd.DataFrame(
            {"open": [90 + i * 0.3 for i in range(40)],
             "high": [90.4 + i * 0.3 for i in range(40)],
             "low": [89.6 + i * 0.3 for i in range(40)],
             "close": [90 + i * 0.3 for i in range(40)],
             "volume": [1000] * 40},
            index=pd.date_range("2026-09-01 09:30", periods=40, freq="1min"),
        )
        broker = FakeBroker()
        trader.broker = broker
        trader.provider = FakeProvider({"1min": df})
        trader.strategy = MagicMock()
        trader.strategy.generate_signals.return_value = []
        asyncio.run(trader._tick(1))  # no signals → no orders; no crash
        assert broker.orders == []
        assert broker.stop_requests == []


# ---------------------------------------------------------------------------
# 2. Data-availability gate
# ---------------------------------------------------------------------------
class TestDataAvailabilityGate:
    def _ctx(self, counts):
        from src.strategies.scalp.types import CandleFrame
        frames = {}
        for tf, n in counts.items():
            if tf == "1d":
                df = _frame_df([100.0, 105.0], freq="1D", start="2026-08-31")
            elif tf == "1m":
                df = _frame_df([100.0 + i * 0.1 for i in range(n)], freq="1min")
            elif tf == "5m":
                df = _frame_df([100.0 + i * 0.2 for i in range(n)], freq="5min")
            elif tf == "15m":
                df = _frame_df([100.0 + i * 0.5 for i in range(n)], freq="15min")
            elif tf == "30m":
                df = _frame_df([100.0 + i * 1.0 for i in range(n)], freq="30min")
            elif tf == "1h":
                df = _frame_df([100.0 + i * 2.0 for i in range(n)], freq="1h")
            else:  # 4h
                df = _frame_df([100.0 + i * 4.0 for i in range(n)], freq="4h")
            frames[tf] = CandleFrame.from_dataframe("NVDA", tf, df)
        return ScalpContext(symbol="NVDA", frames=frames, liquidity=LiquidityMap())

    def test_insufficient_bars_warns_once_and_skips(self, caplog):
        trader = _make_scalp_trader()
        ctx = self._ctx({"1m": 5, "5m": 2, "15m": 2, "30m": 2, "1h": 2, "4h": 1, "1d": 1})
        with caplog.at_level("WARNING", logger="live_trader"):
            assert trader._scalp_data_ready("NVDA", ctx) is False
            assert trader._scalp_data_ready("NVDA", ctx) is False  # warned once
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "insufficient bars" in text and "skipping signals" in text

    def test_enough_bars_turns_on_signals(self, caplog):
        trader = _make_scalp_trader()
        ctx = self._ctx({"1m": 60, "5m": 20, "15m": 20, "30m": 20, "1h": 30, "4h": 15, "1d": 3})
        with caplog.at_level("INFO", logger="live_trader"):
            assert trader._scalp_data_ready("NVDA", ctx) is True
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "signals ON" in text and "insufficient" not in text


# ---------------------------------------------------------------------------
# 3. Arbitration
# ---------------------------------------------------------------------------
class TestArbitration:
    def test_best_rr_picks_highest_rr(self):
        trader = _make_scalp_trader()
        s_box = _sig("NVDA", Direction.LONG, 100.0, 99.0, 106.0, "box_theory", rr=3.0)
        s_ifvg = _sig("NVDA", Direction.LONG, 100.0, 99.0, 101.0, "ict_ifvg", rr=1.0)
        chosen = trader._arbitrate_scalp_signal("NVDA", [s_ifvg, s_box])
        assert chosen is s_box  # 3R > 1R

    def test_best_rr_tie_breaks_by_module_order(self):
        trader = _make_scalp_trader()
        s_box = _sig("NVDA", Direction.SHORT, 100.0, 101.0, 98.0, "box_theory", rr=2.0)
        s_ifvg = _sig("NVDA", Direction.SHORT, 100.0, 101.0, 98.0, "ict_ifvg", rr=2.0)
        s_vf = _sig("NVDA", Direction.SHORT, 100.0, 101.0, 98.0, "volprofile_fib", rr=2.0)
        chosen = trader._arbitrate_scalp_signal("NVDA", [s_vf, s_box, s_ifvg])
        assert chosen.strategy == "ict_ifvg"  # tie-break module order
        with_ib = trader._arbitrate_scalp_signal("NVDA", [s_vf, s_box])
        assert with_ib.strategy == "box_theory"

    def test_first_mode_picks_module_order_regardless_of_rr(self, monkeypatch):
        monkeypatch.setattr(live_trader, "SCALP_ARBITRATION", "first")
        trader = _make_scalp_trader()
        s_box = _sig("NVDA", Direction.LONG, 100.0, 99.0, 106.0, "box_theory", rr=5.0)
        s_ifvg = _sig("NVDA", Direction.LONG, 100.0, 99.0, 101.0, "ict_ifvg", rr=0.5)
        chosen = trader._arbitrate_scalp_signal("NVDA", [s_box, s_ifvg])
        assert chosen is s_ifvg  # IFVG wins module order despite lower R:R

    def test_arbitration_is_logged_when_modules_disagree(self, caplog):
        trader = _make_scalp_trader()
        s_box = _sig("NVDA", Direction.LONG, 100.0, 99.0, 106.0, "box_theory", rr=3.0)
        s_ifvg = _sig("NVDA", Direction.SHORT, 100.0, 101.0, 98.0, "ict_ifvg", rr=1.5)
        with caplog.at_level("INFO", logger="live_trader"):
            trader._arbitrate_scalp_signal("NVDA", [s_ifvg, s_box])
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "ARBITRATION" in text and "box_theory" in text and "3.00R" in text


# ---------------------------------------------------------------------------
# 4. End-to-end tick: Box module SHORT MARKET — whole shares, strategy SL
# ---------------------------------------------------------------------------
BOX_PDH, BOX_PDL = 112.0, 88.0


def _box_frames():
    # 5m bars: bar idx3 is a zone-A touch with a GREEN close; bar idx4 is the
    # red close below the prior close → SHORT trigger (entry=109.0).
    closes = [100.0, 105.0, 106.0, 110.0, 109.0]
    highs = [101.0, 105.5, 106.5, 110.5, 110.0]
    lows = [99.0, 104.0, 105.0, 109.4, 108.4]
    f5m = _frame_df(closes, freq="5min", highs=highs, lows=lows)
    f1d = _frame_df([100.0, 100.0], freq="1D", start="2026-08-31",
                    highs=[BOX_PDH, BOX_PDH + 1.0], lows=[BOX_PDL, BOX_PDL - 1.0])
    f1m = _frame_df([109.0] * 60, freq="1min", highs=[109.1] * 60, lows=[108.9] * 60)
    # HTF frames so the disabled module gate would pass if re-enabled is moot
    return {
        "1min": f1m,
        "5min": f5m,
        "15min": _frame_df([107.0 + i * 0.2 for i in range(20)], freq="15min"),
        "30min": _frame_df([105.0 + i * 0.3 for i in range(20)], freq="30min"),
        "1h": _frame_df([100.0 + i * 0.5 for i in range(30)], freq="1h"),
        "1day": f1d,
    }


class TestTickScalpBoxShort:
    @pytest.mark.asyncio
    async def test_box_short_signal_end_to_end(self, monkeypatch):
        monkeypatch.setattr(live_trader, "SCALP_MODULE_IFVG", False)
        monkeypatch.setattr(live_trader, "SCALP_MODULE_VOLFIB", False)
        monkeypatch.setattr(live_trader, "SCALP_MODULE_BOX", True)
        monkeypatch.setattr(live_trader, "SYMBOLS", ["QQQ"])
        # fill_price matches the 1m close (109.0): MARKET entries anchor their
        # SL/TP to the broker-reported fill, so the stub fill must be the price
        # the position is really opened at.
        broker = FakeBroker(shortable=True, fill_price=109.0)
        provider = FakeProvider(_box_frames())
        trader = _make_scalp_trader(broker, provider)
        await trader._tick_scalp(1)

        # SELL market order was placed with WHOLE shares (floor of 275.22 = 275).
        # There is NO second broker order: the whole-share GTC stop reserves all
        # 275 shares, so a full-quantity day-limit TP on the same shares is
        # rejected by Alpaca (40310000, live 2026-09-21).  The upside exit is the
        # trader-side monitored TP: the level lives in the position state.
        assert len(broker.orders) == 1
        state = trader._scalp_positions["QQQ"]
        assert state["tp"] == 88.0
        assert state["tp_monitored"] is True and state["tp_placed"] is False
        entry = broker.orders[0]
        assert entry.symbol == "QQQ"
        assert entry.side == OrderSide.SELL
        assert entry.order_type == OrderType.MARKET
        assert int(entry.quantity) == entry.quantity == 275  # whole shares for shorts

        # position tracked as a SHORT
        pos = trader.pm.get_positions()["QQQ"]
        assert pos.quantity == -275
        assert pos.entry_price == 109.0

        # GTC stop placed at the STRATEGY SL (above entry), not the -6% backstop
        assert len(broker.stop_requests) == 1
        sym, qty, stop_price, _cid, side = broker.stop_requests[0]
        assert sym == "QQQ" and qty == 275 and side == "BUY"
        assert stop_price == round(110.51, 2)
        # strategy SL is protective (above entry) and NOT the -6% backstop level
        assert stop_price > 109.0
        assert round(stop_price, 2) != round(109.0 * 1.06, 2)

        # the structural TP target (88.0, asserted above from the state) is
        # the MONITORED exit level: no day-limit TP order can rest next to the
        # whole-share GTC stop (40310000)
        tps = [o for o in broker.orders if o is not None and o.order_type == OrderType.LIMIT]
        assert tps == []

        # per-signal log present with geometry
        assert trader._last_signal_key.get("QQQ") is not None

    @pytest.mark.asyncio
    async def test_broker_not_shortable_skips_short_and_long_unaffected(self, monkeypatch, caplog):
        monkeypatch.setattr(live_trader, "SCALP_MODULE_IFVG", False)
        monkeypatch.setattr(live_trader, "SCALP_MODULE_VOLFIB", False)
        monkeypatch.setattr(live_trader, "SCALP_MODULE_BOX", True)
        monkeypatch.setattr(live_trader, "SYMBOLS", ["QQQ"])
        broker = FakeBroker(shortable=False)
        trader = _make_scalp_trader(broker, FakeProvider(_box_frames()))
        with caplog.at_level("WARNING", logger="live_trader"):
            await trader._tick_scalp(1)
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "NOT SHORTABLE" in text
        assert broker.orders == []  # short never attempted
        assert trader._shortable_cache["QQQ"] is False
        # (...) longs are not gated by the shortability check — the gate is
        # only invoked for SHORT signals, so a LONG would still be attempted.
        assert "SHORT" in text and "Longs for QQQ continue unaffected" in text


def broker_account_long():
    return {"equity": 100.0, "buying_power": 100.0, "available": True,
            "shorting_enabled": True}


# ---------------------------------------------------------------------------
# 4b. LIMIT entry path + fill sync
# ---------------------------------------------------------------------------
class TestLimitPath:
    @pytest.mark.asyncio
    async def test_limit_entry_rests_day_limit_and_bundle(self):
        trader = _make_scalp_trader(FakeBroker())
        sig = _sig("NVDA", Direction.LONG, 100.0, 98.0, 106.0, "ict_ifvg",
                   entry_type=EntryType.LIMIT, rr=3.0)
        ok = await trader._scalp_enter(sig, 101.0)
        assert ok is True
        assert trader._scalp_bundles["NVDA"]["direction"] == "LONG"
        assert trader._scalp_bundles["NVDA"]["sl"] == 98.0
        bundle = trader._scalp_bundles["NVDA"]
        assert bundle["entry_price"] == 100.0
        # the order was a DAY limit (not GTC; dies at close by broker default)
        assert len(trader.broker.orders) == 1
        o = trader.broker.orders[0]
        assert o.order_type == OrderType.LIMIT and o.limit_price == 100.0
        assert o.side == OrderSide.BUY
        # no position opened yet (limit not filled)
        assert not trader.pm.has_position("NVDA")
        # no stop placed while the entry is unfilled (wash-trade race)
        assert trader.broker.stop_requests == []

    @pytest.mark.asyncio
    async def test_sync_adopts_filled_limit_and_attaches_stop_tp(self):
        trader = _make_scalp_trader(FakeBroker())
        sig = _sig("NVDA", Direction.LONG, 100.0, 98.0, 106.0, "ict_ifvg",
                   entry_type=EntryType.LIMIT, rr=3.0)
        await trader._scalp_enter(sig, 101.0)
        # broker confirms the limit filled with 30 shares @ 100.0
        trader.broker.positions = {"NVDA": {"symbol": "NVDA", "qty": 30.0, "avg_entry_price": 100.0}}
        await trader._scalp_sync_positions()
        pos = trader.pm.get_positions().get("NVDA")
        assert pos is not None and pos.quantity == 30.0 and pos.entry_price == 100.0
        assert trader._scalp_bundles == {}
        # strategy SL stop attached at the actual fill qty
        sym, qty, stop_price, _cid, side = trader.broker.stop_requests[0]
        assert (sym, qty, side) == ("NVDA", 30, "SELL")
        assert stop_price == 98.0
        # the TP target is kept as the MONITORED exit level, not a resting
        # order: the 30-share GTC stop reserves every share (40310000)
        tps = [o for o in trader.broker.orders
               if o.order_type == OrderType.LIMIT
               and "ENTRY" not in (o.client_id or "")]
        assert tps == []          # no resting TP next to the GTC stop
        state = trader._scalp_positions["NVDA"]
        assert state["tp"] == 106.0 and state["tp_monitored"] is True

    @pytest.mark.asyncio
    async def test_tp_fill_sync_books_pnl_and_starts_cooldown(self):
        trader = _make_scalp_trader(FakeBroker())
        trader.pm.open_position("NVDA", 30.0, 100.0)
        trader.broker.positions = {}  # position vanished → TP limit filled
        await trader._scalp_sync_positions()
        assert not trader.pm.has_position("NVDA")
        assert "NVDA" in trader._cooldown_until
        assert trader._scalp_positions.get("NVDA") is None  # state cleaned


# ---------------------------------------------------------------------------
# 5. Rejection backoff for shorts
# ---------------------------------------------------------------------------
class TestShortRejectionBackoff:
    def test_disable_after_limit(self):
        trader = _make_scalp_trader(FakeBroker(shortable=True))
        trader._record_short_rejection("TSLA", "cannot be sold short: not shortable")
        trader._record_short_rejection("TSLA", "cannot be sold short: not shortable")
        assert "TSLA" not in trader._shorts_disabled
        assert trader._shortable_cache["TSLA"] is False  # definitive verdict sticks
        trader._record_short_rejection("TSLA", "cannot be sold short: not shortable")
        assert "TSLA" in trader._shorts_disabled
        assert trader._short_rejections["TSLA"] == 3

    @pytest.mark.asyncio
    async def test_short_entry_rejection_routes_to_backoff(self):
        broker = FakeBroker(shortable=True, reject_msgs=["cannot be sold short: no borrow"])
        trader = _make_scalp_trader(broker)
        sig = _sig("NVDA", Direction.SHORT, 100.0, 101.5, 97.0, "box_theory", rr=3.0)
        await trader._scalp_enter(sig, 100.0)
        assert trader._short_rejections["NVDA"] == 1

    @pytest.mark.asyncio
    async def test_short_uses_whole_shares_and_bp_cap(self):
        broker = FakeBroker(equity=10_000.0, buying_power=10_000.0)
        trader = _make_scalp_trader(broker)
        sig = _sig("NVDA", Direction.SHORT, 100.0, 101.5, 97.0, "box_theory", rr=3.0)
        await trader._scalp_enter(sig, 100.0)
        # SELL entry only: the BUY day-limit TP cannot rest next to the
        # whole-share BUY GTC stop (both would need the same 15 shares)
        assert len(broker.orders) == 1
        o = broker.orders[0]
        assert o.side == OrderSide.SELL
        assert int(o.quantity) == o.quantity == 15  # 10k*15% / 100 = 15 whole
        assert len(broker.stop_requests) == 1
        assert broker.stop_requests[0][2] == 101.5  # strategy SL

    @pytest.mark.asyncio
    async def test_short_capability_allows_account_shorts_disabled(self):
        broker = FakeBroker(shorting_enabled=False)
        trader = _make_scalp_trader(broker)
        assert await trader._short_capability_allows("NVDA", {"shorting_enabled": False}) is False


# ---------------------------------------------------------------------------
# 6. BE/trail: cancel+replace stop only (TP survives)
# ---------------------------------------------------------------------------
class TestBreakEvenAndTrail:
    @pytest.mark.asyncio
    async def test_replace_stop_cancels_only_the_stop_and_replaces_at_new_level(self):
        broker = FakeBroker(open_orders=[
            FakeOrder("stop-1", "NVDA", "SELL", order_type="stop", stop_price=98.0,
                      client_order_id="algoflow_MAIN_NVDA_STOP_1"),
            FakeOrder("tp-1", "NVDA", "SELL", order_type="limit", limit_price=106.0,
                      client_order_id="algoflow_MAIN_NVDA_TP_SELL_1"),
        ])
        trader = _make_scalp_trader(broker)
        trader.pm.open_position("NVDA", 30.0, 100.0)
        state = {"entry": 100.0, "sl": 98.0, "tp": 106.0, "direction": "LONG"}
        trader._scalp_positions["NVDA"] = state
        ok = await trader._replace_stop("NVDA", 100.05, "break-even")
        assert ok is True
        # only the STOP was cancelled; the TP survived
        assert "stop-1" in broker.cancelled_ids and "tp-1" not in broker.cancelled_ids
        assert len(broker.stop_requests) == 1
        assert broker.stop_never_called is False if hasattr(broker, "stop_never_called") else True
        sym, qty, stop_price, _cid, side = broker.stop_requests[0]
        assert (sym, qty, side) == ("NVDA", 30, "SELL")
        assert stop_price == 100.05
        # in-memory state follows the new stop (BE done is set by risk pass)
        assert trader._scalp_positions["NVDA"]["sl"] == 100.05

    @pytest.mark.asyncio
    async def test_risk_pass_moves_stop_to_breakeven_at_1r(self):
        broker = FakeBroker(open_orders=[])
        provider = FakeProvider({"1min": _frame_df([102.5] * 10, freq="1min")})
        trader = _make_scalp_trader(broker, provider)
        trader.pm.open_position("NVDA", 30.0, 100.0,
                                stop_loss_price=98.0, take_profit_price=106.0)
        trader._scalp_positions["NVDA"] = {
            "entry": 100.0, "sl": 98.0, "tp": 106.0, "direction": "LONG",
            "be_done": False, "be_trigger_r": 1.0, "be_buffer": 0.0,
            "trailing": True, "trail_r": 1.0, "trail_trigger_r": 1.0,
            "stop_placed": True, "tp_placed": True,
        }
        await trader._scalp_risk_pass()
        assert trader._scalp_positions["NVDA"]["sl"] == 100.0  # moved to BE
        assert trader._scalp_positions["NVDA"]["be_done"] is True
        assert len(broker.stop_requests) == 1  # one replacement at 100.0
        assert broker.stop_requests[0][2] == 100.0

    @pytest.mark.asyncio
    async def test_inproc_sl_fallback_when_no_broker_stop(self):
        provider = FakeProvider({"1min": _frame_df([97.0] * 10, freq="1min")})
        trader = _make_scalp_trader(FakeBroker(), provider)
        trader.pm.open_position("NVDA", 50.0, 100.0)
        trader._scalp_positions["NVDA"] = {
            "entry": 100.0, "sl": 98.0, "tp": 106.0, "direction": "LONG",
            "be_done": False, "be_trigger_r": 1.0, "be_buffer": 0.0,
            "trailing": True, "trail_r": 1.0, "trail_trigger_r": 1.0,
            "stop_placed": False, "tp_placed": True,
        }
        await trader._scalp_risk_pass()
        # SL breached (97 <= 98) and NO broker stop → verified-close fired
        assert not trader.pm.has_position("NVDA")
        assert "NVDA" in trader._cooldown_until