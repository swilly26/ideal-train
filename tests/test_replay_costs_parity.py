"""Parity: the extracted cost model must bill exactly what the turbo replay bills.

Stage 1 of the strategy search reuses the turbo replay's honest cost lines
instead of re-deriving them (brief §4.1).  Reuse is only real if the same
``(symbol, price, size, timestamp)`` costs the same through both paths, so this
file pins them together three ways:

1. the five cost fields of a ``TurboConfig`` are exactly the model's fields,
2. ``slip / fill / fees`` agree for every variant and a spread of prices/sizes,
3. a whole round trip — the arithmetic ``TurboReplay._close`` composes from
   signal prices to net P&L and same-fill drag — agrees to the cent.

If someone edits one path and not the other, this file goes red.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtesting.replay_costs import (  # noqa: E402
    COST_FIELDS,
    CostModel,
    cost_fields_match_config,
)
from src.backtesting.turbo_engine import TurboConfig, TurboReplay  # noqa: E402

PRICES = (4.75, 12.50, 30.00, 61.33, 173.50, 500.00, 3000.00)
QTYS = (1.0, 7.0, 250.0, 1250.5)
VARIANTS = ("baseline", "zero_cost", "pessimistic")


def _config(variant: str) -> TurboConfig:
    cfg = TurboConfig.classic()
    if variant == "baseline":
        return cfg
    if variant == "zero_cost":
        return cfg.zero_cost()
    return TurboConfig.pessimistic()


def _model(variant: str) -> CostModel:
    return {"baseline": CostModel.baseline,
            "zero_cost": CostModel.zero_cost,
            "pessimistic": CostModel.pessimistic}[variant]()


def _synthetic_frame(sessions: int = 2, bars: int = 45) -> pd.DataFrame:
    """Two flat 45-bar sessions — enough for ``TurboReplay._prep`` to run."""
    idx = []
    for d in ("2025-01-02", "2025-01-03"):
        idx += [pd.Timestamp(f"{d} 09:{30 + i // 60:02d}:{i % 60:02d}") for i in range(bars)]
    n = len(idx)
    px = np.full(n, 25.0)
    return pd.DataFrame({"open": px, "high": px + 0.05, "low": px - 0.05,
                         "close": px, "volume": np.full(n, 1000.0)},
                        index=pd.DatetimeIndex(idx, name="ts"))


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_cost_fields_are_the_same_five_numbers(variant):
    cfg = _config(variant)
    model = _model(variant)
    assert cost_fields_match_config(model, cfg)
    for f in COST_FIELDS:
        assert float(getattr(cfg, f)) == float(getattr(model, f))


def test_baseline_model_is_the_turbo_defaults():
    """The numbers the plan quotes every result at: 2 bps or 1 cent per fill."""
    m = CostModel.baseline()
    assert (m.slippage_pct, m.slippage_abs, m.half_spread_pct,
            m.commission_per_share, m.commission_pct) == (0.0002, 0.01, 0.0, 0.0, 0.0)
    assert cost_fields_match_config(m, TurboConfig.classic())


def test_pessimistic_matches_the_runner_variant():
    from scripts.run_turbo_backtest import build_config
    assert cost_fields_match_config(CostModel.pessimistic(), build_config("classic", "pessimistic"))
    assert cost_fields_match_config(CostModel.zero_cost(), build_config("classic", "zero_cost"))


@pytest.mark.parametrize("variant", VARIANTS)
def test_slip_fill_and_fees_agree_with_the_engine(variant):
    engine = TurboReplay({"SOXL": _synthetic_frame()}, _config(variant))
    model = _model(variant)
    for price in PRICES:
        assert engine._slip(price) == pytest.approx(model.slip_per_share(price), abs=1e-12)
        for is_buy in (True, False):
            assert engine._market_fill(price, is_buy) == pytest.approx(
                model.fill_price(price, is_buy), abs=1e-12)
        for qty in QTYS:
            assert engine._fees(qty, price) == pytest.approx(model.fees(qty, price), abs=1e-12)


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_round_trip_costs_the_same_through_both_paths(variant):
    """Recompose ``TurboReplay._close`` from the engine's own primitives."""
    engine = TurboReplay({"SOXL": _synthetic_frame()}, _config(variant))
    model = _model(variant)
    for side, entry_signal, exit_signal in ((1, 30.00, 31.20), (1, 173.50, 168.00),
                                            (-1, 61.33, 60.10), (-1, 12.50, 13.40)):
        for qty in QTYS:
            entry_fill = engine._market_fill(entry_signal, is_buy=side > 0)
            exit_fill = engine._market_fill(exit_signal, is_buy=side < 0)
            entry_fee = engine._fees(qty, entry_fill)
            exit_fee = engine._fees(qty, exit_fill)
            gross = side * (exit_fill - entry_fill) * qty
            expected = {
                "entry_fill": entry_fill,
                "exit_fill": exit_fill,
                "gross": gross,
                "fees": entry_fee + exit_fee,
                "net": gross - (entry_fee + exit_fee),
                "drag": qty * (engine._slip(entry_signal) + engine._slip(exit_signal))
                        + entry_fee + exit_fee,
            }
            got = model.round_trip(qty, entry_signal, exit_signal, side=side)
            for key, want in expected.items():
                assert got[key] == pytest.approx(want, abs=1e-9), (key, side, qty)


def test_a_short_round_trip_has_costs_the_same_sign_as_a_long_one():
    """Slippage is adverse whichever way the book is leaning."""
    m = CostModel.baseline()
    long_rt = m.round_trip(100, 30.0, 30.30, side=1)
    short_rt = m.round_trip(100, 30.0, 29.70, side=-1)
    assert long_rt["gross"] > 0 and short_rt["gross"] > 0
    assert long_rt["drag"] == pytest.approx(short_rt["drag"])
    assert long_rt["net"] == pytest.approx(short_rt["net"])
