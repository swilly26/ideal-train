"""The revision-3 engine deltas (brief §R3-7), each with a failing-first test.

Four behaviours, each of which fails on the tree this delta lands against:

* **(a) the hash covers the real channel.**  ``SearchConfig.params()`` used to
  merge ``extras`` **over** the real fields, so an ``extras`` entry shadowed a
  real field and two genuinely different configs produced the same params — and
  therefore the same hash, the same name and the same artefact.  The hash now
  takes the config with ``extras`` **nested** plus the **resolved signal dict**.
* **(b) the leverage guard is gross.**  The cash check was the **net** outlay, so
  a long+short basket opened two full notionals of gross exposure at ~$0 of
  outlay and passed a $100k equity check.  The limit is declared: 100 % of
  ``initial_equity`` on ``Σ|qty·fill|``.
* **(c) future-blindness is dense.**  One cut misses the failure mode; the cut
  set is now the first and last bar of every month plus every session's first
  two bars (never 09:30), it asserts on the declared feature frame as well as
  the arrays, and it adds the mechanical invariant
  ``build_features(df.iloc[:t]) == build_features(df).iloc[:t]``.
* **(d)** ``MIN_TRIPS_PER_MONTH = 5``: a month with one trip is a coin flip, not
  a positive fold.

The harness helpers live in ``tests.test_strategy_search_engine`` so the file
still imports against the tree this delta lands against.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtesting.replay_costs import CostModel  # noqa: E402
from src.backtesting.strategy_search import engine, families, screen  # noqa: E402
from src.backtesting.strategy_search import features as features_mod  # noqa: E402
from src.backtesting.strategy_search.engine import (  # noqa: E402
    Instrument,
    Leg,
    Market,
    SearchConfig,
    run_search,
)
from src.backtesting.strategy_search.features import FeatureBook  # noqa: E402
from tests.future_blindness import (  # noqa: E402
    assert_blind_to_the_future_dense,
    assert_features_causal,
    dense_cuts,
)
from tests.test_strategy_search_engine import (  # noqa: E402
    OPEN_MIN,
    flat_session,
    mk_frame,
    simple_cfg,
)

FLAT_PX = 100.0


# ── (a) the hash covers the real channel ───────────────────────────────


def _declaration_hash(family: str, cfg: SearchConfig, signal=None) -> str:
    """Hash over ``params()`` alone — the channel E2's defect came through."""
    return families.canonical_hash({"family": family, "config": cfg.params(),
                                    "signal": signal or {}})


def test_params_keeps_extras_nested_and_never_over_a_real_field():
    """An ``extras`` key must not shadow a real ``SearchConfig`` field."""
    cfg = SearchConfig(family="A", name="x", stop_pct=0.010,
                       extras={"stop_pct": 0.020, "stop_atr_mult": 1.0})
    p = cfg.params()
    assert p["stop_pct"] == 0.010, "a real field is never shadowed by extras"
    assert p["extras"] == {"stop_pct": 0.020, "stop_atr_mult": 1.0}
    assert "stop_atr_mult" not in p or p["extras"]["stop_atr_mult"] == 1.0


def test_two_configs_differing_only_in_an_extras_key_hash_differently():
    """The shadowing case: two different configs used to hash — and record — alike."""
    a = SearchConfig(family="A", name="x", stop_pct=0.010, extras={})
    b = SearchConfig(family="A", name="x", stop_pct=0.010,
                     extras={"stop_pct": 0.020})
    assert _declaration_hash("A", a) != _declaration_hash("A", b), \
        "merged extras: two different configs produced the same params hash"
    assert families.config_hash("A", a, {}) != families.config_hash("A", b, {})
    c = SearchConfig(family="A", name="x", stop_pct=0.010,
                     extras={"stop_atr_mult": 1.0})
    assert families.config_hash("A", a, {}) != families.config_hash("A", c, {})


def test_two_configs_differing_only_in_a_signal_param_hash_differently():
    """The resolved **signal** channel is part of the identity, not decoration."""
    m, aligned = _five_symbol_market()
    spec = dict(families.GRID_D[0])
    _cfg, _inst, res30 = families.build("D", spec, m, aligned)
    spec20 = dict(spec)
    spec20["signal"] = dict(spec["signal"], z_window=20)
    _cfg2, _inst2, res20 = families.build("D", spec20, m, aligned)
    assert families.resolved_hash("D", res30) != families.resolved_hash("D", res20)
    # the same config with the same signal resolves to the same hash
    assert families.resolved_hash("D", res30) == families.resolved_hash("D", res30)
    # and the hash is the name the config carries
    assert _cfg.name == families.resolved_name("D", res30)


def test_the_hash_of_a_grid_is_one_per_resolved_config():
    """``len({hash}) == len({name})`` for the round-2 grids (R3-7 E2)."""
    for family, grid in ((f, families.GRIDS[f]) for f in families.ROUND2_FAMILIES):
        hashes = [families.resolved_hash(family, families.resolve_spec(family, s))
                  for s in grid]
        names = [s["name"] for s in grid]
        assert len(set(hashes)) == len(hashes) == len(grid), family
        assert len(set(names)) == len(grid), family


# ── (b) the leverage guard is gross ────────────────────────────────────


def _basket(notional: float, initial_equity: float = 100_000.0):
    """One long+short pair trip on a flat tape at *notional* of gross target."""
    bars = flat_session("2025-01-02", FLAT_PX, 390)
    m = Market({"SOXL": mk_frame({"2025-01-02": bars}),
                "TQQQ": mk_frame({"2025-01-02": list(bars)})})
    legs = (Leg("SOXL", 1, 1.0), Leg("TQQQ", -1, 1.0))
    entry = np.zeros(m.n(), dtype=np.int8)
    entry[5] = 1
    inst = Instrument(key="SOXL/TQQQ", legs=legs, entry_dir=entry,
                      exit_now=np.full(m.n(), 2, dtype=np.int8),
                      valid=m.valid(legs), kind="pair")
    cfg = simple_cfg(allow_short=True, max_positions=4, notional_usd=notional,
                     initial_equity=initial_equity)
    return run_search(m, {"SOXL/TQQQ": inst}, cfg, CostModel.baseline())


def test_the_declared_gross_limit_is_100_percent_of_equity():
    assert engine.MAX_GROSS_LEVERAGE == 1.0
    res = _basket(50_000.0)
    assert res.stats["max_gross_leverage"] == 1.0
    assert res.stats["gross_limit_usd"] == pytest.approx(100_000.0)


def test_a_long_short_basket_over_the_gross_limit_is_refused():
    """$150k of gross at ~$0 of net outlay: the check must be gross."""
    res = _basket(150_000.0)
    assert res.stats["skipped"]["gross_leverage"] == 1
    assert res.stats["skipped"]["cash"] == 0, \
        "the net-outlay check never fired — the refusal is the gross one"
    assert res.stats["entries"] == 0
    assert len(res.trades) == 0
    assert res.stats["entries"] + res.stats["skipped_total"] == res.stats["signals"]


def test_a_basket_at_exactly_the_declared_limit_is_allowed():
    """The limit is ``>``, not ``>=``: 100 % of equity is inside the rule."""
    res = _basket(100_000.0)
    assert res.stats["skipped"]["gross_leverage"] == 0
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["gross_notional"] == pytest.approx(100_000.0)


def test_net_outlay_is_still_reported_beside_gross():
    """``net_outlay`` stays in the record, next to the gross it is *not* the guard on."""
    res = _basket(50_000.0)
    tr = res.trades.iloc[0]
    assert tr["gross_notional"] == pytest.approx(50_000.0)
    assert abs(tr["net_outlay"]) < 1_000.0, \
        "a long+short basket's net outlay is ~zero; the gross is the exposure"
    assert res.stats["gross_notional_per_trip"] == pytest.approx(50_000.0)
    assert abs(res.stats["net_outlay_per_trip"]) < 1_000.0


# ── (c) dense future-blindness ─────────────────────────────────────────


BLIND_DAYS = ("2025-01-30", "2025-01-31", "2025-02-03", "2025-02-04",
              "2025-02-05")


def r3_frames(n: int = 120) -> dict[str, pd.DataFrame]:
    """Two calendar months, five sessions, a rising-then-falling 4-symbol tape."""
    frames: dict[str, pd.DataFrame] = {}
    for j, sym in enumerate(("SOXL", "SPY")):
        sessions = {}
        for i, day in enumerate(BLIND_DAYS):
            drift = 0.50 if i < 3 else -0.40
            base = 100.0 + 2.0 * i + 5.0 * j
            sessions[day] = [(base + drift * k, base + drift * k + 0.05,
                              base + drift * k - 0.05, base + drift * k + 0.02)
                             for k in range(n)]
        frames[sym] = mk_frame(sessions)
    return frames


def _family_spec(family: str, cell: int = 0) -> dict:
    spec = dict(families.GRIDS[family][cell])
    spec["params"] = dict(spec["params"])
    spec["signal"] = dict(spec["signal"])
    spec["symbols"] = ["SOXL"]
    return spec


def test_the_dense_cut_set_covers_every_session_and_month():
    frames = r3_frames()
    axis = frames["SOXL"].index
    cuts = dense_cuts(axis)
    cset = set(cuts)
    assert cuts, "the dense cut set is empty"
    assert all((int(c.hour) * 60 + int(c.minute)) != OPEN_MIN for c in cuts), \
        "a cut at 09:30 is not a cut: that bar's close *is* the session open"
    months = sorted({str(c)[:7] for c in cuts})
    assert months == ["2025-01", "2025-02"]
    for month in months:
        m_axis = axis[axis.strftime("%Y-%m") == month]
        assert m_axis[-1] in cset, f"the last bar of {month} is not a cut"
        if (int(m_axis[0].hour) * 60 + int(m_axis[0].minute)) != OPEN_MIN:
            assert m_axis[0] in cset, f"the first bar of {month} is not a cut"
        else:
            assert m_axis[1] in cset, \
                f"{month} opens at 09:30, so the 09:31 cut must be there instead"
    for day in BLIND_DAYS:
        d_axis = axis[axis.normalize() == pd.Timestamp(day)]
        assert d_axis[1] in cset, f"the 09:31 bar of {day} is not a cut"
    assert len(cset) == len(cuts), "the cut set carries a duplicate"


@pytest.mark.parametrize("family,cell", [("A", 0), ("C", 1)])
def test_family_a_and_c_are_blind_to_the_future_at_every_dense_cut(family, cell):
    frames = r3_frames()
    out = assert_blind_to_the_future_dense(family, _family_spec(family, cell),
                                          frames)
    axis = frames["SOXL"].index
    assert out["cuts"] >= len(BLIND_DAYS), out
    assert out["cut_times"] == [str(c) for c in dense_cuts(axis)], out
    assert out["keys"] == ["SOXL"]
    assert out["feature_rows_checked"] > 0


def test_build_features_is_causal_on_a_prefix_at_every_dense_cut():
    """The mechanical invariant: no feature may be revised by a later bar."""
    frames = r3_frames()
    cuts = dense_cuts(frames["SOXL"].index)
    out = assert_features_causal(frames, cuts)
    assert out["checked"] >= 2 * len(cuts) - 2, \
        "each symbol/cut pair before the end of the frame must be checked"


def test_the_dense_signal_harness_catches_a_planted_leak(monkeypatch):
    """A fixture that cannot fail proves nothing (round 1's lesson).

    The planted family reads the **next bar's close** off the market arrays
    directly — the purest form of a signal that peeks forward, with no feature
    column involved — so the *signal* comparison has to be what catches it.
    """
    frames = r3_frames()

    def peeking(spec, market, feats):
        n = market.n()
        out = {}
        for sym in spec["symbols"]:
            c = market.c[sym]
            entry = np.zeros(n, dtype=np.int8)
            entry[:-1][c[1:] > c[:-1]] = 1          # reads bar k+1 at bar k
            out[sym] = Instrument(key=sym, legs=(Leg(sym, 1, 1.0),),
                                  entry_dir=entry,
                                  exit_now=np.zeros(n, dtype=np.int8),
                                  valid=market.valid((Leg(sym, 1, 1.0),)))
        return out

    monkeypatch.setitem(families.BUILDERS, "LEAK", peeking)
    monkeypatch.setitem(families.SIGNAL_KEYS, "LEAK", ())
    monkeypatch.setitem(families.SIGNAL_REQUIRED, "LEAK", ())
    spec = dict(name="leak", symbols=["SOXL"],
                params=families._base_params(), extras={}, signal={})
    with pytest.raises(AssertionError, match="reads the future"):
        assert_blind_to_the_future_dense("LEAK", spec, frames,
                                         feature_invariant=False)


def test_the_features_invariant_catches_a_planted_whole_session_column(monkeypatch):
    """The invariant must fail on a column that a later bar revises."""
    frames = r3_frames()
    real = features_mod.build_features

    def leaky_build(df):
        out = real(df)
        day = pd.Series(df.index.normalize(), index=df.index)
        out["leak_session_last_close"] = df["close"].groupby(day).transform("last")
        return out

    monkeypatch.setattr(features_mod, "build_features", leaky_build)
    with pytest.raises(AssertionError, match="after the cut"):
        assert_features_causal(frames, dense_cuts(frames["SOXL"].index))


# ── (d) MIN_TRIPS_PER_MONTH = 5 ────────────────────────────────────────


def test_min_trips_per_month_is_five():
    assert screen.MIN_TRIPS_PER_MONTH == 5


def _months(trips_per_month: int, bps: float = 1.0) -> list[dict]:
    return [{"month": f"2025-{m:02d}", "round_trips": trips_per_month,
             "net_bps_per_trip": bps} for m in range(1, 13)]


def test_a_thin_month_fails_the_stability_rule():
    assert screen.fold_stability(_months(5))[0]
    rows = _months(5)
    rows[3]["round_trips"] = 4
    ok, why, thin = screen.fold_stability(rows)
    assert not ok and thin == 1 and "month" in why


def test_a_thin_month_is_not_counted_as_positive():
    rows = _months(5)
    rows[3]["round_trips"] = 1
    rows[3]["net_bps_per_trip"] = 99.0
    assert screen.positive_months(rows) == 11


def test_the_feature_set_carries_the_causal_prior_session_atr():
    """``atr_prev_abs``/``atr_prev_pct`` are declared features, not a typo."""
    from src.backtesting.strategy_search.features import DECLARED_FEATURES

    assert "atr_prev_abs" in DECLARED_FEATURES
    assert "atr_prev_pct" in DECLARED_FEATURES


def test_the_prior_session_atr_is_the_prior_session_only():
    """Day 2's normaliser is day 1's ATR at its last bar; day 1 has none."""
    s1 = [(100.0, 100.5, 99.5, 100.0)] * 30          # TR = 1.0 everywhere
    s2 = [(200.0, 200.5, 199.5, 200.0)] * 30
    df = mk_frame({"2025-01-02": s1, "2025-01-03": s2})
    f = features_mod.build_features(df)
    first2 = np.asarray(df.index.normalize() == pd.Timestamp("2025-01-03"))
    i = int(np.argmax(first2))
    assert np.isnan(f["atr_prev_abs"].iloc[0])
    assert f["atr_prev_abs"].iloc[i] == pytest.approx(1.0)
    assert f["atr_prev_pct"].iloc[i] == pytest.approx(1.0 / 100.0)
    # it is constant across the session: nothing about today's bars revises it
    assert f["atr_prev_abs"].iloc[i:i + 5].nunique() == 1


def _five_symbol_market(n: int = 60):
    bars = flat_session("2025-01-02", FLAT_PX, n)
    frames = {s: mk_frame({"2025-01-02": list(bars)})
              for s in ("SOXL", "TQQQ", "FNGU", "SPXL", "SPY")}
    m = Market(frames)
    return m, FeatureBook(frames).align(m.axis)
