"""The round-2 gate audit's four must-fix defects (§A1–A4) + record integrity (§A5).

Source: ``/home/team/shared/ROUND2_GATE_AUDIT_FINDINGS.md`` §A, made governing
by ``STRATEGY_SEARCH_ROUND2_BRIEF.md`` §R3-10: *no screen runs on this engine
until these are fixed and proven.*  Every test in this file fails on the
pre-fix commit ``90f0c53`` for the reason named in its own docstring; the
recorded run of this file against a scratch worktree at that commit is the
committed artefact ``docs/ROUND2_FOURFIXES_PREFIX_PYTEST.txt`` (16 failed),
with its header in ``docs/ROUND2_FOURFIXES_PREFIX_EVIDENCE.md``.

* **A1** — exit slippage was priced off the exit bar's *open* while four of the
  five exit paths fill at that bar's *close* (the EOD flatten, the time exit) or
  at a stop/target level, so ``cost_drag`` booked a toll the fills never charged
  and the engine's own zero-cost reconstruction disagreed with a real zero-cost
  replay on any non-flat tape priced above the 1¢ slippage floor.
* **A2** — the zero-cost column was ``pnl_gross + cost_drag``, i.e. it added the
  commissions a **second** time: zero error at baseline (fees are 0) and
  +fees of fabricated profit at every fee-charging cost level.
* **A3** — the runner asserted the recorded run set was W1/baseline/both-sizings
  only, so a family with a **W1 survivor** raised *after* its replays and wrote
  no verdict, while a family killed on W1 passed.
* **A4** — ``dataclasses.replace(cfg, sizing=sizing)`` happened *after* the hash,
  so one cell's two sizing modes shared a hash and the ``equity_fraction``
  record claimed ``params.sizing == "fixed_notional"``; and the W1/W2 identity
  guard was keyed with the window, so it could only ever compare a cell to
  itself.
* **A5** — ``engine_sha()`` returned the string ``"unknown"`` on any failure,
  run records were replaced silently by an unconditional ``write_text``, and
  the filename carried neither hash.
"""

from __future__ import annotations

import dataclasses
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtesting.replay_costs import CostModel  # noqa: E402
from src.backtesting.strategy_search import engine, families  # noqa: E402
from src.backtesting.strategy_search.engine import (  # noqa: E402
    Instrument,
    Leg,
    Market,
    run_search,
)
from src.backtesting.strategy_search.features import FeatureBook  # noqa: E402
from tests.test_strategy_search_engine import (  # noqa: E402
    OPEN_MIN,
    flat_session,
    mk_frame,
    simple_cfg,
)
from tests.test_strategy_search_round2_gate import (  # noqa: E402
    LEG_SETS,
    flat_trip,
)

PX = 100.0                                # above $50: 2 bps × price > the 1¢ floor
SESSION_BARS = 390                        # 09:30 .. 15:59
EOD_INDEX = 15 * 60 + 30 - OPEN_MIN       # the 15:30 bar — where the flatten fills
TIME_EXIT_INDEX = 11                      # entry fills on bar 6; +5 minutes is bar 11


# ── helpers ────────────────────────────────────────────────────────────


def tape(index: int, o: float, c: float, px: float = PX, n: int = SESSION_BARS):
    """A flat session except bar *index*, whose open and close **differ**.

    That is the whole point of the fixture: above $50 the slippage
    ``max(2bps·P, 1¢)`` is different at the two prices, so charging it on the
    price the fill was *not* made at is visible.  A flat tape (open == close)
    cannot tell the two apart, which is why the pre-fix reconciliation test
    could never fail.
    """
    bars = flat_session("2025-01-02", px, n)
    bars[index] = (float(o), max(o, c) + 0.02, min(o, c) - 0.02, float(c))
    return bars


def one_leg_market(bars, symbol: str = "SPY"):
    """One long signal on bar 5 (it fills on bar 6's open) and one symbol."""
    m = Market({symbol: mk_frame({"2025-01-02": bars})})
    entry = np.zeros(m.n(), dtype=np.int8)
    entry[5] = 1
    inst = Instrument(key=symbol, legs=(Leg(symbol, 1, 1.0),), entry_dir=entry,
                      exit_now=np.zeros(m.n(), dtype=np.int8),
                      valid=m.valid((Leg(symbol, 1, 1.0),)))
    return m, {symbol: inst}


def leg_rows(tr) -> list[dict]:
    """The per-leg qty and fills the trade record carries in ``leg_detail``."""
    out = []
    for chunk in str(tr["leg_detail"]).split(";"):
        i, sym, side, qty, ef, xf = chunk.split("|")
        out.append(dict(index=int(i), symbol=sym, side=int(side), qty=float(qty),
                        entry_fill=float(ef), exit_fill=float(xf)))
    return out


def expected_fees(costs: CostModel, legs: list[dict]) -> float:
    return sum(costs.fees(l["qty"], l["entry_fill"])
               + costs.fees(l["qty"], l["exit_fill"]) for l in legs)


def bars(n: int = SESSION_BARS, px: float = PX) -> pd.DataFrame:
    return mk_frame({"2025-01-02": flat_session("2025-01-02", px, n)})


def runner_mod():
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import run_strategy_search as mod
    return mod


def spec_copy(family: str = "A", index: int = 3, symbols=("SOXL",)) -> dict:
    """A family-A cell with a one-symbol universe (no cache, no SPY needed)."""
    spec = dict(families.GRIDS[family][index])
    spec["params"] = dict(spec["params"])
    spec["signal"] = dict(spec["signal"])
    spec["extras"] = dict(spec.get("extras", {}))
    spec["symbols"] = list(symbols)
    return spec


def canned(frames, dates=("2025-09-01", "2026-09-01")) -> dict:
    """A ``Runner._window`` payload: bars + features, with no cache involved."""
    m = Market(frames)
    return {"frames": frames, "market": m,
            "aligned": FeatureBook(frames).align(m.axis),
            "cov": {}, "dates": list(dates)}


def canned_runner(out, windows):
    """A real ``Runner`` whose window data is injected."""
    mod = runner_mod()
    runner = mod.Runner(cache_dir=Path("/nonexistent-cache"), out_dir=Path(out),
                        verbose=False)
    runner._window = lambda window: windows[window]
    return runner


# ── A1: the exit slip is charged on the price the fill was made at ─────


def test_an_eod_flatten_prices_exit_slippage_at_the_close_it_filled_at():
    """The flatten fills at the 15:30 **close**; the slip must be charged there."""
    costs = CostModel.baseline()
    exit_open, exit_close = 110.0, 120.0
    m, insts = one_leg_market(tape(EOD_INDEX, exit_open, exit_close))
    tr = run_search(m, insts, simple_cfg(), costs).trades.iloc[0]
    assert tr["exit_reason"] == "eod"
    leg = leg_rows(tr)[0]
    charged_at_the_close = leg["qty"] * (costs.slip_per_share(PX)
                                         + costs.slip_per_share(exit_close))
    priced_off_the_open = leg["qty"] * (costs.slip_per_share(PX)
                                        + costs.slip_per_share(exit_open))
    assert abs(charged_at_the_close - priced_off_the_open) > 0.5, \
        "the fixture must be able to tell the two prices apart"
    assert tr["cost_drag"] == pytest.approx(charged_at_the_close, rel=1e-6), (
        "exit slippage must be charged on the price the exit fill was made at "
        f"(the {exit_close} close), not on the exit bar's {exit_open} open")
    # the fill itself was always right; it is the accounting that was wrong
    assert tr["exit_price"] == pytest.approx(
        exit_close - costs.slip_per_share(exit_close), rel=1e-6)


def test_a_time_exit_prices_exit_slippage_at_the_close_it_filled_at():
    """The time exit fills at the close of the bar that completes the hold."""
    costs = CostModel.baseline()
    exit_open, exit_close = 108.0, 118.0
    m, insts = one_leg_market(tape(TIME_EXIT_INDEX, exit_open, exit_close))
    tr = run_search(m, insts, simple_cfg(time_exit_minutes=5), costs).trades.iloc[0]
    assert tr["exit_reason"] == "time"
    assert tr["exit_time"] == m.axis[TIME_EXIT_INDEX]
    leg = leg_rows(tr)[0]
    assert tr["cost_drag"] == pytest.approx(
        leg["qty"] * (costs.slip_per_share(PX) + costs.slip_per_share(exit_close)),
        rel=1e-6), (
        f"the time exit fills at the {exit_close} close, not the {exit_open} open")


def test_the_zero_cost_reconstruction_matches_a_real_replay_on_a_non_flat_tape():
    """The runner **raises** on this residual: the guard fires on real data.

    ``reconcile_zero_cost`` compares the engine's own reconstruction against a
    real replay with every cost switched off.  On any non-flat tape priced above
    the 1¢ floor the pre-fix engine's exit slip was charged on a different price
    from its fill, and the residual was the whole difference — ~1 order of
    magnitude per trip, seven orders above the 1e-6 tolerance.
    """
    m, insts = one_leg_market(tape(EOD_INDEX, 110.0, 120.0))
    cfg = simple_cfg()
    base = run_search(m, insts, cfg, CostModel.baseline())
    zero = run_search(m, insts, cfg, CostModel.zero_cost())
    rec = engine.reconcile_zero_cost(base, zero)
    assert rec["same_trips"] and rec["same_fill_timestamps"]
    assert rec["within_tolerance"], rec["mismatches"]
    assert "pnl_gross + slip_drag" in rec["reconstruction_form"]


# ── A2: the zero-cost column must not add the commissions ──────────────


#: Three fee-charging cost levels: the pessimistic model ($0.005/share) and the
#: same $0.005/share scaled ×1.5 and ×2.  At baseline fees are 0, which is the
#: only reason the old column looked right.
FEE_LEVELS = {
    "pessimistic": CostModel.pessimistic(),
    "pessimistic_x1_5": dataclasses.replace(CostModel.pessimistic(),
                                            commission_per_share=0.0075,
                                            label="pessimistic_x1_5"),
    "pessimistic_x2": dataclasses.replace(CostModel.pessimistic(),
                                          commission_per_share=0.010,
                                          label="pessimistic_x2"),
}


@pytest.mark.parametrize("label", sorted(FEE_LEVELS))
def test_a_fee_charging_flat_pair_books_no_fabricated_zero_cost_profit(label):
    """A flat pair has **zero** gross edge at any cost level; with fees on, the
    pre-fix zero-cost column claimed ``+fees`` of profit out of nothing."""
    costs = FEE_LEVELS[label]
    res = flat_trip(LEG_SETS["one_symbol_unequal"], key="SOXL/SOXL", costs=costs)
    tr = res.trades.iloc[0]
    legs = leg_rows(tr)
    fees = expected_fees(costs, legs)
    assert fees > 0.0, "this variant must actually charge commissions"
    assert res.stats["pnl_zero_cost_same_fills"] == pytest.approx(0.0, abs=1e-6), (
        "a flat tape's zero-cost P&L is exactly zero; the pre-fix column added "
        f"the ${fees:,.2f} of commissions a second time")
    assert res.stats["fees_paid"] == pytest.approx(fees, rel=1e-6)


@pytest.mark.parametrize("label", sorted(FEE_LEVELS))
def test_the_cost_columns_split_slippage_from_fees(label):
    """``slip_drag``, ``fees`` and ``cost_drag`` are three different numbers."""
    costs = FEE_LEVELS[label]
    tr = flat_trip(LEG_SETS["one_symbol_unequal"], key="SOXL/SOXL",
                   costs=costs).trades.iloc[0]
    assert {"slip_drag", "fees", "cost_drag", "pnl_gross_precost"} <= set(tr.index), \
        "the trade record must carry the split columns, not one merged drag"
    legs = leg_rows(tr)
    total_qty = sum(l["qty"] for l in legs)
    assert tr["slip_drag"] == pytest.approx(
        2.0 * costs.slip_per_share(PX) * total_qty, rel=1e-6), \
        "slippage only: both legs, entry and exit, on a flat tape"
    assert tr["fees"] == pytest.approx(expected_fees(costs, legs), rel=1e-6)
    assert tr["cost_drag"] == pytest.approx(tr["slip_drag"] + tr["fees"], rel=1e-6)
    # the three published identities, all of which must hold together
    assert tr["pnl_after_costs"] == pytest.approx(tr["pnl_gross"] - tr["fees"],
                                                  rel=1e-6)
    assert tr["pnl_gross_precost"] == pytest.approx(tr["pnl_gross"] + tr["slip_drag"],
                                                    rel=1e-6)
    assert tr["pnl_after_costs"] == pytest.approx(
        tr["pnl_gross_precost"] - tr["slip_drag"] - tr["fees"], rel=1e-6)
    # the explicit flat-pair invariant, split into its two halves
    assert tr["pnl_gross"] + tr["slip_drag"] == pytest.approx(0.0, abs=1e-6)
    assert tr["pnl_gross"] + tr["cost_drag"] - tr["fees"] == pytest.approx(0.0,
                                                                           abs=1e-6)


# ── A3: a W1 survivor must reach a written verdict ─────────────────────


SURVIVOR = ("a_trend_trail50_noreg", "fixed_notional")


def _stub_run(mod):
    """A ``Runner.run`` replacement that fabricates records: one W1 survivor.

    The point of the test is the *screen's* control flow and the record-set
    assertion, not the replay: every declared cell × sizing gets a W1 record,
    exactly one of them clears the gate, and the stub registers each record
    under the real ``run_key`` so the runner's own bookkeeping is exercised.
    """

    def stub(self, family, spec, window, cost_label, sizing):
        cell = spec["name"]
        survivor = (cell, sizing) == SURVIVOR
        trips = 200 if survivor else 10
        bps = 5.0 if survivor else -3.0
        rec = {
            "family": family, "cell_id": cell, "config": cell,
            "window": window, "sizing": sizing, "cost_label": cost_label,
            "trips": trips, "net_bps_per_trip": bps,
            "net_pnl": 1_000.0 if survivor else -50.0,
            "net_pct": 0.01 if survivor else -0.001,
            "net_bps_t_stat": 2.0, "max_drawdown": -0.05,
            "cost_drag_per_trip_bps": 4.0,
            "folds_positive": 9, "folds_total": 12,
            "folds": [{"month": f"2025-{m:02d}", "round_trips": 20,
                       "net_bps_per_trip": bps} for m in range(1, 13)],
            "resolved_hash": "0" * 16, "engine_sha": self.sha,
        }
        self.recorded[mod.run_key(family, cell, window, cost_label, sizing)] = rec
        return rec

    return stub


def test_a_family_with_a_w1_survivor_reaches_a_written_verdict(monkeypatch, tmp_path):
    """A survivor licenses W2 **and** both zero-cost runs — and must not abort.

    On the pre-fix runner this raised ``RuntimeError: family A screen: recorded
    run set does not equal the declared set (missing [], unexpected ['A|…|W2|
    baseline|fixed_notional', …])`` *after* the replays and *before*
    ``screen_A.json`` was written — invisible for the whole of round 1, because
    every family there was killed on W1 and never took that branch.
    """
    mod = runner_mod()
    monkeypatch.setattr(mod.Runner, "run", _stub_run(mod))
    out = tmp_path / "screen-out"
    rc = mod.main(["--screen", "--family", "A", "--out-dir", str(out), "--quiet"])
    assert rc == 0
    verdict_path = out / "screen_A.json"
    assert verdict_path.exists(), "a W1 survivor must produce a written verdict"
    verdict = json.loads(verdict_path.read_text())
    assert verdict["w1_survivors"] == [SURVIVOR[0]]
    assert [SURVIVOR[1]] == [r["sizing"] for r in verdict["ranked"]]
    keys = set(verdict["runs"])
    assert f"W2|baseline|{SURVIVOR[1]}|{SURVIVOR[0]}" in keys, \
        "the survivor's W2 confirmation run is part of the declared set"
    for window in ("W1", "W2"):
        assert f"{window}|zero_cost|{SURVIVOR[1]}|{SURVIVOR[0]}" in keys


def test_the_declared_run_set_equals_what_the_screen_records():
    """The declared set is the base matrix **plus** the licensed branch keys."""
    mod = runner_mod()
    specs = families.GRIDS["A"]
    first = specs[0]["name"]
    base = {mod.run_key("A", s["name"], "W1", "baseline", sz)
            for s in specs for sz in ("fixed_notional", "equity_fraction")}
    assert mod.declared_screen_run_set("A", specs, [], []) == base
    licensed = mod.declared_screen_run_set(
        "A", specs, [[first, "fixed_notional"]], [[first, "fixed_notional"]])
    assert licensed - base == {
        mod.run_key("A", first, "W2", "baseline", "fixed_notional"),
        mod.run_key("A", first, "W1", "zero_cost", "fixed_notional"),
        mod.run_key("A", first, "W2", "zero_cost", "fixed_notional"),
    }


# ── A4: the hash covers the run's own sizing, and W1/W2 must match ─────


def test_the_two_sizing_modes_of_one_cell_hash_differently(tmp_path):
    """The hash (and the generated name) must describe the config **that ran**."""
    windows = {"W1": canned({"SOXL": bars()})}
    runner = canned_runner(tmp_path, windows)
    spec = spec_copy()
    fixed = runner.run("A", spec, "W1", "baseline", "fixed_notional")
    equity = runner.run("A", spec, "W1", "baseline", "equity_fraction")
    assert fixed["resolved_hash"] != equity["resolved_hash"], \
        "one cell's two sizing modes are two different configs"
    assert fixed["config_name_resolved"] != equity["config_name_resolved"]
    assert equity["resolved_params"]["params"]["sizing"] == "equity_fraction", \
        "the equity_fraction record must not claim it ran fixed_notional"
    assert equity["sizing"] == "equity_fraction"
    assert fixed["declaration"]["params"]["sizing"] == "fixed_notional", \
        "the declaration is what the grid declared; the applied sizing is separate"


def test_a_w1_w2_config_mismatch_is_refused_loudly(tmp_path):
    """A cell whose resolved config differs between the two windows is refused.

    Simulated the only way it can arise: the cell resolves differently on W2
    than on W1 (a data-dependent default — the shape of the stage-2 ``z_window``
    defect).  The pre-fix guard was keyed *with* the window, so it compared the
    W2 run against an empty slot and accepted the mismatch silently.
    """
    windows = {"W1": canned({"SOXL": bars()}),
               "W2": canned({"SOXL": bars(px=101.0)},
                            dates=("2024-09-01", "2025-09-01"))}
    runner = canned_runner(tmp_path, windows)
    spec = spec_copy()
    runner.run("A", spec, "W1", "baseline", "fixed_notional")
    mutated = spec_copy()
    mutated["params"]["stop_pct"] = 0.02
    with pytest.raises(RuntimeError, match=r"differs\s+between runs") as excinfo:
        runner.run("A", mutated, "W2", "baseline", "fixed_notional")
    message = str(excinfo.value)
    assert "W1/baseline" in message and "W2/baseline" in message
    assert runner.recorded.get(
        runner_mod().run_key("A", spec["name"], "W2", "baseline",
                             "fixed_notional")) is None, \
        "a refused run must not be recorded"


def test_the_same_cell_resolves_alike_on_window_1_and_window_2(tmp_path):
    """The replacement for the tautological test (which built the same frames
    twice and compared a deterministic function with itself): this compares two
    **different** windows' bar sets.  It passes on both trees — it pins the
    property the guard enforces, not the defect it catches."""
    windows = {"W1": canned({"SOXL": bars()}, ("2025-09-01", "2026-09-01")),
               "W2": canned({"SOXL": bars(n=361, px=250.0)},
                            ("2024-09-01", "2025-09-01"))}
    runner = canned_runner(tmp_path, windows)
    spec = spec_copy()
    w1 = runner.run("A", spec, "W1", "baseline", "fixed_notional")
    w2 = runner.run("A", spec, "W2", "baseline", "fixed_notional")
    assert w1["window"] != w2["window"]
    assert w1["resolved_hash"] == w2["resolved_hash"]
    assert w1["config_name_resolved"] == w2["config_name_resolved"]
    assert w1["identity_key"] == w2["identity_key"]


# ── A5: run-record integrity ───────────────────────────────────────────


def test_the_engine_sha_is_a_real_commit_or_a_loud_failure(monkeypatch):
    """``"unknown"`` is not provenance; the record must refuse to be written."""
    mod = runner_mod()
    assert re.fullmatch(r"[0-9a-f]{40}", mod.engine_sha()), \
        "a real 40-hex commit, not a placeholder string"

    def boom(*_args, **_kw):
        raise OSError("git is not installed")

    monkeypatch.setattr(mod.subprocess, "run", boom)
    with pytest.raises(RuntimeError, match="provenance"):
        mod.engine_sha()

    class _Failed:
        returncode = 1
        stdout = ""
        stderr = "fatal: not a git repository"

    monkeypatch.setattr(mod.subprocess, "run", lambda *_a, **_k: _Failed())
    with pytest.raises(RuntimeError, match="refusing to write a run record"):
        mod.engine_sha()


def test_a_run_record_carries_both_hashes_and_is_never_silently_replaced(tmp_path):
    """The writer must refuse to replace evidence, and the name must carry the
    config hash and the engine SHA."""
    windows = {"W1": canned({"SOXL": bars()})}
    runner = canned_runner(tmp_path, windows)
    spec = spec_copy()
    rec = runner.run("A", spec, "W1", "baseline", "fixed_notional")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        runner.run("A", spec_copy(), "W1", "baseline", "fixed_notional")
    files = sorted(p.name for p in Path(tmp_path).glob("*.json"))
    assert len(files) == 1, files
    assert not list(Path(tmp_path).glob("*.partial"))
    assert rec["resolved_hash"][:8] in files[0], files[0]
    assert rec["engine_sha"][:8] in files[0], files[0]
    assert rec["sizing"] in files[0] and rec["window"] in files[0]
