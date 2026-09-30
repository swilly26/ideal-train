"""An artifact filename must identify the run: same ``--tag`` cannot silently clobber.

Round 1 of the turbo edge search passed the **same** ``--tag Vcap2`` for both
the baseline and the zero-cost variant of one lever.  The ``--tag`` value *was*
the whole filename, so the second run overwrote the first run's stats/trades/
folds and half of the experiment disappeared without an error.  These tests pin
the fix: the name always carries the logic, the cost-model variant, every
``--set`` knob and the traded window, in a fixed order.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_turbo_backtest as mod  # noqa: E402

WINDOW = ("2024-09-01", "2025-09-01")


# ── the pure name builder ───────────────────────────────────────────────────
def test_the_tag_alone_no_longer_decides_the_filename():
    """The round-1 clobber: identical ``--tag``, different cost model."""
    base = mod.artifact_tag("classic", "baseline", tag="Vcap2", window=WINDOW)
    zero = mod.artifact_tag("classic", "zero_cost", tag="Vcap2", window=WINDOW)
    assert base != zero, "two variants under one tag still share a filename"
    assert "baseline" in base and "zero_cost" in zero
    assert base.startswith("Vcap2__") and zero.startswith("Vcap2__")


def test_the_name_carries_logic_variant_knobs_and_window():
    tag = mod.artifact_tag("classic", "zero_cost", tag="w2_cap2",
                           overrides=["max_entries_per_session=2",
                                      "max_signal_age_bars=5"],
                           window=WINDOW)
    assert tag.startswith("w2_cap2__classic__zero_cost__")
    assert "max_entries_per_session-2" in tag
    assert "max_signal_age_bars-5" in tag
    assert "2024-09-01" in tag and "2025-09-01" in tag


def test_the_name_is_deterministic_whatever_order_the_knobs_arrive():
    a = mod.artifact_tag("classic", "baseline", overrides=["b=2", "a=1"])
    b = mod.artifact_tag("classic", "baseline", overrides=["a=1", "b=2"])
    assert a == b
    assert a == "classic__baseline__a-1__b-2"


def test_the_window_alone_separates_two_runs_of_the_same_variant():
    old = mod.artifact_tag("classic", "baseline", window=("2025-09-01", "2026-09-01"))
    new = mod.artifact_tag("classic", "baseline", window=WINDOW)
    assert old != new


def test_a_run_with_no_tag_and_no_knobs_keeps_the_historical_name():
    assert mod.artifact_tag("classic", "baseline") == "classic__baseline"


# ── end to end through main(): the files that actually land ─────────────────
def _stub_result(pnl: float) -> "mod.TurboReplayResult":
    from src.backtesting.turbo_engine import TurboConfig, TurboReplayResult
    idx = pd.DatetimeIndex(["2025-01-02 10:00"], name="ts")
    stats = dict(bars=1, sessions=1, round_trips=1, trades_per_session=1.0,
                 pnl_after_costs=pnl, total_return=pnl / 100_000.0,
                 initial_equity=100_000.0, pnl_gross=pnl, profit_factor=1.0,
                 win_rate=0.5, break_even_win_rate=0.5, max_drawdown=0.0,
                 session_sharpe=0.0, expectancy=pnl, folds_positive=1,
                 folds_total=1, exits={}, skipped={}, fees_paid=0.0,
                 cost_drag_same_fills=0.0)
    return TurboReplayResult(
        trades=pd.DataFrame({"symbol": ["SOXL"], "pnl_after_costs": [pnl]}),
        equity_curve=pd.Series([100_000.0 + pnl], index=idx, name="equity"),
        stats=stats, config=TurboConfig.classic(),
        per_symbol=pd.DataFrame({"symbol": ["SOXL"], "pnl": [pnl]}),
        folds_monthly=pd.DataFrame({"month": ["2025-01"], "pnl": [pnl]}))


@pytest.fixture
def stubbed_engine(monkeypatch):
    """Replace the loader + engine so only the artifact writing is exercised."""
    def _frames(symbols, start=None, end=None):
        idx = pd.DatetimeIndex(["2025-01-02 10:00"], name="ts")
        return {s: pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0],
                                 "close": [1.0], "volume": [1.0]}, index=idx)
                for s in symbols}

    def _replay(frames, cfg, trade_window=None):
        return _stub_result(0.0 if cfg.commission_per_share == 0 else -100.0)

    monkeypatch.setattr(mod, "load_frames", _frames)
    monkeypatch.setattr(mod, "replay", _replay)


def _run(out: Path, variant: str, tag: str, *sets: str) -> int:
    argv = ["--logic", "classic", "--variant", variant, "--symbols", "SOXL",
            "--start", "2024-09-01", "--end", "2025-09-01",
            "--out-dir", str(out), "--tag", tag]
    for pair in sets:
        argv += ["--set", pair]
    return mod.main(argv)


def test_both_variants_survive_when_they_share_a_tag(stubbed_engine, tmp_path):
    out = tmp_path / "out"
    assert _run(out, "baseline", "Vcap2") == 0
    assert _run(out, "zero_cost", "Vcap2") == 0
    names = sorted(p.name for p in out.glob("turbo_stats_*.json"))
    assert len(names) == 2, f"a run was clobbered: {names}"
    variants = {json.loads((out / n).read_text())["variant"] for n in names}
    assert variants == {"baseline", "zero_cost"}
    # one artifact per family per run -- nothing overwritten, nothing missing
    for family in ("turbo_trades_*", "turbo_equity_*", "turbo_per_symbol_*",
                   "turbo_folds_monthly_*", "turbo_summary_*"):
        assert len(list(out.glob(family))) == 2, family
    # the old (buggy) name is never written
    assert not list(out.glob("turbo_stats_Vcap2.json"))


def test_two_knob_variants_of_one_cost_model_survive_a_shared_tag(
        stubbed_engine, tmp_path):
    out = tmp_path / "out"
    assert _run(out, "baseline", "cap", "max_entries_per_session=2") == 0
    assert _run(out, "baseline", "cap", "max_entries_per_session=5") == 0
    assert len(list(out.glob("turbo_stats_*.json"))) == 2


def test_the_run_record_says_which_window_and_knobs_it_was(
        stubbed_engine, tmp_path):
    out = tmp_path / "out"
    assert _run(out, "baseline", "cap", "max_entries_per_session=2") == 0
    (path,) = list(out.glob("turbo_stats_*.json"))
    st = json.loads(path.read_text())
    assert st["variant"] == "baseline" and st["logic"] == "classic"
    assert st["knob_overrides"] == ["max_entries_per_session=2"]
    assert st["window"] == ["2024-09-01", "2025-09-01"]
