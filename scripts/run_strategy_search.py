#!/usr/bin/env python3
"""Stage-1 strategy search: run the declared grids through the screen.

    # one config, one window (a single replay)
    .venv/bin/python scripts/run_strategy_search.py --family A \
        --config a_trend_trail50_reg --window W1 --sizing fixed_notional

    # the full pre-registered screen for one or more families (order A, C, D, B)
    .venv/bin/python scripts/run_strategy_search.py --screen --family A

Every run writes one machine-readable record to ``--out-dir``; the screen writes
a family verdict JSON next to them.  Nothing is adopted here: stage 1 only
decides which configs deserve stage 2.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtesting.replay_costs import CostModel  # noqa: E402
from src.backtesting.strategy_search import engine, families, screen  # noqa: E402
from src.backtesting.strategy_search.engine import (  # noqa: E402
    Market,
    per_symbol,
    monthly_folds,
    run_search,
)
from src.backtesting.strategy_search.features import (  # noqa: E402
    DEFAULT_CACHE,
    FeatureBook,
    coverage,
    load_window,
)

WINDOWS = {
    "W1": ("2025-09-01", "2026-09-01"),   # development
    "W2": ("2024-09-01", "2025-09-01"),   # confirmation
}
DEFAULT_OUT = Path("/home/team/shared/strategy_search/stage1")
FAMILY_SYMBOLS = ("SOXL", "TQQQ", "FNGU", "SPXL", "SPY")
FAMILY_ORDER = ("A", "C", "D", "B")
#: The one decision sizing mode (P2); the other is a reported robustness column
#: and does not count as a hypothesis.
DECISION_SIZING = "fixed_notional"
#: Every cost level a screened cell may run at (R3-5/R3-7).  ``baseline`` is the
#: realistic model the gates read; ``zero_cost`` is the gross-edge diagnostic.
SCREEN_COSTS = ("baseline", "zero_cost")


def engine_sha() -> str:
    """The commit the harness actually ran from (recorded in every artefact).

    A real 40-hex SHA, or a **loud failure** (A5).  This used to swallow every
    exception and return the string ``"unknown"``, which is worse than no
    provenance at all: a run record whose ``engine_sha`` is ``"unknown"`` still
    parses, still looks complete, and cannot be traced to any tree — and nothing
    anywhere asserted otherwise.
    """
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=10)
    except Exception as exc:                                     # noqa: BLE001
        raise RuntimeError(
            f"engine_sha: `git rev-parse HEAD` failed in {ROOT} ({exc!r}); "
            f"refusing to write a run record whose provenance is unknown") from exc
    sha = (out.stdout or "").strip()
    if out.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RuntimeError(
            f"engine_sha: `git rev-parse HEAD` in {ROOT} returned {sha!r} "
            f"(rc={out.returncode}, stderr={(out.stderr or '').strip()!r}); "
            f"refusing to write a run record whose provenance is unknown")
    return sha


def run_key(family: str, cell: str, window: str, cost: str, sizing: str) -> str:
    """The unique identity of one run record."""
    return f"{family}|{cell}|{window}|{cost}|{sizing}"


def identity_key(family: str, cell: str, sizing: str) -> str:
    """The **window-independent** identity of one cell (A4).

    ``run_key`` carries the window, so a guard built on it can only ever compare
    a cell against itself and can never notice that W2 ran a *different*
    resolved config from W1 — the E2 failure class, reintroduced.  This key
    deliberately leaves the window (and the cost level) out: what must be
    identical across windows is the config, not the record.
    """
    return f"{family}|{cell}|{sizing}"


def record_filename(family: str, cell: str, window: str, cost: str, sizing: str,
                    resolved_hash: str, sha: str) -> str:
    """The run record's filename: identity **and** both hashes (A5).

    The filename carries the 8-hex config hash and the 8-hex engine SHA, so a
    record on disk can be matched to a tree and a config without opening it, and
    two different resolved configs can never land on one name.
    """
    return (f"{family}__{cell}__{window}__{cost}__{sizing}"
            f"__{resolved_hash[:8]}-{sha[:8]}.json")


def write_record(path: Path | str, record: dict, overwrite: bool = False) -> Path:
    """Write one run record, refusing to replace one that already exists (A5).

    The writer used to be an unconditional ``write_text``: a re-run silently
    replaced the evidence of the first run, with nothing in the artefact or the
    directory to say it had happened.  The write is atomic (temp file + rename)
    so a killed run cannot leave a half-written record either.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise RuntimeError(
            f"refusing to overwrite the existing run record {path}: an existing "
            f"record is evidence — delete it deliberately (or pass "
            f"--overwrite) if it is known stale")
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(record, indent=2, default=str))
    tmp.replace(path)
    return path


def assert_exact_run_set(recorded: set[str], expected: set[str], where: str) -> None:
    """The recorded set must equal the declared set exactly (E2).

    Called before any verdict is written: a cell that silently failed to build
    or run must not look like a cell that was screened and died.
    """
    missing = sorted(expected - recorded)
    extra = sorted(recorded - expected)
    if missing or extra:
        raise RuntimeError(
            f"{where}: recorded run set does not equal the declared set "
            f"(missing {missing}, unexpected {extra})")


def declared_screen_run_set(family: str, specs: Sequence[dict],
                            w1_pass_pairs: Sequence[Sequence[str]] = (),
                            w2_confirmed_pairs: Sequence[Sequence[str]] = ()
                            ) -> set[str]:
    """Every run key the pre-registered screen declares **in one branch** (A3).

    Unconditional: W1/baseline for every declared cell × both sizings — a cell
    that silently failed to build or run must never look like a cell that was
    screened and died.  Conditional, and read off the screen's own gate
    outcomes: W2/baseline for each (cell, sizing) that cleared W1, and both
    windows' zero-cost runs for each that then confirmed on W2.  A family whose
    best cell dies on W1 licenses neither, which is why the old W1-only
    assertion passed for all of round 1.
    """
    expected = {run_key(family, s["name"], "W1", "baseline", sizing)
                for s in specs for sizing in screen.SIZING_MODES}
    for cell, sizing in (tuple(p) for p in w1_pass_pairs):
        expected.add(run_key(family, cell, "W2", "baseline", sizing))
    for cell, sizing in (tuple(p) for p in w2_confirmed_pairs):
        for window in WINDOWS:
            for cost in SCREEN_COSTS:
                expected.add(run_key(family, cell, window, cost, sizing))
    return expected


def declared_round2_run_set(family: str, specs: Sequence[dict],
                            sizing: str = screen.ROUND2_SIZING,
                            cost: str = screen.ROUND2_BASE_COST) -> set[str]:
    """R3-3(iii): **every** declared cell, on **both** windows (audit §B4).

    The stage-1 set (:func:`declared_screen_run_set`) is conditional past W1 — a
    cell that loses W1 licenses no W2 record — which is the round-1 kill rule.
    Round 2 is unconditional: two run keys per declared cell, so a 36-cell grid
    declares 36 x 2 = **72** base records and a run that produced 71 is a hard
    failure rather than a slightly smaller grid whose report still says N = 36.
    """
    return {run_key(family, s["name"], window, cost, sizing)
            for s in specs for window in WINDOWS}


class Runner:
    """Loads a window's bars + features once, then replays configs against it."""

    def __init__(self, cache_dir: Path | str, out_dir: Path, verbose: bool = True,
                 reconcile: bool = True, overwrite: bool = False) -> None:
        self.cache_dir = Path(cache_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.reconcile = reconcile
        self.overwrite = overwrite
        self.sha = engine_sha()
        self.recorded: dict[str, dict] = {}
        #: ``identity_key`` → {resolved_hash, window, cost, sizing}: the guard the
        #: record's own key cannot express (A4).
        self.identity: dict[str, dict] = {}
        self._cache: dict[str, dict] = {}

    def _window(self, window: str) -> dict:
        if window not in self._cache:
            start, end = WINDOWS[window]
            frames = load_window(FAMILY_SYMBOLS, start, end, self.cache_dir)
            missing = [s for s in FAMILY_SYMBOLS if s not in frames]
            if missing:
                raise SystemExit(f"no cached bars for {missing} under {self.cache_dir}")
            market = Market(frames)
            book = FeatureBook(frames)
            aligned = book.align(market.axis)
            self._cache[window] = {
                "frames": frames, "market": market, "aligned": aligned,
                "cov": coverage(frames), "dates": [start, end],
            }
            if self.verbose:
                print(f"[{window}] axis={market.n():,} bars "
                      f"{market.axis[0]} .. {market.axis[-1]} "
                      f"symbols={sorted(frames)}", flush=True)
        return self._cache[window]

    # ── the run's identity ─────────────────────────────────────────────
    @staticmethod
    def _sized_spec(spec: dict, sizing: str) -> dict:
        """The declared cell with this run's **sizing** applied, before hashing.

        ``symbols`` is filled in from the family universe here, exactly as the
        runner used to do on the raw spec, so a grid cell that declares no
        universe still resolves to the same config it always did.
        """
        out = dict(spec)
        out.setdefault("symbols", list(FAMILY_SYMBOLS[:-1]))
        out["params"] = {**dict(spec.get("params", {})), "sizing": sizing}
        return out

    def _check_identity(self, family: str, cell: str, sizing: str, window: str,
                        cost_label: str, resolved_hash: str) -> None:
        """W1 and W2 must execute the **same resolved config** (A4).

        Keyed by :func:`identity_key`, which carries no window: the guard this
        replaces read ``self.recorded[run_key(...)]``, and because that key
        contains the window it could only ever compare a cell to itself — an
        E2-class defect (a knob the engine ran that the artefact did not record)
        could pass through W1 unnoticed and W2 would not be checked against it
        at all.
        """
        ik = identity_key(family, cell, sizing)
        prev = self.identity.get(ik)
        if prev is not None and prev["resolved_hash"] != resolved_hash:
            raise RuntimeError(
                f"{family} {cell} sizing={sizing}: the resolved config differs "
                f"between runs — {prev['window']}/{prev['cost']} hashed "
                f"{prev['resolved_hash']} while {window}/{cost_label} hashes "
                f"{resolved_hash}. W1 and W2 (and every cost level) must execute "
                f"the same resolved config, so this run is refused rather than "
                f"recorded")
        self.identity[ik] = {"resolved_hash": resolved_hash, "window": window,
                             "cost": cost_label, "sizing": sizing}

    def run(self, family: str, spec: dict, window: str, cost_label: str,
            sizing: str) -> dict:
        w = self._window(window)
        market: Market = w["market"]
        declared = dict(spec)
        # The run's sizing is applied to the *declaration* **before** the config
        # is built and hashed (A4).  Replacing it on the built config afterwards
        # (``dataclasses.replace(cfg, sizing=sizing)``) left the resolved hash and
        # the generated name describing ``fixed_notional`` while the record's
        # ``sizing`` said ``equity_fraction`` — the two sizing modes of one cell
        # shared a hash and a name, and the artefact contradicted itself.
        spec = self._sized_spec(declared, sizing)
        cfg, instruments, resolved = families.build(family, spec, market,
                                                    w["aligned"])
        if resolved["params"].get("sizing") != sizing:
            raise RuntimeError(
                f"{family} {spec['name']}: the resolved config declares "
                f"sizing={resolved['params'].get('sizing')!r} but this run is "
                f"{sizing!r} — the recorded hash would not describe the run")
        costs = CostModel.baseline() if cost_label == "baseline" else CostModel.zero_cost()
        costs = dataclasses.replace(costs, label=cost_label)
        t0 = time.time()
        res = run_search(market, instruments, cfg, costs)
        elapsed = time.time() - t0
        stats = res.stats
        folds = monthly_folds(res.trades)
        persym = per_symbol(res.trades)
        fold_rows = json.loads(folds.to_json(orient="records"))
        fold_ok, fold_why, months_zero = screen.fold_stability(fold_rows)
        stats["folds_positive"] = screen.positive_months(fold_rows)
        stats["folds_total"] = int(len(folds))
        folds = folds[[c for c in folds.columns]]
        #: The hash the record carries is the *config's* identity — the same
        #: value ``cfg.name`` is generated from, over the resolved params
        #: (sizing included), the resolved signal dict and the extras.
        resolved_hash = families.resolved_hash(family, resolved)
        if resolved_hash not in cfg.name:
            raise RuntimeError(
                f"{family} {spec['name']}: the recorded hash {resolved_hash} is "
                f"not the one {cfg.name!r} was generated from")
        key = run_key(family, spec["name"], window, cost_label, sizing)
        self._check_identity(family, spec["name"], sizing, window, cost_label,
                             resolved_hash)
        reconciliation = None
        if cost_label == "baseline" and self.reconcile:
            zero = run_search(market, instruments, cfg, CostModel.zero_cost())
            reconciliation = engine.reconcile_zero_cost(res, zero)
            if not (reconciliation["same_trips"]
                    and reconciliation["same_fill_timestamps"]
                    and reconciliation["within_tolerance"]):
                raise RuntimeError(
                    f"{key}: the zero-cost reconstruction does not match a real "
                    f"zero-cost replay of the same config — the zero-cost column "
                    f"the fold rule runs on is not a P&L: "
                    f"{reconciliation['mismatches']}")
        record = {
            "family": family,
            "cell_id": spec["name"],
            "config": spec["name"],
            "config_name_resolved": cfg.name,
            "window": window,
            "window_dates": w["dates"],
            "cost_model": costs.as_dict(),
            "cost_label": cost_label,
            "sizing": sizing,
            "sizing_mode": sizing,
            "engine_sha": self.sha,
            "resolved_params": resolved,
            "resolved_hash": resolved_hash,
            "identity_key": identity_key(family, spec["name"], sizing),
            "declaration": {"params": dict(declared.get("params", {})),
                            "signal": dict(declared.get("signal", {})),
                            "extras": dict(declared.get("extras", {})),
                            "symbols": list(declared.get("symbols", ())),
                            "sizing_applied": sizing},
            "config_params": cfg.params(),
            "notional_usd": cfg.notional_usd,
            "position_size_pct": cfg.position_size_pct,
            "signal_params": resolved["signal"],
            "zero_cost_reconciliation": reconciliation,
            "coverage": w["cov"],
            "trips": stats["round_trips"],
            "trips_per_session": stats["trips_per_session"],
            "net_pnl": stats["pnl_after_costs"],
            "net_pct": stats["total_return"],
            "final_equity": stats["final_equity"],
            "gross_same_fills": stats["pnl_gross_same_fills"],
            "zero_cost_same_fills": stats["pnl_zero_cost_same_fills"],
            "profit_factor": stats["profit_factor"],
            "win_rate": stats["win_rate"],
            "break_even_win_rate": stats["break_even_win_rate"],
            "max_drawdown": stats["max_drawdown"],
            "max_drawdown_note": stats["max_drawdown_note"],
            "net_bps_per_trip": stats["net_bps_per_trip"],
            "net_bps_per_trip_sd": stats["net_bps_per_trip_sd"],
            "net_bps_t_stat": stats["net_bps_t_stat"],
            "cost_drag_per_trip": stats["cost_drag_per_trip"],
            "cost_drag_per_trip_bps": (stats["cost_drag_per_trip"]
                                       / stats["notional_per_trip"] * 1e4
                                       if stats["notional_per_trip"] else 0.0),
            "cost_drag_total": stats["cost_drag_total"],
            "slip_drag_total": stats["slip_drag_total"],
            "slip_drag_per_trip": stats["slip_drag_per_trip"],
            "slip_drag_per_trip_bps": (stats["slip_drag_per_trip"]
                                       / stats["notional_per_trip"] * 1e4
                                       if stats["notional_per_trip"] else 0.0),
            "fees_paid": stats["fees_paid"],
            "notional_per_trip": stats["notional_per_trip"],
            "gross_notional_per_trip": stats["gross_notional_per_trip"],
            "net_outlay_per_trip": stats["net_outlay_per_trip"],
            "avg_hold_minutes": stats["avg_hold_minutes"],
            "median_hold_minutes": stats["median_hold_minutes"],
            "folds_positive": stats["folds_positive"],
            "folds_total": stats["folds_total"],
            "fold_stability": {"pass": fold_ok, "why": fold_why,
                               "months_with_zero_trips": months_zero},
            "folds": json.loads(folds.to_json(orient="records")),
            "per_symbol": json.loads(persym.to_json(orient="records")),
            "exits": stats["exits"],
            "skipped": stats["skipped"],
            "skipped_total": stats["skipped_total"],
            "identity_max_abs_residual": stats["identity_max_abs_residual"],
            "entries": stats["entries"],
            "signals": stats["signals"],
            "runtime_s": round(elapsed, 2),
        }
        entry_ok = record["entries"] + record["skipped_total"] == record["signals"]
        record["signal_accounting_ok"] = bool(entry_ok)
        if not entry_ok:
            raise RuntimeError(f"{key}: entries + skipped != signals")
        self.recorded[key] = record
        name = record_filename(family, spec["name"], window, cost_label, sizing,
                               resolved_hash, self.sha)
        write_record(self.out_dir / name, record, overwrite=self.overwrite)
        if self.verbose:
            print(f"  {spec['name']:<26} {window} {cost_label:<10} {sizing:<16} "
                  f"trips={stats['round_trips']:>5} net=${stats['pnl_after_costs']:>12,.0f} "
                  f"({stats['total_return']:+.1%}) bps={stats['net_bps_per_trip']:+.2f} "
                  f"PF={stats['profit_factor']:.2f} {elapsed:.1f}s", flush=True)
        return record

    def assert_screen_recorded(self, family: str, specs: Sequence[dict],
                               verdict: dict, where: str) -> None:
        """The recorded set must be **exactly** the set the protocol declares (A3).

        The declared set is the full base matrix — every cell × both sizings on
        W1/baseline, unconditionally — **plus** the branch keys the gate outcomes
        license: W2/baseline for each (cell, sizing) that cleared W1, and both
        windows' zero-cost runs for each that then confirmed on W2.

        The call this replaces asserted ``("W1",) × ("baseline",) × SIZING_MODES``
        only, while :func:`screen.screen_family` also records W2 and zero-cost
        runs for a survivor — and ``assert_exact_run_set`` raises on *extra*
        keys.  A family with a W1 survivor therefore died with
        ``RuntimeError: … unexpected [...]`` **after** the replays and **before**
        any verdict was written, while a family killed on W1 (every round-1
        family) passed.  The guard was inverted: it fired only where it did not
        matter and stayed silent where it did.
        """
        expected = declared_screen_run_set(
            family, specs, verdict.get("w1_pass_pairs", []),
            verdict.get("w2_confirmed_pairs", []))
        mine = {k for k in self.recorded if k.startswith(f"{family}|")}
        assert_exact_run_set(mine, expected, where)

    def assert_round2_grid_recorded(self, family: str, specs: Sequence[dict],
                                    where: str) -> None:
        """The recorded round-2 set must be every cell × both windows (R3-3(iii)).

        Called before any round-2 number is read.  A missing key is a **hard
        failure**, so a cell that could not be resolved or run cannot survive as
        an absence: the grid either ran whole or it did not run.
        """
        expected = declared_round2_run_set(family, specs)
        mine = {k for k in self.recorded if k.startswith(f"{family}|")}
        assert_exact_run_set(mine, expected, where)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="A", help="A, C, D, B or a comma list")
    ap.add_argument("--config", default=None, help="single config name (single run)")
    ap.add_argument("--window", default="W1", choices=sorted(WINDOWS))
    ap.add_argument("--sizing", default="fixed_notional", choices=screen.SIZING_MODES)
    ap.add_argument("--cost", default="baseline", choices=("baseline", "zero_cost"))
    ap.add_argument("--screen", action="store_true",
                    help="run the pre-registered screen for the whole family")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-reconcile", action="store_true",
                    help="skip the zero-cost reconciliation run (not advised)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing run records (refused by default: an "
                         "existing record is evidence)")
    args = ap.parse_args(argv)

    runner = Runner(args.cache_dir, Path(args.out_dir), verbose=not args.quiet,
                    reconcile=not args.no_reconcile, overwrite=args.overwrite)
    fams = [f.strip().upper() for f in args.family.split(",") if f.strip()]
    if not args.screen:
        fam = fams[0]
        specs = families.GRIDS[fam]
        if args.config:
            specs = [s for s in specs if s["name"] == args.config]
            if not specs:
                raise SystemExit(f"no config {args.config!r} in family {fam}")
        for spec in specs:
            runner.run(fam, spec, args.window, args.cost, args.sizing)
        return 0

    verdicts = []
    for fam in fams:
        specs = families.GRIDS[fam]
        print(f"=== family {fam}: screening {len(specs)} configs on W1 ===", flush=True)
        verdict = screen.screen_family(
            fam, specs,
            lambda spec, window, cost, sizing, fam=fam:
                runner.run(fam, spec, window, cost, sizing),
            log=print if not args.quiet else (lambda *_: None))
        # A3: before any verdict is written, the recorded set must equal the set
        # the screen's own stage decisions declare — the W1/baseline matrix for
        # every cell plus the W2 / zero-cost keys each survivor's outcome
        # licenses.  Nothing missing, nothing extra.
        runner.assert_screen_recorded(fam, specs, verdict, f"family {fam} screen")
        verdict["runs"] = {k: _trim(v) for k, v in verdict["runs"].items()}
        (Path(args.out_dir) / f"screen_{fam}.json").write_text(
            json.dumps(verdict, indent=2, default=str))
        print(f"=== family {fam}: {verdict['verdict']} "
              f"(w1: {verdict['w1_survivors']}, w2: {verdict['w2_survivors']}) ===",
              flush=True)
        verdicts.append(verdict)
    summary = [{k: v for k, v in ver.items() if k != "runs"} for ver in verdicts]
    (Path(args.out_dir) / "screen_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    return 0


def _trim(rec: dict) -> dict:
    """Keep the screen's verdicts and headline numbers in the family JSON."""
    keep = ("family", "config", "cell_id", "config_name_resolved", "window",
            "sizing", "engine_sha", "resolved_hash", "resolved_params",
            "identity_key", "declaration",
            "trips", "net_pnl", "net_pct",
            "net_bps_per_trip", "net_bps_t_stat", "cost_drag_per_trip",
            "cost_drag_per_trip_bps", "slip_drag_per_trip",
            "slip_drag_per_trip_bps", "fees_paid", "profit_factor", "win_rate",
            "break_even_win_rate", "max_drawdown", "max_drawdown_note",
            "folds_positive", "folds_total", "fold_stability",
            "avg_hold_minutes", "trips_per_session", "w1_gate", "w2_gate",
            "signal_params", "config_params", "skipped", "skipped_total",
            "entries", "signals", "signal_accounting_ok",
            "identity_max_abs_residual", "gross_notional_per_trip",
            "zero_cost_reconciliation")
    return {k: v for k, v in rec.items() if k in keep}


if __name__ == "__main__":
    raise SystemExit(main())
