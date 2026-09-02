"""Degradation-aware grading of candidate parameter sets.

The core anti-overfitting discipline: a candidate is judged **both** on its
absolute out-of-sample objective **and** on how much it degrades from
in-sample to out-of-sample.  A parameter set that looks great in-sample but
collapses out-of-sample is over-fit and must be graded down or discarded —
no matter how good its IS score was.

Grades (S / A / B / C / F) are assigned from two gates:

1. **OOS minimum profit factor** — the absolute bar a candidate must clear
   out-of-sample (S >= 1.5, A >= 1.2, B >= 1.05, C >= 1.0).
2. **Max degradation** — the allowed relative drop of the objective from IS
   to OOS (S <= 25 %, A <= 40 %, B <= 60 %, C <= 80 %).

Anything that loses money out-of-sample (PF < 1.0), has too few OOS trades
to be meaningful, or degrades more than 80 % is graded **F** (discard).
Thresholds are configurable via :class:`GradeThresholds`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# The objective scores compared here are profit-factor-based (see
# ``src.optimization.realized.trade_objective``).  A PF of exactly 1.0 means
# breakeven; anything below 1.0 loses money.
BREAKEVEN_PF = 1.0

_GRADES = ("S", "A", "B", "C", "F")


@dataclass
class GradeThresholds:
    """Cut-offs used by :func:`grade_config`.

    Attributes
    ----------
    min_oos_pf : dict[str, float]
        Per-grade minimum OOS profit factor (higher grade = higher bar).
    max_degradation : dict[str, float]
        Per-grade maximum allowed relative degradation (IS -> OOS).
    min_oos_trades : int
        Minimum number of OOS trades for a grade to be meaningful; below
        this the candidate is graded F (insufficient evidence).
    """

    min_oos_pf: dict[str, float] = field(
        default_factory=lambda: {"S": 1.5, "A": 1.2, "B": 1.05, "C": 1.0}
    )
    max_degradation: dict[str, float] = field(
        default_factory=lambda: {"S": 0.25, "A": 0.40, "B": 0.60, "C": 0.80}
    )
    min_oos_trades: int = 8

    def validate(self) -> None:
        """Sanity-check that thresholds are monotonic and positive."""
        if not (set(self.min_oos_pf) == set(self.max_degradation) == set(_GRADES[:-1])):
            raise ValueError("Grade thresholds must define S, A, B, C")
        pfs = list(self.min_oos_pf.values())
        degs = list(self.max_degradation.values())
        if any(p <= 0 for p in pfs) or any(d <= 0 for d in degs):
            raise ValueError("Grade thresholds must be positive")


def degradation_ratio(is_score: float, oos_score: float) -> float:
    """Relative drop of the objective from IS to OOS.

    ``(is_score - oos_score) / |is_score|``.  Positive values mean the
    candidate degraded out-of-sample; negative values mean it *improved*
    out-of-sample.  Handles ``inf``/``-inf``/``NaN`` defensively.

    Returns
    -------
    float
        The degradation ratio (0.40 = the OOS score is 40 % below IS).
        ``inf`` means the IS score was positive but OOS collapsed, or IS was
        ``-inf`` (no trades in-sample) and OOS is finite.
    """
    if not np.isfinite(is_score) or not np.isfinite(oos_score):
        if not np.isfinite(is_score) and np.isfinite(oos_score):
            # IS had no valid trades; OOS does — can't measure degradation.
            return float("inf")
        if np.isfinite(is_score) and not np.isfinite(oos_score):
            return float("inf")  # OOS produced no trades → total collapse
        return 0.0  # both non-finite — treat as no-information, not a penalty
    denom = abs(is_score)
    if denom < 1e-12:
        return 0.0 if oos_score >= is_score else float("inf")
    return float((is_score - oos_score) / denom)


@dataclass
class Grade:
    """Grading verdict for a single candidate parameter set.

    Attributes
    ----------
    letter : str
        ``S`` / ``A`` / ``B`` / ``C`` / ``F``.
    is_score / oos_score : float
        Objective scores on the in-sample and out-of-sample windows.
    oos_n_trades : int
        Number of OOS trades the score was computed from.
    degradation : float
        IS -> OOS relative degradation (see :func:`degradation_ratio`).
    passed : bool
        True when the candidate is worth considering for adoption (A or S).
    reason : str
        Short human-readable explanation of the verdict.
    """

    letter: str
    is_score: float
    oos_score: float
    oos_n_trades: int
    degradation: float
    passed: bool = False
    reason: str = ""

    def as_dict(self) -> dict:
        """Plain dict for serialisation into reports."""
        return {
            "grade": self.letter,
            "is_score": _fmt(self.is_score),
            "oos_score": _fmt(self.oos_score),
            "oos_n_trades": int(self.oos_n_trades),
            "degradation": _fmt(self.degradation),
            "passed": self.passed,
            "reason": self.reason,
        }


def _fmt(x: float) -> str:
    if not np.isfinite(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.3f}"


def grade_config(
    is_score: float,
    oos_score: float,
    oos_n_trades: int,
    thresholds: GradeThresholds | None = None,
) -> Grade:
    """Grade a candidate by its IS score, OOS score and OOS trade count.

    Parameters
    ----------
    is_score : float
        In-sample objective (profit-factor based).
    oos_score : float
        Out-of-sample objective (profit-factor based).
    oos_n_trades : int
        Number of trades the OOS score came from.
    thresholds : GradeThresholds, optional
        Grading cut-offs.  Defaults to :data:`DEFAULT_THRESHOLDS`.

    Returns
    -------
    Grade
    """
    thr = thresholds or DEFAULT_THRESHOLDS
    thr.validate()

    deg = degradation_ratio(is_score, oos_score)

    # --- Hard failures ---------------------------------------------
    if oos_n_trades < thr.min_oos_trades:
        return Grade(
            letter="F",
            is_score=is_score,
            oos_score=oos_score,
            oos_n_trades=oos_n_trades,
            degradation=deg,
            passed=False,
            reason=(
                f"F: only {oos_n_trades} OOS trades (< {thr.min_oos_trades}) — "
                "insufficient out-of-sample evidence"
            ),
        )
    if not np.isfinite(oos_score) or oos_score < BREAKEVEN_PF:
        return Grade(
            letter="F",
            is_score=is_score,
            oos_score=oos_score,
            oos_n_trades=oos_n_trades,
            degradation=deg,
            passed=False,
            reason=(
                "F: out-of-sample profit factor below breakeven (1.0) — "
                "the edge does not survive out-of-sample"
            ),
        )
    if deg > thr.max_degradation["C"]:
        return Grade(
            letter="F",
            is_score=is_score,
            oos_score=oos_score,
            oos_n_trades=oos_n_trades,
            degradation=deg,
            passed=False,
            reason=(
                f"F: OOS degradation {deg:.0%} exceeds the C ceiling "
                f"({thr.max_degradation['C']:.0%}) — over-fit candidate"
            ),
        )

    # --- Best passing grade ----------------------------------------
    for letter in ("S", "A", "B", "C"):
        if (oos_score >= thr.min_oos_pf[letter]) and (deg <= thr.max_degradation[letter]):
            grade = Grade(
                letter=letter,
                is_score=is_score,
                oos_score=oos_score,
                oos_n_trades=oos_n_trades,
                degradation=deg,
                passed=letter in ("S", "A"),
                reason=(
                    f"{letter}: OOS PF {oos_score:.2f} >= {thr.min_oos_pf[letter]:.2f} "
                    f"and degradation {deg:.0%} <= {thr.max_degradation[letter]:.0%}"
                ),
            )
            return grade

    # Degrades too much for B/C but is still profitable OOS.
    return Grade(
        letter="C",
        is_score=is_score,
        oos_score=oos_score,
        oos_n_trades=oos_n_trades,
        degradation=deg,
        passed=False,
        reason=(
            f"C: OOS PF {oos_score:.2f} >= 1.0 but degradation {deg:.0%} "
            "exceeds the B ceiling — marginal, do not adopt without more data"
        ),
    )


DEFAULT_THRESHOLDS = GradeThresholds()


def grade_table(
    rows: list[dict],
    thresholds: GradeThresholds | None = None,
) -> pd.DataFrame:
    """Build a tidy DataFrame of grades from candidate score rows.

    Each row of *rows* should have keys ``config``, ``is_score``, ``oos_score``,
    ``oos_n_trades`` (and optionally ``is_n_trades``, ``holdout_score``,
    ``holdout_n_trades``).  Returns a DataFrame sorted best-grade first,
    each row carrying grade, degradation and pass/fail.
    """
    thr = thresholds or DEFAULT_THRESHOLDS
    thr.validate()

    records: list[dict] = []
    for row in rows:
        g = grade_config(
            row["is_score"], row["oos_score"], row["oos_n_trades"], thr,
        )
        record = {
            "config": row["config"],
            "is_score": row["is_score"],
            "is_n_trades": row.get("is_n_trades", 0),
            "oos_score": row["oos_score"],
            "oos_n_trades": row["oos_n_trades"],
            "degradation": g.degradation,
            "grade": g.letter,
            "passed": g.passed,
            "holdout_score": row.get("holdout_score", float("nan")),
            "holdout_n_trades": row.get("holdout_n_trades", 0),
            "reason": g.reason,
        }
        records.append(record)

    df = pd.DataFrame(records)
    if not df.empty:
        order = {"S": 0, "A": 1, "B": 2, "C": 3, "F": 4}
        df["_grade_rank"] = df["grade"].map(order)
        df = (
            df.sort_values(["_grade_rank", "oos_score"], ascending=[True, False])
            .drop(columns="_grade_rank")
            .reset_index(drop=True)
        )
    return df