"""Tests for degradation-based S/A/B/C/F grading."""

import pandas as pd
import pytest

from src.optimization.degradation import (
    DEFAULT_THRESHOLDS,
    Grade,
    GradeThresholds,
    degradation_ratio,
    grade_config,
    grade_table,
)


def test_degradation_ratio_basic():
    assert degradation_ratio(2.0, 1.0) == pytest.approx(0.5)
    assert degradation_ratio(1.5, 1.8) == pytest.approx(-0.2)  # improved OOS
    assert degradation_ratio(0.0, 0.0) == 0.0
    assert degradation_ratio(2.0, float("inf")) == float("inf")
    assert degradation_ratio(float("-inf"), 1.5) == float("inf")
    assert degradation_ratio(float("-inf"), float("-inf")) == 0.0


def test_grade_s_passes():
    g = grade_config(is_score=2.0, oos_score=1.7, oos_n_trades=12)
    assert g.letter == "S"
    assert g.passed is True


def test_grade_a_passes():
    g = grade_config(is_score=1.6, oos_score=1.3, oos_n_trades=15)
    assert g.letter == "A"
    assert g.passed is True


def test_grade_b_marginal_not_passed():
    g = grade_config(is_score=1.4, oos_score=1.1, oos_n_trades=10)
    assert g.letter == "B"
    assert g.passed is False  # B/C are "do not adopt without more data"


def test_grade_f_insufficient_oos_trades():
    g = grade_config(is_score=2.0, oos_score=1.6, oos_n_trades=3)
    assert g.letter == "F"
    assert "insufficient" in g.reason.lower()


def test_grade_f_oos_loses_money():
    g = grade_config(is_score=1.9, oos_score=0.8, oos_n_trades=20)
    assert g.letter == "F"
    assert "below breakeven" in g.reason.lower()


def test_grade_f_overfit_degrades_hard():
    # Great IS, barely-breakeven OOS, but 90% degradation → must be discarded.
    g = grade_config(is_score=10.0, oos_score=1.01, oos_n_trades=20)
    assert g.letter == "F"
    assert "over-fit" in g.reason.lower()


def test_grade_c_when_degradation_between_b_and_c():
    # IS 3.0 → OOS 1.1: degradation 63%, above B ceiling (60%), below C (80%).
    g = grade_config(is_score=3.0, oos_score=1.1, oos_n_trades=15)
    assert g.letter == "C"
    assert g.passed is False


def test_custom_thresholds():
    thr = GradeThresholds(min_oos_pf={"S": 1.2, "A": 1.1, "B": 1.03, "C": 1.0},
                          max_degradation={"S": 0.5, "A": 0.6, "B": 0.7, "C": 0.9},
                          min_oos_trades=4)
    g = grade_config(is_score=1.3, oos_score=1.25, oos_n_trades=6, thresholds=thr)
    assert g.letter == "S"


def test_invalid_thresholds_raise():
    thr = GradeThresholds(min_oos_pf={"S": 1.0, "A": 1.0, "B": 1.0, "C": 1.0},
                          max_degradation={"S": 0, "A": 0, "B": 0, "C": 0},
                          min_oos_trades=4)
    with pytest.raises(ValueError):
        grade_config(1.0, 1.0, 5, thresholds=thr)


def test_grade_as_dict_serializable():
    g = grade_config(is_score=2.0, oos_score=1.7, oos_n_trades=12)
    d = g.as_dict()
    assert d["grade"] == "S"
    assert d["oos_score"] == "1.700"


def test_grade_table_sorts_best_first():
    rows = [
        {"config": "bad", "is_score": 2.5, "oos_score": 0.7, "oos_n_trades": 20},
        {"config": "good", "is_score": 1.8, "oos_score": 1.5, "oos_n_trades": 14},
        {"config": "marginal", "is_score": 1.2, "oos_score": 1.05, "oos_n_trades": 9},
    ]
    df = grade_table(rows)
    assert df.iloc[0]["config"] == "good"
    assert list(df["grade"]) == ["S", "B", "F"]
    assert "holdout_score" in df.columns