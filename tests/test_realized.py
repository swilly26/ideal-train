"""Tests for realized trade loading and trade-objective scoring."""

import numpy as np
import pandas as pd
import pytest

from src.optimization.realized import (
    PF_CAP,
    load_realized_trades,
    profit_factor,
    realized_metrics,
    summarize_realized_window,
    trade_objective,
)

SAMPLE_CSV = """trader,date,time,symbol,entry,exit,pnl,pct,reason,strategy
MAIN,2026-07-27,15:32:11,AAPL,338.44,338.93,14.59,0.15,signal,
MAIN,2026-07-27,15:40:17,TSLA,307.93,308.51,18.66,0.19,signal,
MAIN,2026-07-28,09:31:00,SPY,738.26,737.16,-14.82,-0.15,signal,
MAIN,2026-07-28,09:45:00,SPY,736.73,736.99,3.52,0.04,signal,
"""


def _write_sample(tmp_path):
    p = tmp_path / "trades.csv"
    p.write_text(SAMPLE_CSV)
    return p


def test_load_realized_trades_normalizes(tmp_path):
    df = load_realized_trades(_write_sample(tmp_path))
    assert len(df) == 4
    assert {"exit_dt", "pnl", "pct", "symbol"} <= set(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df["exit_dt"])
    assert df["exit_dt"].is_monotonic_increasing
    assert df.iloc[0]["symbol"] == "AAPL"


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_realized_trades(tmp_path / "nope.csv")


def test_load_bad_columns_raises(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_realized_trades(p)


def test_profit_factor():
    trades = pd.DataFrame({"pnl": [100.0, 50.0, -25.0, -25.0]})
    assert profit_factor(trades) == pytest.approx(3.0)
    assert profit_factor(pd.DataFrame({"pnl": []})) == 0.0
    assert profit_factor(pd.DataFrame({"pnl": [1.0, 2.0]})) == float("inf")


def test_realized_metrics():
    trades = pd.DataFrame(
        {"pnl": [100.0, 50.0, -25.0, -25.0], "pct": [1.0, 0.5, -0.3, -0.3]}
    )
    m = realized_metrics(trades)
    assert m["n_trades"] == 4
    assert m["profit_factor"] == pytest.approx(3.0)
    assert m["win_rate"] == pytest.approx(0.5)
    assert m["net_pnl"] == pytest.approx(100.0)
    assert m["avg_pct"] == pytest.approx(0.225)


def test_trade_objective_min_trades_gate():
    few = pd.DataFrame({"pnl": [100.0]})
    assert trade_objective(few, min_trades=8) == float("-inf")
    many = pd.DataFrame({"pnl": [10.0] * 8 + [-2.0] * 2})
    assert trade_objective(many, min_trades=8) == pytest.approx(80.0 / 4.0)


def test_trade_objective_pf_cap():
    all_winners = pd.DataFrame({"pnl": [1.0] * 12})
    assert trade_objective(all_winners, min_trades=8) == pytest.approx(PF_CAP)


def test_summarize_realized_window(tmp_path):
    df = load_realized_trades(_write_sample(tmp_path))
    summarized = summarize_realized_window(
        df,
        is_days=["2026-07-27"],
        oos_days=["2026-07-28"],
        holdout_days=[],
    )
    assert summarized["is"]["n_trades"] == 2
    assert summarized["oos"]["n_trades"] == 2
    assert summarized["holdout"]["n_trades"] == 0
    assert summarized["is"]["days"] == ["2026-07-27"]