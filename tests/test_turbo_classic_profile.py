"""Frozen-profile contract tests for turbo_trader.

These lock the TURBO_PROFILE=classic definition (pre-2026-08-11 turbo) and,
just as importantly, assert that the profile switch can NOT turn off the
safety/accounting guards introduced by #29 (78f746e) and #30 (3f82a9d).
"""
import importlib
import os
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _load(profile: str):
    """Import/reload turbo_trader with TURBO_PROFILE set."""
    os.environ["TURBO_PROFILE"] = profile
    import turbo_trader  # noqa: WPS433 (deliberate late import)

    return importlib.reload(turbo_trader)


def teardown_module(module):  # noqa: WPS442
    os.environ.pop("TURBO_PROFILE", None)


def test_current_profile_is_the_default_and_unchanged():
    t = _load("current")
    assert t.TURBO_PROFILE == "current"
    assert t.CLASSIC_PROFILE is False
    assert t.LEGACY_SELL_CONFIDENCE is False
    assert t.ENABLE_VIOLENCE_TIER is True
    assert t.ENABLE_REGIME_GATE is True
    assert t.ENABLE_SHORT_SELLING is True
    assert t.ENABLE_MEAN_REVERSION_SHORT is True
    assert t.SYMBOLS == list(t.TURBO_SYMBOLS) + list(t.VIOLENCE_SYMBOLS)


def test_missing_env_var_defaults_to_current():
    os.environ.pop("TURBO_PROFILE", None)
    import turbo_trader

    t = importlib.reload(turbo_trader)
    assert t.TURBO_PROFILE == "current"
    assert t.CLASSIC_PROFILE is False


def test_classic_profile_is_the_pre_aug_11_configuration():
    t = _load("classic")
    assert t.TURBO_PROFILE == "classic"
    assert t.CLASSIC_PROFILE is True
    # pool: base 4 only (no VIOLENCE tier)
    assert t.ENABLE_VIOLENCE_TIER is False
    assert t.SYMBOLS == list(t.TURBO_SYMBOLS) == ["SOXL", "TQQQ", "FNGU", "SPXL"]
    # decision-making restored: no gate, long-only, no MR-short, legacy SELL conf
    assert t.ENABLE_REGIME_GATE is False
    assert t.ENABLE_SHORT_SELLING is False
    assert t.ENABLE_MEAN_REVERSION_SHORT is False
    assert t.LEGACY_SELL_CONFIDENCE is True
    # winning-period risk/size/hold constants (identical at HEAD, asserted here)
    assert t.STRATEGY_CONFIG.stop_loss_pct == 0.06
    assert t.STRATEGY_CONFIG.take_profit_pct == 0.08
    assert t.MAX_POSITIONS == 2
    assert t.POSITION_SIZE_PCT == 0.50
    assert t.MAX_HOLD_MINUTES == 30
    assert t.MANDATORY_CLOSE_MINUTES == 30
    assert t.CONFIDENCE_THRESHOLD == 0.4


def test_profile_never_disables_the_safety_and_accounting_guards():
    for profile in ("current", "classic"):
        t = _load(profile)
        # #29 guards
        assert t.BUYING_POWER_USAGE_PCT == 0.95
        assert t.SHORT_WHOLE_SHARES_ONLY is True
        assert t.SHORT_REQUIRE_BROKER_SHORTABLE is True
        # #30 accounting is imported, not flag-gated
        assert callable(t.close_position_verified)


def _soft_decline_frame():
    """30 gently declining 1-minute closes: last close just below its MA10."""
    closes = [100.0 - 0.05 * i for i in range(30)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-08-11 13:30", periods=30, freq="min", tz="UTC"),
            "open": closes,
            "high": [c + 0.02 for c in closes],
            "low": [c - 0.02 for c in closes],
            "close": closes,
            "volume": [1000] * 30,
        }
    )


def test_legacy_sell_confidence_cannot_reach_the_activation_threshold():
    """The pre-#23 formula scores a small distance-from-MA move far below 0.4,
    which is why the momentum SELL never fired during the winning period."""
    data = _soft_decline_frame()
    classic = _load("classic")
    legacy = classic._generate_momentum_signals(data, symbol="SOXL", **{
        k: classic.MOMENTUM_CONFIG[k] for k in ("ma_period", "trend_periods", "rsi_period", "rsi_threshold")
    })
    legacy_sells = [s for s in legacy if s.signal_type.name == "SELL"]
    assert legacy_sells, "expected a SELL signal from the declining series"
    assert max(s.confidence for s in legacy_sells) < classic.CONFIDENCE_THRESHOLD

    current = _load("current")
    modern = current._generate_momentum_signals(data, symbol="SOXL", **{
        k: current.MOMENTUM_CONFIG[k] for k in ("ma_period", "trend_periods", "rsi_period", "rsi_threshold")
    })
    modern_sells = [s for s in modern if s.signal_type.name == "SELL"]
    assert modern_sells
    assert max(s.confidence for s in modern_sells) >= current.CONFIDENCE_THRESHOLD
