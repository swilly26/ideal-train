"""Data validation and cleaning utilities for market data.

Strategies consume data from multiple providers and expect a consistent
format. This module validates, cleans, and normalises incoming OHLCV
DataFrames so that downstream code doesn't need to deal with edge cases.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standardised OHLCV column names
_STANDARD_COLUMNS = ["open", "high", "low", "close", "volume"]

# Accepted column name variations from providers
_COLUMN_ALIASES: dict[str, str] = {
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the standard lowercase OHLCV format.

    Supported input column names include ``Open`` / ``open``, ``O`` / ``o``,
    ``H`` / ``h``, etc.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with OHLCV columns in any recognised casing.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardised ``open, high, low, close, volume`` columns.
    """
    df = df.copy()
    rename = {}
    for col in df.columns:
        if col in _COLUMN_ALIASES:
            rename[col] = _COLUMN_ALIASES[col]
    if rename:
        df = df.rename(columns=rename)
    return df


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Check that *df* has all five standard OHLCV columns.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    missing = set(_STANDARD_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(missing)}. "
            f"Found: {sorted(df.columns)}"
        )
    return df


def validate_market_data(
    df: pd.DataFrame,
    *,
    allow_nan: bool = False,
    max_gap_minutes: Optional[int] = None,
) -> list[str]:
    """Validate an OHLCV DataFrame and return a list of issues found.

    Checks performed:

    * Required columns are present.
    * Prices are non-negative.
    * High >= Low for every bar.
    * Open and Close are within the [Low, High] range.
    * Volume is non-negative.
    * No NaN values (unless *allow_nan* is ``True``).
    * No zero-volume bars.
    * *(optional)* No gaps larger than *max_gap_minutes*.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with standardised columns.
    allow_nan : bool
        If ``True``, NaN values do not trigger a warning.
    max_gap_minutes : int | None
        If set, flags any gap between consecutive timestamps larger than
        this many minutes.

    Returns
    -------
    list[str]
        Human-readable descriptions of each issue found. Empty list means
        the data is clean.
    """
    issues: list[str] = []

    # Required columns
    try:
        ensure_required_columns(df)
    except ValueError as exc:
        issues.append(str(exc))
        return issues  # Can't validate further without columns

    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # NaN check
    if not allow_nan:
        nan_cols = [col for col in _STANDARD_COLUMNS if df[col].isna().any()]
        if nan_cols:
            nan_counts = {col: int(df[col].isna().sum()) for col in nan_cols}
            issues.append(f"NaN values found: {nan_counts}")

    # Negative prices
    for col in ["open", "high", "low", "close"]:
        neg_count = int((df[col] < 0).sum())
        if neg_count > 0:
            issues.append(f"Negative {col} values: {neg_count} rows")

    # High < Low
    hl_bad = int((h < l).sum())
    if hl_bad > 0:
        issues.append(f"High < Low in {hl_bad} rows")

    # Open / Close outside [Low, High]
    o_bad = int(((o < l) | (o > h)).sum())
    c_bad = int(((c < l) | (c > h)).sum())
    if o_bad > 0:
        issues.append(f"Open outside [Low, High] in {o_bad} rows")
    if c_bad > 0:
        issues.append(f"Close outside [Low, High] in {c_bad} rows")

    # Negative volume
    neg_vol = int((v < 0).sum())
    if neg_vol > 0:
        issues.append(f"Negative volume in {neg_vol} rows")

    # Zero volume
    zero_vol = int((v == 0).sum())
    if zero_vol > 0:
        issues.append(f"Zero volume in {zero_vol} rows (may indicate bad data)")

    # Gaps
    if max_gap_minutes is not None and isinstance(df.index, pd.DatetimeIndex):
        diffs = df.index.to_series().diff().dropna()
        gap_mask = diffs > pd.Timedelta(minutes=max_gap_minutes)
        gap_count = int(gap_mask.sum())
        if gap_count > 0:
            max_gap = diffs[gap_mask].max()
            issues.append(
                f"{gap_count} gaps larger than {max_gap_minutes} min "
                f"(max gap: {max_gap})"
            )

    return issues


def detect_outliers(
    df: pd.DataFrame,
    *,
    price_zscore: float = 5.0,
    volume_zscore: float = 8.0,
) -> pd.DataFrame:
    """Flag outlier bars whose price change or volume is extreme.

    Uses z-score per column: a bar is flagged if any of ``open, high, low,
    close`` has a z-score beyond *price_zscore*, or ``volume`` beyond
    *volume_zscore*.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    price_zscore : float
        Z-score threshold for price columns.
    volume_zscore : float
        Z-score threshold for volume.

    Returns
    -------
    pd.DataFrame
        Boolean mask with the same shape as *df*, ``True`` where a value
        is an outlier.
    """
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            std_val = df[col].std(ddof=0)
            std_val = max(float(std_val), 1e-9)
            z = (df[col] - df[col].mean()).abs() / std_val
            mask[col] = z > price_zscore

    if "volume" in df.columns:
        std_val = df["volume"].std(ddof=0)
        std_val = max(float(std_val), 1e-9)
        z = (df["volume"] - df["volume"].mean()).abs() / std_val
        mask["volume"] = z > volume_zscore

    return mask


def clean_market_data(
    df: pd.DataFrame,
    *,
    remove_nan: bool = True,
    remove_outliers: bool = False,
    price_zscore: float = 5.0,
) -> pd.DataFrame:
    """Clean an OHLCV DataFrame by removing bad rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame (columns will be normalised).
    remove_nan : bool
        Drop rows with any NaN values.
    remove_outliers : bool
        Drop rows flagged as outliers (see :func:`detect_outliers`).
    price_zscore : float
        Z-score threshold passed to :func:`detect_outliers`.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = normalize_columns(df)

    if remove_nan:
        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        if dropped > 0:
            logger.info("Dropped %d rows with NaN values", dropped)

    # Fix High < Low by swapping
    bad_hl = df["high"] < df["low"]
    if bad_hl.any():
        logger.warning("Swapping High/Low in %d rows", bad_hl.sum())
        df.loc[bad_hl, ["high", "low"]] = df.loc[bad_hl, ["low", "high"]].values

    # Clamp Open/Close to [Low, High]
    for col in ["open", "close"]:
        df[col] = df[col].clip(lower=df["low"], upper=df["high"])

    # Zero or negative volume → NaN then drop
    bad_vol = df["volume"] <= 0
    if bad_vol.any():
        logger.warning("Removing %d rows with non-positive volume", bad_vol.sum())
        df.loc[bad_vol, "volume"] = np.nan

    if remove_nan:
        df = df.dropna()

    if remove_outliers:
        outliers = detect_outliers(df, price_zscore=price_zscore)
        outlier_rows = outliers.any(axis=1)
        if outlier_rows.any():
            logger.info("Dropped %d outlier rows", outlier_rows.sum())
            df = df[~outlier_rows]

    return df
