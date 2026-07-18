"""Common technical indicators shared across trading strategies.

Every indicator accepts a ``pd.Series`` of prices (typically close) and
returns one or more ``pd.Series`` (or tuples of Series) aligned to the
same index.  Strategies use these as building blocks in ``generate_signals``.
"""

from __future__ import annotations

import pandas as pd


def sma(prices: pd.Series, period: int = 20) -> pd.Series:
    """Simple Moving Average over *period* bars.

    Returns a Series with the same index; leading ``period-1`` values
    are NaN.
    """
    return prices.rolling(window=period).mean()


def rolling_high(prices: pd.Series, period: int = 20) -> pd.Series:
    """Rolling maximum of *prices* over *period* bars."""
    return prices.rolling(window=period).max()


def rolling_low(prices: pd.Series, period: int = 20) -> pd.Series:
    """Rolling minimum of *prices* over *period* bars."""
    return prices.rolling(window=period).min()


def bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_dev_multiplier: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands — SMA ± (std_dev * multiplier).

    Returns (middle, upper, lower) as three Series.
    """
    middle = sma(prices, period)
    std = prices.rolling(window=period).std()
    upper = middle + std * std_dev_multiplier
    lower = middle - std * std_dev_multiplier
    return middle, upper, lower


def z_score(prices: pd.Series, period: int = 20) -> pd.Series:
    """Rolling Z-score: (price – SMA) / rolling_std.

    Measures how many standard deviations price is from its rolling mean.
    """
    middle = sma(prices, period)
    std = prices.rolling(window=period).std()
    return (prices - middle) / std


def momentum(prices: pd.Series, period: int = 20) -> pd.Series:
    """Normalised momentum: (price – SMA) / SMA.

    Negative values indicate downward momentum; positive = upward.
    Typically used with an entry/exit threshold.
    """
    ma = sma(prices, period)
    return (prices - ma) / ma


def breakout_distance(
    prices: pd.Series,
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """Distance above rolling-high and below rolling-low channels.

    Returns (upside_breakout, downside_breakout) as two Series.
    Positive values mean price is beyond the channel.

    The channel is computed from *past* bars only (shifted by 1) so the
    current bar can break out of it.
    """
    # Compute rolling extremes on the *previous* period bars so the
    # current bar is compared against a channel it hasn't influenced yet.
    rh = rolling_high(high.shift(1), period)
    rl = rolling_low(low.shift(1), period)
    upside = (prices - rh) / rh  # > 0 means breakout above
    downside = (rl - prices) / rl  # > 0 means breakdown below
    return upside, downside
