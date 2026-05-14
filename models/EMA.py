import pandas as pd
import numpy as np


def compute_ema_crossover_signal(prices_window: pd.DataFrame, fast=9, slow=20) -> pd.Series:
    """
    EMA crossover signal per stock.

    Returns pd.Series with values:
      +1  fast EMA > slow EMA
      -1  fast EMA < slow EMA
       0  insufficient data or equal
    """
    if len(prices_window) < slow + 5:
        return pd.Series(0, index=prices_window.columns)

    fast_ema = prices_window.ewm(span=fast, adjust=False).mean().iloc[-1]
    slow_ema = prices_window.ewm(span=slow, adjust=False).mean().iloc[-1]

    signal = pd.Series(0, index=prices_window.columns)
    signal[fast_ema > slow_ema] =  1
    signal[fast_ema < slow_ema] = -1

    return signal


def compute_adx(high_window: pd.DataFrame, low_window: pd.DataFrame,
                close_window: pd.DataFrame, period=14) -> pd.Series:
    """
    ADX (Average Directional Index) per stock using true ATR (High, Low, Close).

    Returns pd.Series of ADX values [0, 100].
    """
    min_len = period * 2
    if len(close_window) < min_len:
        return pd.Series(np.nan, index=close_window.columns)

    up   = high_window.diff()
    down = -low_window.diff()

    plus_dm  = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    # True Range: max(H-L, |H-prev_C|, |L-prev_C|)
    prev_close = close_window.shift(1)
    tr = pd.concat([
        (high_window - low_window).abs(),
        (high_window - prev_close).abs(),
        (low_window  - prev_close).abs(),
    ]).groupby(level=0).max()

    # Wilder smoothing via EWM
    smoothed_plus_dm  = plus_dm.ewm(span=period, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(span=period, adjust=False).mean()
    smoothed_atr      = tr.ewm(span=period, adjust=False).mean()

    eps = 1e-10
    plus_di  = 100 * smoothed_plus_dm.iloc[-1]  / (smoothed_atr.iloc[-1] + eps)
    minus_di = 100 * smoothed_minus_dm.iloc[-1] / (smoothed_atr.iloc[-1] + eps)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + eps)

    return dx


def compute_volume_ratio(volume_window: pd.DataFrame) -> pd.Series:
    """
    Volume ratio: latest volume / 20-day average volume.

    Zeros are treated as missing (replaced with NaN) so they fail the >= 1.2 gate.

    Returns pd.Series of ratio values.
    """
    vol = volume_window.replace(0, np.nan)

    if len(vol) < 21:
        return pd.Series(np.nan, index=volume_window.columns)

    latest      = vol.iloc[-1]
    avg_20      = vol.iloc[-20:].mean()
    ratio       = latest / avg_20

    return ratio
