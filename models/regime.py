import numpy as np
import pandas as pd


def compute_realized_vol_score(market_ret: pd.Series) -> pd.Series:
    rolling_vol = market_ret.rolling(20).std() * np.sqrt(252)
    score = pd.cut(rolling_vol, bins=[-np.inf, 0.12, 0.22, np.inf], labels=[0, 1, 2])
    return score.astype(float).shift(1)


def compute_variance_ratio_score(market_ret: pd.Series) -> pd.Series:
    var_1d = market_ret.rolling(60).var().replace(0, np.nan)
    var_kd = market_ret.rolling(5).sum().rolling(60).var()
    vr = var_kd / (5 * var_1d)
    score = pd.cut(vr, bins=[-np.inf, 0.85, 1.15, np.inf], labels=[0, 1, 2])
    return score.astype(float).shift(1)


def compute_avg_correlation_score(rets: pd.DataFrame) -> pd.Series:
    n_stocks = rets.shape[1]
    np.random.seed(42)
    sample = rets.sample(min(40, n_stocks), axis=1)

    rolling_corr = sample.rolling(30).corr()

    avg_corr = pd.Series(np.nan, index=rets.index)
    stocks = sample.columns
    n = len(stocks)

    for date in rets.index:
        try:
            corr_mat = rolling_corr.loc[date]
            if corr_mat.shape == (n, n):
                vals = corr_mat.values
                mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                upper = np.abs(vals[mask])
                avg_corr[date] = np.nanmean(upper)
        except (KeyError, TypeError):
            pass

    score = pd.cut(avg_corr, bins=[-np.inf, 0.25, 0.45, np.inf], labels=[0, 1, 2])
    return score.astype(float).shift(1)


def detect_regime(rets: pd.DataFrame, return_signals: bool = False):
    """
    Detect market regime from returns DataFrame.

    Parameters
    ----------
    rets           : DataFrame of daily stock returns (stocks × dates).
    return_signals : If True, also return (regime, composite, vol_score, vr_score, corr_score).

    Returns
    -------
    regime_series  : pd.Series[Int64] with values 0 (favorable), 1 (neutral), 2 (trending).
                     First ~69 dates will be NaN (warmup period).
    """
    market_ret = rets.mean(axis=1)

    vol_score  = compute_realized_vol_score(market_ret)
    vr_score   = compute_variance_ratio_score(market_ret)
    corr_score = compute_avg_correlation_score(rets)

    composite = vol_score + vr_score + corr_score
    smoothed  = composite.rolling(5).median().round()
    regime    = pd.cut(smoothed, bins=[-1, 1, 4, 6], labels=[0, 1, 2]).astype('Int64')

    if return_signals:
        return regime, composite, vol_score, vr_score, corr_score
    return regime
