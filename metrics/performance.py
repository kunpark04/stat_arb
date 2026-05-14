import numpy as np
import pandas as pd

TRADING_DAYS = 252

def performance_analysis(cum_rets, rets, show_stats=True, risk_free_rate=0.0):
    """
    Compute strategy performance metrics.

    Parameters
    ----------
    cum_rets        : Cumulative returns Series or DataFrame.
    rets            : Daily net returns Series or DataFrame.
    show_stats      : Print a summary table when True.
    risk_free_rate  : Annualized risk-free rate (e.g. 0.04 for 4%). Default 0.
    """
    rf_daily = risk_free_rate / TRADING_DAYS

    # --- Total & Annualized Return ---
    total_ret     = (cum_rets.iloc[-1] - 1) * 100
    n_days        = len(rets)
    ann_ret       = ((cum_rets.iloc[-1]) ** (TRADING_DAYS / n_days) - 1) * 100

    # --- Sharpe Ratio (excess return) ---
    excess_daily  = rets.mean(axis=0) - rf_daily
    sharpe_ratio  = (excess_daily / rets.std(axis=0)) * np.sqrt(TRADING_DAYS)

    # --- Sortino Ratio (penalizes downside vol only) ---
    downside_rets = rets.copy()
    downside_rets[downside_rets > 0] = 0
    downside_std  = downside_rets.std(axis=0)
    sortino_ratio = (excess_daily / (downside_std + 1e-10)) * np.sqrt(TRADING_DAYS)

    # --- Drawdown Metrics ---
    running_max   = cum_rets.cummax()
    drawdowns     = (cum_rets / running_max) - 1
    max_dd        = drawdowns.min(axis=0) * 100

    # --- Calmar Ratio ---
    calmar_ratio  = ann_ret / (abs(max_dd) + 1e-10)

    # Compile metrics into a DataFrame
    metrics = pd.DataFrame({
        'Total Return (%)':      np.array(total_ret,     ndmin=1),
        'Annualized Return (%)': np.array(ann_ret,       ndmin=1),
        'Sharpe Ratio':          np.array(sharpe_ratio,  ndmin=1),
        'Sortino Ratio':         np.array(sortino_ratio, ndmin=1),
        'Max Drawdown (%)':      np.array(max_dd,        ndmin=1),
        'Calmar Ratio':          np.array(calmar_ratio,  ndmin=1),
    })
    metrics = metrics.apply(pd.to_numeric, errors='coerce')

    # Statistical summary (used for Monte Carlo distributions)
    stats_summary = metrics.describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ).T
    stats_summary['skew']     = metrics.skew()
    stats_summary['kurtosis'] = metrics.kurtosis()
    order = ['count', 'mean', 'std', 'min', 'max',
             '1%', '5%', '25%', '50%', '75%', '95%', '99%', 'skew', 'kurtosis']
    stats_summary = stats_summary[order]

    if show_stats:
        report_type = "Monte Carlo" if isinstance(total_ret, pd.Series) else "Strategy Backtest"
        print("=" * 50)
        print(f"  PERFORMANCE REPORT — {report_type}")
        print("=" * 50)
        m = stats_summary
        print(f"  Total Return:        {m.loc['Total Return (%)',      'mean']:>8.2f}%")
        print(f"  Annualized Return:   {m.loc['Annualized Return (%)', 'mean']:>8.2f}%")
        print(f"  Sharpe Ratio:        {m.loc['Sharpe Ratio',          'mean']:>8.2f}")
        print(f"  Sortino Ratio:       {m.loc['Sortino Ratio',         'mean']:>8.2f}")
        print(f"  Max Drawdown:        {m.loc['Max Drawdown (%)',       'mean']:>8.2f}%")
        print(f"  Calmar Ratio:        {m.loc['Calmar Ratio',           'mean']:>8.2f}")
        print("=" * 50)
        print()

    return metrics, stats_summary
