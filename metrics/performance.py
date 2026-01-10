import numpy as np
import pandas as pd

def performance_analysis(cum_rets, rets, show_stats=True):
    # Total return
    total_ret = (cum_rets.iloc[-1] - 1) * 100

    # Sharpe ratio
    sharpe_ratio = (rets.mean(axis=0) / rets.std(axis=0)) * np.sqrt(252)

    # Drawdown calculation
    running_max = cum_rets.cummax()
    drawdowns = (cum_rets / running_max) - 1
    max_dd = drawdowns.min(axis=0) * 100

    # Compile metrics
    metrics = pd.DataFrame({'Total Return': np.array(total_ret, ndmin=1),
                            'Sharpe Ratio': np.array(sharpe_ratio, ndmin=1),
                            'Max Drawdown': np.array(max_dd, ndmin=1)})
    metrics = metrics.apply(pd.to_numeric, errors='coerce')

    # Statistical Summary
    stats_summary = metrics.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    stats_summary['skew'] = metrics.skew()
    stats_summary['kurtosis'] = metrics.kurtosis()
    order = ['count', 'mean', 'std', 'min', 'max', '1%', '5%', '25%', '50%', '75%', '95%', '99%', 'skew', 'kurtosis']
    stats_summary = stats_summary[order]

    # Print Detailed Stats
    if show_stats:
        report_type = "Monte Carlo" if type(total_ret) == pd.Series else "Strategy Backtest"
        print("="*40)
        print(f"STRATEGY PERFORMANCE REPORT ({report_type})")
        print("="*40)
        print(f"Mean Total Return:      {stats_summary.loc['Total Return', 'mean']:.2f}%")
        print(f"Mean Annualized Sharpe: {stats_summary.loc['Sharpe Ratio', 'mean']:.2f}")
        print(f"Mean Max Drawdown:      {stats_summary.loc['Max Drawdown', 'mean']:.2f}%")
        print("="*40)
        print()

    return metrics, stats_summary