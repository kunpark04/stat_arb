import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

TRADING_DAYS = 252


def monte_carlo(rets, num_sims=1000, replacement=True, plot=False, show_analysis=True):
    """
    Bootstrap Monte Carlo simulation on strategy returns.

    Parameters
    ----------
    replacement    : True = standard bootstrap (with replacement).
                     False = permutation (shuffle order, no replacement).
    show_analysis  : Print and plot a detailed statistical breakdown of
                     the simulation distribution.
    """
    actual_cum_rets = (1 + rets).cumprod()
    num_days = len(rets)
    rets_index = rets.index

    # --- Sampling ---
    if replacement:
        sim_rets = np.random.choice(rets, size=(num_days, num_sims), replace=True)
    else:
        sim_rets = np.array([np.random.permutation(rets.values) for _ in range(num_sims)]).T

    cum_rets = np.cumprod(1 + sim_rets, axis=0)

    rets_df     = pd.DataFrame(sim_rets, index=rets_index, columns=range(num_sims))
    cum_rets_df = pd.DataFrame(cum_rets, index=rets_index, columns=range(num_sims))
    final_rets  = cum_rets_df.iloc[-1] - 1

    # --- Equity path plot ---
    if plot:
        plt.figure(figsize=(12, 6))
        plot_subset = cum_rets_df.iloc[:, :min(num_sims, 100)]
        plt.plot(plot_subset, color='gray', alpha=0.1)
        plt.plot(actual_cum_rets, color='red', linewidth=2, label='Actual Backtest')
        plt.plot(cum_rets_df.median(axis=1), color='blue', linestyle='--',
                 linewidth=1.5, label='Median Path')
        title = (f"Monte Carlo: Bootstrap {'with' if replacement else 'without'} "
                 f"Replacement ({num_sims:,} simulations)")
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --- Statistical analysis ---
    if show_analysis:
        _analyze_simulations(rets_df, cum_rets_df, final_rets, actual_cum_rets, rets, replacement)

    return rets_df, cum_rets_df, final_rets


# ─────────────────────────────────────────────────────────────────────────────


def _compute_sim_metrics(rets_df, cum_rets_df):
    """Compute Sharpe, annualized return, and max drawdown for each simulation."""
    n_days = len(rets_df)

    ann_rets   = (cum_rets_df.iloc[-1] ** (TRADING_DAYS / n_days) - 1) * 100
    sharpes    = (rets_df.mean() / rets_df.std()) * np.sqrt(TRADING_DAYS)
    running_max = cum_rets_df.cummax()
    drawdowns   = ((cum_rets_df / running_max) - 1).min() * 100

    # Sortino
    downside = rets_df.copy()
    downside[downside > 0] = 0
    sortinos = (rets_df.mean() / (downside.std() + 1e-10)) * np.sqrt(TRADING_DAYS)

    # Calmar
    calmars = ann_rets / (drawdowns.abs() + 1e-10)

    return pd.DataFrame({
        'Annualized Return (%)': ann_rets,
        'Sharpe Ratio':          sharpes,
        'Sortino Ratio':         sortinos,
        'Max Drawdown (%)':      drawdowns,
        'Calmar Ratio':          calmars,
    })


def _var_cvar(series, alpha=0.05):
    """Compute VaR and CVaR at confidence level alpha (left tail)."""
    var  = series.quantile(alpha)
    cvar = series[series <= var].mean()
    return var, cvar


def _analyze_simulations(rets_df, cum_rets_df, final_rets, actual_cum_rets, actual_rets,
                          replacement):
    """Print detailed statistics and plot metric distributions."""
    sim_metrics = _compute_sim_metrics(rets_df, cum_rets_df)
    num_sims    = len(rets_df.columns)
    n_days      = len(rets_df)

    # Actual backtest metrics
    actual_ann_ret  = (actual_cum_rets.iloc[-1] ** (TRADING_DAYS / n_days) - 1) * 100
    actual_sharpe   = (actual_rets.mean() / actual_rets.std()) * np.sqrt(TRADING_DAYS)
    actual_dd       = ((actual_cum_rets / actual_cum_rets.cummax()) - 1).min() * 100
    actual_final    = actual_cum_rets.iloc[-1] - 1

    sim_type = "Bootstrap (w/ replacement)" if replacement else "Permutation (w/o replacement)"

    print("=" * 60)
    print(f"  MONTE CARLO ANALYSIS — {sim_type}")
    print(f"  {num_sims:,} simulations | {n_days} trading days")
    print("=" * 60)

    for col in sim_metrics.columns:
        series     = sim_metrics[col]
        var, cvar  = _var_cvar(series)
        pct_above  = (series > series.median()).mean() * 100  # symmetry check

        # Map column name to actual metric
        actual_vals = {
            'Annualized Return (%)': actual_ann_ret,
            'Sharpe Ratio':          actual_sharpe,
            'Max Drawdown (%)':      actual_dd,
            'Sortino Ratio':         (actual_rets.mean() /
                                      (actual_rets[actual_rets < 0].std(ddof=1) + 1e-10)) * np.sqrt(TRADING_DAYS),
            'Calmar Ratio':          actual_ann_ret / (abs(actual_dd) + 1e-10),
        }
        actual_val = actual_vals.get(col, float('nan'))
        pct_rank   = (series < actual_val).mean() * 100

        print(f"\n  {col}")
        print(f"    Mean:      {series.mean():>8.3f}   |  Actual:  {actual_val:>8.3f}  "
              f"(pctile: {pct_rank:.0f}%)")
        print(f"    Std:       {series.std():>8.3f}   |  "
              f"VaR(5%): {var:>8.3f}   CVaR(5%): {cvar:>8.3f}")
        print(f"    Min:       {series.min():>8.3f}   |  "
              f"p5:  {series.quantile(0.05):>8.3f}   p95: {series.quantile(0.95):>8.3f}")
        print(f"    Skew:      {series.skew():>8.3f}   |  "
              f"Kurt: {series.kurtosis():>8.3f}")

    print()

    # Distribution plots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Monte Carlo Distribution Analysis — {sim_type}\n"
                 f"({num_sims:,} simulations)", fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    plot_items = [
        ('Annualized Return (%)',  actual_ann_ret, 'steelblue'),
        ('Sharpe Ratio',           actual_sharpe,  'seagreen'),
        ('Max Drawdown (%)',        actual_dd,      'tomato'),
        ('Sortino Ratio',          actual_vals['Sortino Ratio'], 'darkorchid'),
        ('Calmar Ratio',           actual_ann_ret / (abs(actual_dd) + 1e-10), 'goldenrod'),
    ]

    # Final returns distribution (6th panel)
    all_items = plot_items + [('Final Total Return (%)', (actual_final) * 100, 'coral')]
    sim_final_pct = (final_rets) * 100

    for idx, (label, actual_val, color) in enumerate(all_items):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])

        if label == 'Final Total Return (%)':
            data = sim_final_pct
        else:
            data = sim_metrics[label] if label in sim_metrics.columns else sim_final_pct

        _bins = 60 if data.std() > 1e-10 else 1
        ax.hist(data, bins=_bins, color=color, alpha=0.7, edgecolor='none')
        ax.axvline(actual_val, color='black', linewidth=2,
                   linestyle='--', label=f'Actual: {actual_val:.2f}')
        ax.axvline(data.quantile(0.05), color='red', linewidth=1,
                   linestyle=':', label=f'p5: {data.quantile(0.05):.2f}')
        ax.axvline(data.quantile(0.95), color='green', linewidth=1,
                   linestyle=':', label=f'p95: {data.quantile(0.95):.2f}')

        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
