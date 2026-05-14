"""
Parameter sweep for gen_positions hyperparameters.

Reuses a single signal_gen run (S_score, Kappa, Sigma_eq) and sweeps
two gen_positions parameters (default: entry_threshold × exit_threshold).
Returns a Sharpe ratio heatmap.

Sweeping signal_gen parameters (window_OU, num_factors, etc.) requires
re-running the full pipeline per combination, which is much slower. A
separate full_sweep() function handles that use case.
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from strategy.strategy import gen_positions, calculate_pnl

TRADING_DAYS = 252


def _label(v):
    """None → 'None' for use as a DataFrame index/column label."""
    return 'None' if v is None else v


def _value(v):
    """'None' → None when passing back to gen_positions."""
    return None if v == 'None' else v


def _sharpe(rets, rf_daily=0.0):
    excess = rets.mean() - rf_daily
    return (excess / rets.std()) * np.sqrt(TRADING_DAYS) if rets.std() > 0 else np.nan


# ─────────────────────────────────────────────────────────────────────────────


def position_sweep(S_score, Kappa, Sigma_eq, rets, Beta, Factor_rets,
                   param1_name='entry_threshold', param1_values=None,
                   param2_name='exit_threshold',  param2_values=None,
                   fixed_kwargs=None, commission=0.0005,
                   risk_free_rate=0.0, plot=True, annot=True):
    """
    Sweep two gen_positions parameters and return a Sharpe ratio heatmap.

    Signals (S_score, Kappa, Sigma_eq) are computed once and reused for every
    combination, so only gen_positions + calculate_pnl re-runs each iteration.

    Parameters
    ----------
    param1_name   : Name of the first parameter (x-axis of heatmap).
    param1_values : List of values for param1.
    param2_name   : Name of the second parameter (y-axis of heatmap).
    param2_values : List of values for param2.
    fixed_kwargs  : Dict of additional fixed kwargs passed to gen_positions.
    commission    : One-way transaction cost per unit of weight traded.
    risk_free_rate: Annualized risk-free rate for Sharpe calculation.
    plot          : Show the heatmap.
    annot         : Annotate cells with Sharpe values.

    Returns
    -------
    sharpe_df : DataFrame with param1 as columns and param2 as index.
    results   : Full dict of (ann_ret, sharpe, sortino, max_dd, calmar) per combo.
    """
    if param1_values is None:
        param1_values = [1.5, 2.0, 2.5, 3.0, 3.5]
    if param2_values is None:
        param2_values = [0.25, 0.5, 0.75, 1.0, 1.25]
    if fixed_kwargs is None:
        fixed_kwargs = {}

    rf_daily = risk_free_rate / TRADING_DAYS

    # Validate: entry > exit to avoid illogical combos
    skip_invalid = (param1_name == 'entry_threshold' and param2_name == 'exit_threshold')

    p1_labels = [_label(v) for v in param1_values]
    p2_labels = [_label(v) for v in param2_values]

    sharpe_matrix  = pd.DataFrame(index=p2_labels, columns=p1_labels, dtype=float)
    ann_ret_matrix = pd.DataFrame(index=p2_labels, columns=p1_labels, dtype=float)
    max_dd_matrix  = pd.DataFrame(index=p2_labels, columns=p1_labels, dtype=float)
    calmar_matrix  = pd.DataFrame(index=p2_labels, columns=p1_labels, dtype=float)

    n_total = len(param1_values) * len(param2_values)
    print(f"Running {n_total} parameter combinations "
          f"({param1_name} × {param2_name})...")

    for i, (v1, v2) in enumerate(itertools.product(param1_values, param2_values)):
        # Skip entry <= exit (nonsensical for entry/exit threshold sweep)
        l1, l2 = _label(v1), _label(v2)

        if skip_invalid and v2 >= v1:
            sharpe_matrix.loc[l2, l1] = np.nan
            continue

        kwargs = {param1_name: _value(v1), param2_name: _value(v2), **fixed_kwargs}

        try:
            _, leveraged_weights = gen_positions(
                S_score=S_score, Kappa=Kappa, Sigma_eq=Sigma_eq, **kwargs
            )
            daily_rets, daily_cum = calculate_pnl(
                rets=rets, Beta=Beta, Factor_rets=Factor_rets,
                weight_history=leveraged_weights, commission=commission
            )
        except Exception as e:
            print(f"  [{v1}, {v2}] failed: {e}")
            sharpe_matrix.loc[l2, l1] = np.nan
            continue

        n_days = len(daily_rets)
        sharpe_matrix.loc[l2, l1]  = _sharpe(daily_rets, rf_daily)
        ann_ret = (daily_cum.iloc[-1] ** (TRADING_DAYS / n_days) - 1) * 100
        ann_ret_matrix.loc[l2, l1] = ann_ret
        dd = ((daily_cum / daily_cum.cummax()) - 1).min() * 100
        max_dd_matrix.loc[l2, l1]  = dd
        calmar_matrix.loc[l2, l1]  = ann_ret / (abs(dd) + 1e-10)

        if (i + 1) % max(1, n_total // 10) == 0 or (i + 1) == n_total:
            print(f"  Progress: {i+1}/{n_total}")

    if plot:
        _plot_heatmap(sharpe_matrix, param1_name, param2_name,
                      title='Sharpe Ratio Heatmap', cmap='RdYlGn',
                      center=0.0, annot=annot)

    results = {
        'Sharpe':          sharpe_matrix,
        'Annualized Return (%)': ann_ret_matrix,
        'Max Drawdown (%)': max_dd_matrix,
        'Calmar Ratio':    calmar_matrix,
    }

    # Print best combo
    best_v2, best_v1 = np.unravel_index(
        np.nanargmax(sharpe_matrix.values.astype(float)),
        sharpe_matrix.shape
    )
    print(f"\nBest Sharpe: {sharpe_matrix.values.astype(float)[best_v2, best_v1]:.3f} "
          f"at {param1_name}={sharpe_matrix.columns[best_v1]}, "
          f"{param2_name}={sharpe_matrix.index[best_v2]}")

    return sharpe_matrix, results


# ─────────────────────────────────────────────────────────────────────────────


def _plot_heatmap(data, xlabel, ylabel, title='Heatmap',
                  cmap='RdYlGn', center=None, annot=True, fmt='.2f'):
    """Render a labeled seaborn heatmap."""
    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        data.astype(float),
        ax=ax,
        cmap=cmap,
        center=center,
        annot=annot,
        fmt=fmt,
        linewidths=0.5,
        linecolor='#cccccc',
        cbar_kws={'label': title},
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticklabels([f'{v}' for v in data.columns], rotation=45, ha='right')
    ax.set_yticklabels([f'{v}' for v in data.index], rotation=0)
    plt.tight_layout()
    plt.show()


def plot_metric_heatmaps(results, param1_name, param2_name):
    """
    Plot all four metric heatmaps (Sharpe, Ann Return, Max DD, Calmar) in a 2×2 grid.
    """
    metrics = [
        ('Sharpe',          'RdYlGn', 0.0),
        ('Annualized Return (%)', 'RdYlGn', None),
        ('Max Drawdown (%)', 'RdYlBu_r', None),
        ('Calmar Ratio',    'RdYlGn', 0.0),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Parameter Sweep: {param1_name} × {param2_name}',
                 fontsize=14, fontweight='bold')

    for ax, (key, cmap, center) in zip(axes.flat, metrics):
        data = results[key].astype(float)
        sns.heatmap(data, ax=ax, cmap=cmap, center=center,
                    annot=True, fmt='.2f', linewidths=0.5,
                    linecolor='#cccccc', cbar_kws={'label': key})
        ax.set_title(key, fontsize=11, fontweight='bold')
        ax.set_xlabel(param1_name, fontsize=9)
        ax.set_ylabel(param2_name, fontsize=9)
        ax.set_xticklabels([f'{v}' for v in data.columns], rotation=45, ha='right')
        ax.set_yticklabels([f'{v}' for v in data.index], rotation=0)

    plt.tight_layout()
    plt.show()
