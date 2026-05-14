import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────────────────────
# Regime detection helpers
# ─────────────────────────────────────────────────────────────────────────────

_REGIME_COLORS = {0: 'seagreen', 1: 'gold', 2: 'crimson'}
_REGIME_ALPHAS = {0: 0.10,       1: 0.06,  2: 0.12}
_REGIME_LABELS = {0: 'Regime 0 (Favorable)', 1: 'Regime 1 (Neutral)', 2: 'Regime 2 (Trending)'}


def _add_regime_shading(ax, date_index, regime_series):
    """Shade contiguous regime blocks on a matplotlib Axes."""
    import matplotlib.patches as mpatches

    ra = regime_series.reindex(date_index).ffill().dropna().astype(int)
    if ra.empty:
        return

    patches_added = set()
    start_date = ra.index[0]
    cur_regime = ra.iloc[0]

    def _shade(start, end, regime):
        color = _REGIME_COLORS[regime]
        alpha = _REGIME_ALPHAS[regime]
        ax.axvspan(start, end, color=color, alpha=alpha, linewidth=0)
        if regime not in patches_added:
            patches_added.add(regime)

    for date, regime in ra.items():
        if regime != cur_regime:
            _shade(start_date, date, cur_regime)
            start_date = date
            cur_regime = regime
    _shade(start_date, ra.index[-1], cur_regime)

    legend_patches = [
        mpatches.Patch(color=_REGIME_COLORS[r], alpha=max(_REGIME_ALPHAS[r] * 4, 0.4),
                       label=_REGIME_LABELS[r])
        for r in sorted(patches_added)
    ]
    ax.legend(handles=legend_patches, fontsize=7, loc='upper left')


# ─────────────────────────────────────────────────────────────────────────────
# Equity + Drawdown (two-panel)
# ─────────────────────────────────────────────────────────────────────────────

def plot_equity_drawdown(rets, cum_rets, regime_series=None,
                         title='Strategy Performance'):
    """
    Two-panel figure: equity curve (top) and drawdown (bottom).
    Regime shading is overlaid on the equity panel if regime_series is provided.
    """
    running_max = cum_rets.cummax()
    drawdown    = (cum_rets / running_max) - 1
    max_dd      = drawdown.min() * 100

    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(16, 8),
        gridspec_kw={'height_ratios': [2, 1]},
        sharex=True,
    )
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ── Equity curve ─────────────────────────────────────────────────────────
    ax_eq.plot(cum_rets.index, cum_rets, color='steelblue', linewidth=1.8, zorder=3)
    ax_eq.fill_between(cum_rets.index, 1, cum_rets,
                       where=(cum_rets >= 1), color='steelblue', alpha=0.08, zorder=2)
    ax_eq.fill_between(cum_rets.index, 1, cum_rets,
                       where=(cum_rets < 1),  color='crimson',   alpha=0.08, zorder=2)
    ax_eq.axhline(1, color='black', linewidth=0.8, linestyle='--', zorder=2)
    ax_eq.set_ylabel('Cumulative Return')
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.2f}x'))
    ax_eq.grid(True, alpha=0.25, zorder=1)
    ax_eq.set_title('Equity Curve', fontweight='bold')

    if regime_series is not None:
        _add_regime_shading(ax_eq, cum_rets.index, regime_series)

    # ── Drawdown ─────────────────────────────────────────────────────────────
    ax_dd.plot(drawdown.index, drawdown * 100, color='crimson', linewidth=1.2)
    ax_dd.fill_between(drawdown.index, drawdown * 100, 0, color='crimson', alpha=0.15)
    ax_dd.axhline(max_dd, color='darkred', linewidth=1, linestyle=':', alpha=0.7,
                  label=f'Max DD: {max_dd:.2f}%')
    ax_dd.set_ylabel('Drawdown (%)')
    ax_dd.set_title('Drawdown', fontweight='bold')
    ax_dd.legend(fontsize=9)
    ax_dd.grid(True, alpha=0.25)

    fig.tight_layout()
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Performance metrics table
# ─────────────────────────────────────────────────────────────────────────────

def display_metrics_table(rets, cum_rets, risk_free_rate=0.04) -> pd.DataFrame:
    """
    Compute key performance metrics, print a formatted table, and return the DataFrame.
    """
    rf_daily  = risk_free_rate / TRADING_DAYS
    n_days    = len(rets)

    total_ret  = (cum_rets.iloc[-1] - 1) * 100
    ann_ret    = (cum_rets.iloc[-1] ** (TRADING_DAYS / n_days) - 1) * 100
    vol        = rets.std() * np.sqrt(TRADING_DAYS) * 100
    sharpe     = ((rets.mean() - rf_daily) / rets.std()) * np.sqrt(TRADING_DAYS)
    running_max = cum_rets.cummax()
    max_dd     = ((cum_rets / running_max) - 1).min() * 100
    win_rate   = (rets > 0).mean() * 100

    metrics = pd.DataFrame(
        {
            'Value': [
                f'{total_ret:+.2f}%',
                f'{ann_ret:+.2f}%',
                f'{vol:.2f}%',
                f'{sharpe:.3f}',
                f'{max_dd:.2f}%',
                f'{win_rate:.1f}%',
            ]
        },
        index=[
            'Total Return',
            'Annualized Return',
            'Annualized Vol',
            'Sharpe Ratio',
            'Max Drawdown',
            'Win Rate',
        ],
    )
    metrics.index.name = 'Metric'

    print()
    print('=' * 36)
    print('  PERFORMANCE METRICS')
    print('=' * 36)
    print(metrics.to_string())
    print('=' * 36)
    print()

    return metrics
