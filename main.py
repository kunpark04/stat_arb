from price_data.fetch import fetch_data
import strategy.strategy as strat
import visualization.plots as vis
from metrics.monte_carlo import monte_carlo
from metrics.param_sweep import position_sweep, plot_metric_heatmaps

# ─────────────────────────────────────────────────────────────────────────────
# 1. FETCH DATA
# ─────────────────────────────────────────────────────────────────────────────
rets, prices, high, low, volume = fetch_data(download=False,
                                             start_date=None,
                                             interval=None,
                                             univ=None,
                                             file_name='price_volume_data.pkl')

# ─────────────────────────────────────────────────────────────────────────────
# 1.5. REGIME DETECTION
# ─────────────────────────────────────────────────────────────────────────────
from models.regime import detect_regime
print('DETECTING REGIME...')
regime_series, composite, vol_sc, vr_sc, corr_sc = detect_regime(rets, return_signals=True)
print(f"  Regime distribution:\n{regime_series.value_counts().sort_index()}")
print(f"  NaN warmup days: {regime_series.isna().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. GENERATE SIGNALS
#    - variance_threshold=0.55 → adaptive PCA factor count (≥55% variance)
#    - num_factors=15           → upper cap on factor count
# ─────────────────────────────────────────────────────────────────────────────
print('GENERATING SIGNALS...')
S_score, Kappa, Sigma_eq, Beta, Factor_rets = strat.signal_gen(
    rets=rets,
    start_trading_date='2021-01-28',
    num_factors=15,
    variance_threshold=0.55,
    window_PCA=252,
    window_OU=60,
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. GENERATE POSITIONS
#    - entry_threshold=2.0, exit_threshold=0.5 (Avellaneda & Lee defaults)
#    - max_position=0.15     → cap each stock at 15% of portfolio
#    - cooldown_days=5       → 5-day block after a stop-loss exit
#    - kappa_percentile=0.5  → only trade top-50% by mean-reversion speed
# ─────────────────────────────────────────────────────────────────────────────
print('\nGENERATING POSITIONS...')
state_history, leveraged_weights = strat.gen_positions(
    S_score=S_score,
    Kappa=Kappa,
    Sigma_eq=Sigma_eq,
    leverage=1.0,
    entry_threshold=2.0,
    exit_threshold=0.5,
    stop_loss=3.0,
    kappa_min=5.0,
    kappa_percentile=0.5,
    max_position=0.15,
    cooldown_days=5,
    regime_series=regime_series,
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. CALCULATE PnL
#    - vol_target=0.10 → scale leverage daily to target 10% annualized vol
# ─────────────────────────────────────────────────────────────────────────────
print('\nCALCULATING STRATEGY RETURNS...')
strat_rets, strat_cum_rets = strat.calculate_pnl(
    rets=rets,
    Beta=Beta,
    Factor_rets=Factor_rets,
    weight_history=leveraged_weights,
    commission=0.0005,
    vol_target=None,
    regime_series=regime_series,
)

# ─────────────────────────────────────────────────────────────────────────────
# 4.5. TRENDING STRATEGY (standalone, all regimes)
# ─────────────────────────────────────────────────────────────────────────────
import strategy.trending as trend

print('\nGENERATING TREND SIGNALS...')
EMA_signal, ADX, Volume_ratio = trend.signal_gen_trend(
    prices=prices,
    high=high,
    low=low,
    rets=rets,
    volume=volume,
    start_trading_date='2021-01-28',
    fast_span=9,
    slow_span=20,
)

print('\nGENERATING TREND POSITIONS (standalone)...')
state_trend, weights_trend = trend.gen_positions_trend(
    EMA_signal, ADX, Volume_ratio,
    leverage=1.0,
    adx_min=20.0,
    vol_ratio_min=1.2,
    max_position=0.10,
    active_regimes=None,  # all regimes
)

print('\nCALCULATING TREND RETURNS...')
trend_rets, trend_cum_rets = trend.calculate_pnl_trend(
    rets=rets,
    weight_history=weights_trend,
    commission=0.0005,
    vol_target=0.10,
)

# ─────────────────────────────────────────────────────────────────────────────
# 4.75. HYBRID STRATEGY (stat-arb regimes 0+1, trending regime 2)
# ─────────────────────────────────────────────────────────────────────────────
print('\nGENERATING TREND POSITIONS (hybrid — regime 2 only)...')
state_hybrid, weights_hybrid = trend.gen_positions_trend(
    EMA_signal, ADX, Volume_ratio,
    leverage=1.0,
    adx_min=20.0,
    vol_ratio_min=1.2,
    max_position=0.10,
    regime_series=regime_series,
    active_regimes=(2,),  # only regime 2
)

hybrid_trend_rets, _ = trend.calculate_pnl_trend(
    rets=rets,
    weight_history=weights_hybrid,
    commission=0.0005,
    vol_target=0.10,
    regime_series=regime_series,
)

combined_rets     = strat_rets + hybrid_trend_rets
combined_cum_rets = (1 + combined_rets).cumprod()

# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALIZE — equity curve with regime shading + drawdown + metrics table
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== STAT-ARB STRATEGY ===')
vis.plot_equity_drawdown(strat_rets, strat_cum_rets, regime_series=regime_series)
vis.display_metrics_table(strat_rets, strat_cum_rets, risk_free_rate=0.04)

print('\n=== TRENDING STRATEGY (all regimes) ===')
vis.plot_equity_drawdown(trend_rets, trend_cum_rets, regime_series=regime_series)
vis.display_metrics_table(trend_rets, trend_cum_rets, risk_free_rate=0.04)

print('\n=== HYBRID STRATEGY (stat-arb R0/R1 + trending R2) ===')
vis.plot_equity_drawdown(combined_rets, combined_cum_rets, regime_series=regime_series)
vis.display_metrics_table(combined_rets, combined_cum_rets, risk_free_rate=0.04)

# ─────────────────────────────────────────────────────────────────────────────
# 6. MONTE CARLO — Bootstrap with replacement
#    show_analysis=True prints detailed stats + distribution plots
# ─────────────────────────────────────────────────────────────────────────────
print('\nMONTE CARLO (bootstrap with replacement)...')
rets_df1, cum_rets_df1, final_rets1 = monte_carlo(
    rets=strat_rets,
    num_sims=1000,
    replacement=True,
    plot=True,
    show_analysis=True,
)

print('\nMONTE CARLO (permutation)...')
rets_df2, cum_rets_df2, final_rets2 = monte_carlo(
    rets=strat_rets,
    num_sims=1000,
    replacement=False,
    plot=True,
    show_analysis=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. PARAMETER SWEEP — OU filtering params (kappa_min × kappa_percentile)
#    Sweeps the two key OU-based position filters; value = Sharpe ratio.
#    Also plots a 2×2 grid with Sharpe, Ann Return, Max DD, Calmar.
# ─────────────────────────────────────────────────────────────────────────────
print('\nPARAMETER SWEEP (kappa_min × kappa_percentile)...')
sharpe_df, sweep_results = position_sweep(
    S_score=S_score,
    Kappa=Kappa,
    Sigma_eq=Sigma_eq,
    rets=rets,
    Beta=Beta,
    Factor_rets=Factor_rets,
    param1_name='kappa_min',
    param1_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0],
    param2_name='kappa_percentile',
    param2_values=[None, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75],
    fixed_kwargs={
        'leverage':        1.0,
        'entry_threshold': 2.0,
        'exit_threshold':  0.5,
        'stop_loss':       3.0,
        'max_position':    0.15,
        'cooldown_days':   5,
    },
    commission=0.0005,
    risk_free_rate=0.04,
    plot=True,
    annot=True,
)

# Full 2×2 metric heatmap grid
plot_metric_heatmaps(sweep_results, 'kappa_min', 'kappa_percentile')
