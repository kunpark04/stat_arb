import pandas as pd
import matplotlib.pyplot as plt

from price_data.data import fetch_data
import processing.strategy as strat
import data_visualization.plots as vis
from metrics.performance import performance_analysis
from metrics.monte_carlo import monte_carlo

#=== Fetch Data ===#
rets = fetch_data(download=False,
                    start_date=None,
                    interval=None,
                    univ=None,
                    file_name='price_volume_data.pkl')

#=== Backtest Strategy ===#
print('GENERATING SIGNALS...')
S_score, Kappa, Sigma_eq, Beta, Factor_rets = strat.signal_gen(rets=rets,
                                                                start_trading_date='2021-01-28',
                                                                num_factors=5,
                                                                window_PCA=252,
                                                                window_OU=60)

print('\nGENERATING POSITIONS...')
state_history, leveraged_weights = strat.gen_positions(S_score=S_score,
                                                        Kappa=Kappa,
                                                        Sigma_eq=Sigma_eq,
                                                        leverage=1.0)
print('\nCALCULATING STRATEGY RETURNS...')
strat_rets, strat_cum_rets = strat.calculate_pnl(rets=rets,
                                                    Beta=Beta,
                                                    Factor_rets=Factor_rets,
                                                    weight_history=leveraged_weights,
                                                    commission=0.0005)

#=== Performance Metrics ===#
metrics, stats_summary = performance_analysis(strat_cum_rets, strat_rets, show_stats=True)

#=== Visualize Performance Metrics ===#
vis.plot_equity(strat_rets, strat_cum_rets)
vis.plot_drawdown(strat_cum_rets)

#=== Monte Carlo (Bootstrap) ===#
rets_df1, cum_rets_df1, final_rets1 = monte_carlo(rets=strat_rets,
                                                  num_sims=1000,
                                                  replacement=False,
                                                  plot=True)
metrics1, stats_summary1 = performance_analysis(cum_rets=cum_rets_df1,
                                                rets=rets_df1,
                                                show_stats=False)

rets_df2, cum_rets_df2, final_rets2 = monte_carlo(rets=strat_rets,
                                                  num_sims=1000,
                                                  replacement=True,
                                                  plot=True)
metrics2, stats_summary2 = performance_analysis(cum_rets=cum_rets_df2,
                                                rets=rets_df2,
                                                show_stats=False)
# print(stats_summary1.to_markdown())
# print(stats_summary2.to_markdown())

# SHOW THE STDEV OF EACH CORE METRIC
# SAVE IMPORTANT DIAGNOSTIC DATA AS EXCEL FILES in "backtest_data"
# BEGIN PARAMETER TUNING TO OPTIMIZE P&L