import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def monte_carlo(rets, num_sims=1000, replacement=True, plot=False):
    # Actual cumulative returns
    actual_cum_rets = (1 + rets).cumprod()
    num_days = len(rets)
    rets_index = rets.index

    # Vectorized bootstrap sampling
    if replacement:
        # Standard Bootstrap: Sample with replacement
        rets = np.random.choice(rets, size=(num_days, num_sims), replace=True)
    else:
        # Permutation: Shuffle the existing returns for each simulation
        rets = np.array([np.random.permutation(rets.values) for _ in range(num_sims)]).T

    # Vectorized cumulative returns
    cum_rets = np.cumprod(1 + rets, axis=0)

    # Convert to DataFrames for easier handling
    rets_df = pd.DataFrame(rets, index=rets_index, columns=range(num_sims))
    cum_rets_df = pd.DataFrame(cum_rets, index=rets_index, columns=range(num_sims))

    # Total returns for each simulation
    final_rets = cum_rets_df.iloc[-1] - 1

    # Plot the actual backtest cumulative returns to compare to simulations
    if plot:
        plt.figure(figsize=(12, 6))
        # Plot a subset of paths for performance and visibility
        plot_subset = cum_rets_df.iloc[:, :min(num_sims, 100)]
        plt.plot(plot_subset, color='gray', alpha=0.1)
        
        # Highlight the actual backtest
        plt.plot(actual_cum_rets, color='red', label='Actual Backtest')
        
        # # Add a "Median Path" for better statistical intuition
        # plt.plot(cum_rets_df.median(axis=1), color='blue', linestyle='--', label='Median Path')
        
        if replacement:
            plt.title(f"Monte Carlo: Bootstrap with Replacement ({num_sims} simulations)")
        else:
            plt.title(f"Monte Carlo: Bootstrap without Replacement ({num_sims} simulations)")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.show()

    return rets_df, cum_rets_df, final_rets