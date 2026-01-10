import pandas as pd
from processing.PCA import PCA_process
from processing.OLS import OLS_process
from processing.OU import OU_process


def signal_gen(rets, start_trading_date, num_factors, window_PCA, window_OU):
    
    # Select trading dates
    days = 0
    Dates = rets.index[rets.index >= start_trading_date]

    # Initialize DataFrames to store OU metrics
    Factor_rets = pd.DataFrame(index=Dates, columns=range(1, num_factors + 1))
    S_score = pd.DataFrame(index=Dates, columns=rets.columns)
    Kappa = pd.DataFrame(index=Dates, columns=rets.columns)
    Sigma_eq = pd.DataFrame(index=Dates, columns=rets.columns)
    Beta = {}

    for date in Dates:

        # Select the relevant window of returns
        # rets_window = rets.loc[:date].tail(window_PCA)
        rets_window = rets.loc[:date].iloc[:-1].tail(window_PCA) # no lookahead bias (exclude current day)

        # Calculate PCA weights every 30 days
        if days % 30 == 0:
            # Filter for valid tickers
            nan_count = rets_window.isna().sum()
            mask = nan_count == 0
            valid_columns = rets_window.columns[mask]
            rets_window = rets_window[valid_columns].fillna(0)

            # Calculate PCA weights
            eigen_weights, variance_pct = PCA_process(rets_window,
                                                      num_factors)
        else:
            rets_window = rets_window[valid_columns].fillna(0)

        # Calculate OU parameters every day
        eigenportfolio_rets = rets_window @ eigen_weights

        # Save factor returns for current day
        current_ret = rets.loc[date, valid_columns].fillna(0)
        current_factor_rets = current_ret @ eigen_weights
        Factor_rets.loc[date] = current_factor_rets

        cum_residuals, beta = OLS_process(rets_window,
                                            eigenportfolio_rets,
                                            window_OU)
        s_score, kappa, sigma_eq = OU_process(cum_residuals)

        # Store OU metrics
        S_score.loc[date, valid_columns] = s_score
        Kappa.loc[date, valid_columns] = kappa
        Sigma_eq.loc[date, valid_columns] = sigma_eq
        Beta[date] = beta

        # Update time counter
        days += 1
        if days % 300 == 0:
            print(f'Processed {days} days')
        elif date == Dates[-1]:
            print(f'Processed {days} days')
            print('Processing complete.')
    
    return S_score, Kappa, Sigma_eq, Beta, Factor_rets

#=========================================================================#

def gen_positions(S_score, Kappa, Sigma_eq, leverage):
    days = 0
    Dates = S_score.index
    current_state = pd.Series(0, index=S_score.columns)
    
    # Initialize history storage
    state_history = pd.DataFrame(index=S_score.index, columns=S_score.columns)
    weight_history = pd.DataFrame(index=S_score.index, columns=S_score.columns)

    # Parameters
    ENTRY_LONG = -2.0
    ENTRY_SHORT = 2.0
    EXIT_SHORT = 0.5
    EXIT_LONG = -0.5   
    STOP_LOSS = 3.0 
    MIN_KAPPA = 5.0    # Filter for fast mean reversion

    for date in Dates:
        s_score = S_score.loc[date]
        kappa = Kappa.loc[date]
        sigma_eq = Sigma_eq.loc[date]
        
        # Exit strategy
        exit_long = (current_state == 1) & ((s_score > EXIT_LONG) | (s_score < -STOP_LOSS))
        current_state[exit_long] = 0
        
        exit_short = (current_state == -1) & ((s_score < EXIT_SHORT) | (s_score > STOP_LOSS))
        current_state[exit_short] = 0

        # Entry strategy
        # Handle potential NaNs in Kappa/Sigma to prevent errors
        valid_data = (kappa.notna()) & (sigma_eq > 0) & (kappa > MIN_KAPPA)
        
        # Enter long
        enter_long = (current_state == 0) & (s_score < ENTRY_LONG) & valid_data
        current_state[enter_long] = 1
        
        # Enter short
        enter_short = (current_state == 0) & (s_score > ENTRY_SHORT) & valid_data
        current_state[enter_short] = -1

        # Portfolio weighting
        # Formula: Weight = State * (1 / Sigma)
        # The more volatile, the more risky, the smaller position size
        raw_weights = current_state / (sigma_eq + 1e-6)
        raw_weights = raw_weights.clip(-2.0, 2.0) 

        # Full state and weight history
        state_history.loc[date] = current_state
        weight_history.loc[date] = raw_weights

        # Update time counter
        days += 1
        if days % 300 == 0:
            print(f'Processed {days} days')
        elif date == Dates[-1]:
            print(f'Processed {days} days')
            print('Processing complete.')

    # Normalize weights
    weight_history = weight_history.astype(float)
    gross_leverage = weight_history.abs().sum(axis=1)
    normalized_weights = weight_history.div(gross_leverage, axis=0).fillna(0)
    leveraged_weights = normalized_weights * leverage

    return state_history, leveraged_weights

#=========================================================================#

def calculate_pnl(rets, Beta, Factor_rets, weight_history, commission=0.0005):
    
    Dates = weight_history.index
    effective_weights = weight_history.shift(1).fillna(0)

    # Align returns with weights
    aligned_rets = rets.loc[Dates]

    # Calculate strategy returns
    strat_rets = effective_weights * aligned_rets
    daily_strat_rets = strat_rets.sum(axis=1)  # Used out-of-sample, so no need to shift?

    # Calculate beta hedge returns
    hedge_rets = pd.DataFrame(index=Dates,
                              columns=weight_history.columns,
                              dtype=float)
    for date in Dates:
        weights = effective_weights.loc[date]
        factor_rets = Factor_rets.loc[date]
        beta = Beta[date].iloc[1:, :]
        hedge_rets.loc[date] = (beta.T @ factor_rets) * weights
    daily_hedge_rets = hedge_rets.sum(axis=1)
    
    # Transaction costs
    weight_changes = weight_history.diff().abs()
    costs = weight_changes * commission
    daily_costs = costs.sum(axis=1)
    
    # Net Returns
    net_rets = strat_rets - hedge_rets - costs
    daily_net_rets = daily_strat_rets - daily_costs - daily_hedge_rets 
    
    # Cumulative Returns
    cum_rets = (1 + net_rets).cumprod()
    daily_cum_rets = (1 + daily_net_rets).cumprod()
    
    return daily_net_rets, daily_cum_rets