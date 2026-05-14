import pandas as pd
import numpy as np
from models.PCA import PCA_process
from models.OLS import OLS_process
from models.OU import OU_process
from strategy.regime_params import REGIME_PARAMS, DEFAULT_REGIME


def signal_gen(rets, start_trading_date, num_factors=None, variance_threshold=0.55,
               window_PCA=252, window_OU=60):
    """
    Roll through trading dates, run PCA every 30 days, OLS+OU every day.
    Returns per-stock S-scores and OU parameters.

    Parameters
    ----------
    num_factors        : Upper cap on PCA factors (None = no cap).
    variance_threshold : Minimum cumulative variance to explain; overrides num_factors
                         if fewer factors suffice. Default 0.55.
    """
    days = 0
    Dates = rets.index[rets.index >= start_trading_date]

    # Initialize output DataFrames
    Factor_rets = pd.DataFrame(index=Dates, columns=range(1, (num_factors or 20) + 1))
    S_score     = pd.DataFrame(index=Dates, columns=rets.columns)
    Kappa       = pd.DataFrame(index=Dates, columns=rets.columns)
    Sigma_eq    = pd.DataFrame(index=Dates, columns=rets.columns)
    Beta        = {}

    # Initialize valid_columns before the loop so it's always defined
    valid_columns = rets.columns
    eigen_weights = None

    for date in Dates:

        # Rolling window excluding current day (no lookahead bias)
        rets_window = rets.loc[:date].iloc[:-1].tail(window_PCA)

        # Recompute PCA every 30 days
        if days % 30 == 0:
            nan_count = rets_window.isna().sum()
            valid_columns = rets_window.columns[nan_count == 0]
            rets_window = rets_window[valid_columns].fillna(0)

            eigen_weights, variance_pct = PCA_process(rets_window,
                                                      num_factors=num_factors,
                                                      variance_threshold=variance_threshold)
        else:
            rets_window = rets_window[valid_columns].fillna(0)

        # Eigenportfolio returns for the current window
        eigenportfolio_rets = rets_window @ eigen_weights

        # Current day's actual factor returns (for beta hedge in PnL)
        current_ret = rets.loc[date, valid_columns].fillna(0)
        current_factor_rets = current_ret @ eigen_weights
        n_factors_used = eigen_weights.shape[1]

        # Re-index Factor_rets columns dynamically if factor count changed
        if n_factors_used not in Factor_rets.columns:
            for c in range(1, n_factors_used + 1):
                if c not in Factor_rets.columns:
                    Factor_rets[c] = float('nan')
        Factor_rets.loc[date, range(1, n_factors_used + 1)] = current_factor_rets.values

        # OLS + OU
        cum_residuals, beta = OLS_process(rets_window, eigenportfolio_rets, window_OU)
        s_score, kappa, sigma_eq = OU_process(cum_residuals)

        # Store results
        S_score.loc[date, valid_columns]  = s_score
        Kappa.loc[date, valid_columns]    = kappa
        Sigma_eq.loc[date, valid_columns] = sigma_eq
        Beta[date] = beta

        days += 1
        if days % 300 == 0:
            print(f'  Processed {days} days')
        elif date == Dates[-1]:
            print(f'  Processed {days} days — done.')

    return S_score, Kappa, Sigma_eq, Beta, Factor_rets


#=========================================================================#


def gen_positions(S_score, Kappa, Sigma_eq, leverage=1.0,
                  entry_threshold=2.0, exit_threshold=0.5, stop_loss=3.0,
                  kappa_min=5.0, kappa_percentile=None,
                  max_position=0.15, cooldown_days=5,
                  regime_series=None):
    """
    State-machine position generator.

    Parameters
    ----------
    entry_threshold  : Enter long at S < -entry_threshold, short at S > +entry_threshold.
    exit_threshold   : Exit long at S > -exit_threshold, exit short at S < +exit_threshold.
    stop_loss        : Stop-loss exit at |S| > stop_loss.
    kappa_min        : Minimum kappa (mean-reversion speed) to allow entry.
    kappa_percentile : If set (e.g. 0.5), only trade stocks with kappa >= this percentile
                       among stocks with kappa > kappa_min. Adapts to market conditions.
    max_position     : Maximum absolute weight per stock after normalization (e.g. 0.15 = 15%).
    cooldown_days    : Days to block re-entry after a stop-loss exit.
    regime_series    : Optional pd.Series of regime labels (0/1/2) indexed by date.
                       When provided, overrides entry/exit/stop/kappa/position params per day.
    """
    Dates = S_score.index
    current_state = pd.Series(0, index=S_score.columns)
    cooldown      = pd.Series(0, index=S_score.columns)

    state_history        = pd.DataFrame(index=Dates, columns=S_score.columns, dtype=float)
    weight_history       = pd.DataFrame(index=Dates, columns=S_score.columns, dtype=float)
    leverage_history     = pd.Series(index=Dates, dtype=float)
    max_position_history = pd.Series(index=Dates, dtype=float)

    for date in Dates:
        # ── Regime-adaptive parameters ────────────────────────────────────────
        if regime_series is not None:
            r = regime_series.get(date, None)
            r = DEFAULT_REGIME if (r is None or pd.isna(r)) else int(r)
            p = REGIME_PARAMS[r]
            _entry, _exit, _sl = p.entry_threshold, p.exit_threshold, p.stop_loss
            _kmin, _maxpos, _lev = p.kappa_min, p.max_position, p.leverage
        else:
            _entry, _exit, _sl = entry_threshold, exit_threshold, stop_loss
            _kmin, _maxpos, _lev = kappa_min, max_position, leverage

        leverage_history[date]     = _lev
        max_position_history[date] = _maxpos

        s_score  = S_score.loc[date].astype(float)
        kappa    = Kappa.loc[date].astype(float)
        sigma_eq = Sigma_eq.loc[date].astype(float)

        # --- Exit logic (detect stop-loss exits before clearing state) ---
        sl_long  = (current_state == 1)  & (s_score < -_sl)
        sl_short = (current_state == -1) & (s_score >  _sl)
        cooldown[sl_long | sl_short] = cooldown_days   # cooldown on actual SL exits

        exit_long  = (current_state == 1)  & ((s_score > -_exit) | (s_score < -_sl))
        exit_short = (current_state == -1) & ((s_score < _exit)  | (s_score >  _sl))
        current_state[exit_long | exit_short] = 0

        # --- Validity filter ---
        valid_kappa = kappa.notna() & (kappa > _kmin)
        if kappa_percentile is not None:
            eligible = kappa[valid_kappa]
            if len(eligible) > 0:
                threshold_val = eligible.quantile(kappa_percentile)
                valid_kappa = valid_kappa & (kappa >= threshold_val)

        valid_data = valid_kappa & (sigma_eq > 0) & (cooldown == 0)

        # --- Entry logic ---
        enter_long  = (current_state == 0) & (s_score < -_entry) & valid_data
        enter_short = (current_state == 0) & (s_score >  _entry) & valid_data
        current_state[enter_long]  = 1
        current_state[enter_short] = -1

        # --- Position weighting: inverse-vol sizing ---
        raw_weights = current_state / (sigma_eq + 1e-6)
        raw_weights = raw_weights.clip(-2.0, 2.0)
        raw_weights[~valid_data & (current_state == 0)] = 0  # zero out flat invalid positions

        state_history.loc[date]  = current_state.values
        weight_history.loc[date] = raw_weights.values

        # Decrement cooldown
        cooldown = (cooldown - 1).clip(lower=0)

    # Normalize to target gross leverage (per-date when regime_series provided)
    gross_lev = weight_history.abs().sum(axis=1)
    norm_w    = weight_history.div(gross_lev, axis=0).fillna(0)
    leveraged_weights = norm_w.mul(leverage_history, axis=0)

    # Per-date position cap then re-normalize
    maxpos_series = max_position_history.fillna(max_position)
    leveraged_weights = leveraged_weights.clip(
        -maxpos_series, maxpos_series, axis=0)
    gross_post = leveraged_weights.abs().sum(axis=1)
    leveraged_weights = (leveraged_weights.div(gross_post, axis=0)
                                          .fillna(0)
                                          .mul(leverage_history, axis=0))

    return state_history, leveraged_weights


#=========================================================================#


def calculate_pnl(rets, Beta, Factor_rets, weight_history, commission=0.0005,
                  vol_target=None, vol_lookback=22, regime_series=None):
    """
    Compute net strategy returns after beta hedge and transaction costs.

    Parameters
    ----------
    vol_target     : If set (e.g. 0.10), scale daily weights so the portfolio targets
                     this annualized volatility. Uses a rolling realized-vol estimate.
    vol_lookback   : Rolling window (days) for realized vol estimate.
    regime_series  : Optional pd.Series of regime labels (0/1/2); when provided,
                     the per-regime vol_target from REGIME_PARAMS overrides vol_target.
    """
    Dates = weight_history.index
    effective_weights = weight_history.shift(1).fillna(0)

    aligned_rets = rets.loc[Dates]

    # --- Strategy gross returns ---
    strat_rets      = effective_weights * aligned_rets
    daily_strat_rets = strat_rets.sum(axis=1)

    # --- Beta hedge ---
    if Beta is not None and Factor_rets is not None:
        hedge_list = []
        for date in Dates:
            weights      = effective_weights.loc[date]
            factor_rets  = Factor_rets.loc[date].dropna()
            beta_df      = Beta[date].iloc[1:, :]  # drop intercept row

            # Align beta columns to valid_columns in weights
            common_stocks  = weights.index.intersection(beta_df.columns)
            common_factors = factor_rets.index.intersection(beta_df.index)
            if len(common_stocks) == 0 or len(common_factors) == 0:
                hedge_list.append(pd.Series(0.0, index=weights.index))
                continue

            beta_sub   = beta_df.loc[common_factors, common_stocks]
            frets_sub  = factor_rets.loc[common_factors]
            w_sub      = weights.loc[common_stocks]
            hedge_vals = (beta_sub.T @ frets_sub) * w_sub

            result = pd.Series(0.0, index=weights.index, dtype=float)
            result.loc[common_stocks] = hedge_vals.astype(float)
            hedge_list.append(result)

        hedge_rets       = pd.DataFrame(hedge_list, index=Dates)
        daily_hedge_rets = hedge_rets.sum(axis=1)
    else:
        daily_hedge_rets = pd.Series(0.0, index=Dates)

    # --- Transaction costs ---
    weight_changes = weight_history.diff().abs()
    daily_costs    = (weight_changes * commission).sum(axis=1)

    # --- Net returns (pre vol-target) ---
    daily_net_rets = daily_strat_rets - daily_costs - daily_hedge_rets

    # --- Volatility targeting ---
    if vol_target is not None or regime_series is not None:
        realized_vol = daily_net_rets.rolling(vol_lookback).std() * np.sqrt(252)
        realized_vol = realized_vol.shift(1).ffill().bfill()

        if regime_series is not None:
            fallback = REGIME_PARAMS[DEFAULT_REGIME].vol_target
            vol_target_series = (regime_series.reindex(Dates)
                                 .apply(lambda r: REGIME_PARAMS[int(r)].vol_target
                                                  if pd.notna(r) else fallback))
        else:
            vol_target_series = pd.Series(vol_target, index=Dates)

        vol_scalar = vol_target_series.div(realized_vol).clip(0.25, 4.0)
        daily_net_rets = daily_net_rets * vol_scalar

    daily_cum_rets = (1 + daily_net_rets).cumprod()

    return daily_net_rets, daily_cum_rets
