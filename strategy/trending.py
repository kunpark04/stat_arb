import pandas as pd
import numpy as np
from models.EMA import compute_ema_crossover_signal, compute_adx, compute_volume_ratio


def signal_gen_trend(prices, high, low, rets, volume, start_trading_date,
                     window_EMA=60, fast_span=9, slow_span=20, adx_period=14):
    """
    Roll through trading dates and compute EMA crossover, ADX, and volume ratio signals.

    No-lookahead: window always excludes the current day via .iloc[:-1].

    Returns
    -------
    EMA_signal   : DataFrame (dates × stocks), values {-1, 0, +1}
    ADX          : DataFrame (dates × stocks), values [0, 100]
    Volume_ratio : DataFrame (dates × stocks), values [0, inf)
    """
    Dates = prices.index[prices.index >= start_trading_date]

    EMA_signal   = pd.DataFrame(index=Dates, columns=prices.columns, dtype=float)
    ADX_df       = pd.DataFrame(index=Dates, columns=prices.columns, dtype=float)
    Vol_ratio_df = pd.DataFrame(index=Dates, columns=prices.columns, dtype=float)

    for i, date in enumerate(Dates):
        prices_window = prices.loc[:date].iloc[:-1].tail(window_EMA)
        high_window   = high.loc[:date].iloc[:-1].tail(window_EMA)
        low_window    = low.loc[:date].iloc[:-1].tail(window_EMA)
        volume_window = volume.loc[:date].iloc[:-1].tail(21)

        ema_sig  = compute_ema_crossover_signal(prices_window, fast=fast_span, slow=slow_span)
        adx_vals = compute_adx(high_window, low_window, prices_window, period=adx_period)
        vol_rat  = compute_volume_ratio(volume_window)

        EMA_signal.loc[date]   = ema_sig
        ADX_df.loc[date]       = adx_vals
        Vol_ratio_df.loc[date] = vol_rat

        if (i + 1) % 300 == 0:
            print(f'  [trend signal] Processed {i + 1} days')
        elif date == Dates[-1]:
            print(f'  [trend signal] Processed {i + 1} days — done.')

    return EMA_signal, ADX_df, Vol_ratio_df


def gen_positions_trend(EMA_signal, ADX, Volume_ratio,
                        leverage=1.0, adx_min=20.0, vol_ratio_min=1.2,
                        max_position=0.10, regime_series=None, active_regimes=None):
    """
    Generate trend-following positions from pre-computed signals.

    Entry rules:
      Long:  EMA == +1  AND  ADX >= adx_min  AND  vol_ratio >= vol_ratio_min
      Short: EMA == -1  AND  ADX >= adx_min  AND  vol_ratio >= vol_ratio_min
      Flat:  any filter fails

    Parameters
    ----------
    active_regimes : None → active on ALL dates (standalone backtest)
                     tuple (e.g. (2,)) → zero weights outside those regimes (hybrid)
    regime_series  : pd.Series of regime labels indexed by date. Required when
                     active_regimes is not None.

    Returns
    -------
    state_history     : DataFrame (dates × stocks), values {-1, 0, +1}
    leveraged_weights : DataFrame (dates × stocks), equal-weight normalized to leverage
    """
    Dates = EMA_signal.index

    state_history    = pd.DataFrame(0.0, index=Dates, columns=EMA_signal.columns)
    weight_history   = pd.DataFrame(0.0, index=Dates, columns=EMA_signal.columns)

    for date in Dates:
        # Regime gate: zero out if not in active_regimes
        if active_regimes is not None and regime_series is not None:
            r = regime_series.get(date, None)
            if r is None or pd.isna(r) or int(r) not in active_regimes:
                state_history.loc[date]  = 0.0
                weight_history.loc[date] = 0.0
                continue

        ema  = EMA_signal.loc[date].astype(float)
        adx  = ADX.loc[date].astype(float)
        vrat = Volume_ratio.loc[date].astype(float)

        trend_filter = (adx >= adx_min) & (vrat >= vol_ratio_min)

        long_entry  = (ema ==  1) & trend_filter
        short_entry = (ema == -1) & trend_filter

        state = pd.Series(0.0, index=EMA_signal.columns)
        state[long_entry]  =  1.0
        state[short_entry] = -1.0

        # Equal-weight sizing (state already encodes direction)
        raw_weights = state.copy()

        state_history.loc[date]  = state.values
        weight_history.loc[date] = raw_weights.values

    # Normalize to target gross leverage
    gross_lev = weight_history.abs().sum(axis=1)
    norm_w    = weight_history.div(gross_lev, axis=0).fillna(0)
    leveraged_weights = norm_w * leverage

    # Per-stock position cap then re-normalize
    leveraged_weights = leveraged_weights.clip(-max_position, max_position)
    gross_post = leveraged_weights.abs().sum(axis=1)
    leveraged_weights = (leveraged_weights.div(gross_post, axis=0)
                                          .fillna(0)
                                          * leverage)

    return state_history, leveraged_weights


def calculate_pnl_trend(rets, weight_history, commission=0.0005,
                        vol_target=None, vol_lookback=22, regime_series=None):
    """
    Compute net trend-strategy returns.

    Thin wrapper around the same logic as strategy.calculate_pnl() but with
    Beta=None / Factor_rets=None (no factor hedge for the trending strategy).
    """
    import strategy.strategy as strat

    return strat.calculate_pnl(
        rets=rets,
        Beta=None,
        Factor_rets=None,
        weight_history=weight_history,
        commission=commission,
        vol_target=vol_target,
        vol_lookback=vol_lookback,
        regime_series=regime_series,
    )
