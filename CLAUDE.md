# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

```bash
python main.py
```

There are no automated tests, linters, or build steps. The full backtest pipeline runs end-to-end from `main.py`.

To refresh market data from yfinance (requires internet):
```python
# In main.py, set download=True and provide parameters
rets = fetch_data(download=True, start_date='2010-01-01', interval='1d', univ=None, file_name='price_volume_data.pkl')
```

To use cached data (default):
```python
rets = fetch_data(download=False, file_name='price_volume_data.pkl')
```

## Architecture

The strategy is based on Avellaneda & Lee (2010) — PCA extracts latent market factors, OLS regresses stock returns against those factors to get residuals, and an OU process fits those residuals to generate mean-reversion signals (S-scores).

### Data flow through `main.py`

```
price_data/fetch.py        → daily returns DataFrame (stocks × dates)
        ↓
strategy/strategy.py       → signal_gen()     → S_score, Kappa, Sigma_eq, Beta, Factor_rets
                           → gen_positions()  → state_history, leveraged_weights
                           → calculate_pnl()  → daily_net_rets, daily_cum_rets
        ↓
metrics/performance.py     → performance_analysis()
metrics/monte_carlo.py     → monte_carlo()
visualization/plots.py     → plot_equity(), plot_drawdown()
```

### Key modules

**`models/`** — Stateless math primitives, each takes a DataFrame window and returns computed values:
- `PCA.py`: Builds empirical correlation matrix, eigen-decomposes it, returns `eigen_weights` (stocks × factors) normalized by per-stock std
- `OLS.py`: Regresses stock returns on eigenportfolio returns over a rolling `window_OU`-day window; returns `cum_residuals` and `beta` coefficients
- `OU.py`: Fits a 1-lag AR model on `cum_residuals` to estimate OU parameters — `kappa` (mean-reversion speed, annualized), `sigma_eq` (equilibrium vol), and `s_score` (deviation from mean in sigma units)

**`strategy/strategy.py`** — Three sequential functions:
- `signal_gen()`: Rolls through trading dates, re-runs PCA every 30 days, OLS+OU every day; returns per-stock S-scores and OU params
- `gen_positions()`: State machine (0/1/-1) per stock; enters long at S < -2, short at S > 2, exits at ±0.5; filters by `kappa > 5`; weights by `1/sigma_eq`, then normalizes to target leverage
- `calculate_pnl()`: Applies weights with 1-day lag, subtracts beta-hedged factor returns and per-trade commission costs

**`price_data/fetch.py`** — Downloads from yfinance or loads from `price_volume_data.pkl`; the `.pkl` is the canonical cached data file

**`metrics/`** — `performance.py` computes Total Return, Sharpe, Max Drawdown; `monte_carlo.py` runs bootstrap simulations (with or without replacement) over strategy returns

### Important implementation details

- PCA runs every 30 days; OLS/OU runs every day — controlled by `days % 30 == 0` in `signal_gen()`
- Lookahead bias is avoided by excluding the current day from the rolling window: `rets.loc[:date].iloc[:-1].tail(window_PCA)`
- S-score uses out-of-sample formulation: `(cum_residuals.iloc[-1] - centered_m) / sigma_eq`
- OU `b` values outside `(0, 1)` are masked to `0.1` to avoid invalid kappa/sigma estimates
- Position weighting: `raw_weights = state / (sigma_eq + 1e-6)`, clipped to `[-2, 2]`, then normalized so gross leverage sums to `leverage` parameter
- Beta hedge in PnL: `(beta.T @ factor_rets) * weights` subtracts factor exposure from each position's return
- The `backtest_data/` folder exists for saving diagnostic outputs (currently unused in code)
