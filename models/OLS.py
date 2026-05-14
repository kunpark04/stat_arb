import numpy as np
import pandas as pd

def OLS_process(rets_window, eigenportfolio_rets, window_OU):

    # Select the relevant window of returns
    rets_OU = rets_window.tail(window_OU)

    # Vectorized OLS for residual calculation
    F = eigenportfolio_rets.tail(window_OU).copy()
    F.insert(0, 0, 1) # This is the intercept
    R = rets_OU

    # Calculate the beta coefficients (weight of each factor)
    beta_values, _, _, _ = np.linalg.lstsq(F, R, rcond=None)
    beta = pd.DataFrame(beta_values,
                        index=F.columns,
                        columns=R.columns)
    residuals = R - (F @ beta)
    cum_residuals = residuals.cumsum()
    
    return cum_residuals, beta