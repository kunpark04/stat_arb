import numpy as np
import pandas as pd

def OU_process(cum_residuals):

    # Calculate OU parameters (1-lag regression model)
    X = cum_residuals.iloc[:-1, :]
    Y = cum_residuals.iloc[1:, :]

    # Univariate approach to calculate OU parameters
    numerator = ((X - X.mean(axis=0)).values * (Y - Y.mean(axis=0)).values).sum(axis=0)
    denominator = ((X - X.mean(axis=0))**2).sum(axis=0)

    # Calculate a and b values (mask for invalid b values)
    b = numerator / denominator
    mask = (b >= 1) | (b <= 0)
    b[mask] = 0.1
    a = Y.mean(axis=0) - (b * X.mean(axis=0))

    # Calculate zeta residuals (mask for invalid b values)
    zeta = Y.values - (a.values + b.values * X.values)
    zeta = pd.DataFrame(zeta,
                        index=Y.index,
                        columns=Y.columns)
    zeta.loc[:, mask] = 0

    # Calculate rest of OU metrics
    m = a / (1 - b)
    centered_m = m - m.mean()
    kappa = -np.log(b) * 252
    sigma_eq = zeta.std(ddof=1) / np.sqrt(1 - b**2)
    # s_score = -centered_m / sigma_eq # Assumes that cum_residuals of the last day is zero
    s_score = (cum_residuals.iloc[-1] - centered_m) / sigma_eq # Out-of-sample s_score

    # Mask OU metrics for invalid b values
    kappa[mask] = 0
    sigma_eq[mask] = 0
    s_score[mask] = 0

    return s_score, kappa, sigma_eq