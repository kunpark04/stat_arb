import numpy as np
import pandas as pd

def PCA_process(rets_window, num_factors=None, variance_threshold=0.55):
    """
    Run PCA on the return window and return eigen_weights for eigenportfolios.

    Factor selection: selects the minimum number of factors that explain
    >= variance_threshold of total variance. num_factors acts as an upper cap.
    If variance_threshold is None, num_factors is used directly.
    """
    rets_PCA = rets_window.copy()

    # Construct empirical correlation matrix
    corr_matrix = rets_PCA.corr()

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    eigenvectors = pd.DataFrame(eigenvectors,
                                index=rets_PCA.columns,
                                columns=np.arange(1, len(eigenvalues) + 1))

    # Percentage of variance explained by each factor
    variance_pct = eigenvalues / np.sum(eigenvalues)

    # Adaptive factor selection: minimum factors to reach variance_threshold
    if variance_threshold is not None:
        cumvar = np.cumsum(variance_pct)
        n_adaptive = int(np.searchsorted(cumvar, variance_threshold)) + 1
        if num_factors is not None:
            n_factors = min(n_adaptive, num_factors)
        else:
            n_factors = n_adaptive
    else:
        n_factors = num_factors

    # Calculate eigen_weights (normalized by per-stock std)
    std = rets_PCA.std()
    eigenvectors_selected = eigenvectors.loc[:, 1:n_factors]
    eigen_weights = eigenvectors_selected.div(std, axis=0)

    return eigen_weights, variance_pct
