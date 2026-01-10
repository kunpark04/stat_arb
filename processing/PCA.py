import numpy as np
import pandas as pd

def PCA_process(rets_window, num_factors):

    # Rename for clarity
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

    # Calculate percentage of variance explained by each factor
    variance_pct = eigenvalues / np.sum(eigenvalues)

    # For now focusing on 15 Factor model so num_factors=15, but can be adjusted for variable number by reaching a threshold of variance explained

    # Calculate weights & factor returns for eigenportfolio
    std = rets_PCA.std()
    eigenvectors_selected = eigenvectors.loc[:, 1:num_factors]
    eigen_weights = eigenvectors_selected.div(std, axis=0)

    return eigen_weights, variance_pct