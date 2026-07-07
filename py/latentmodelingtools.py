import numpy as np
import matplotlib.pyplot as plt

def weighted_whiten(X, weights, eps=1e-12):
    """
    X: (N, p)
    weights: length N

    Rescale and PCA-whiten data with weights.

    Returns:
    Xwhite: (N, p) whitened data
    mu: (p,) weighted means
    W: (p, p) whitening matrix
    """

    weights = np.asarray(weights)
    weights = weights / weights.sum()
    print(np.percentile(weights, [0, 1, 10, 50, 90, 99, 100]))

    # weighted mean
    mu = np.sum(weights[:, None] * X, axis=0)

    Xc = X - mu

    # Weighted std dev
    sigma = np.sqrt(np.sum(weights[:, None] * Xc**2, axis=0))

    Xs = Xc / sigma

    # weighted covariance
    C = (Xs * weights[:, None]).T @ Xs

    eigvals, eigvecs = np.linalg.eigh(C)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    
    # whitening matrix
    W = np.diag(1/sigma) @ eigvecs @ np.diag(1/np.sqrt(eigvals + eps))

    Xwhite = Xc @ W

    return Xwhite, mu, sigma, W


def plot_explained_variance(explained_variance_ratio):
    n_features = len(explained_variance_ratio)

    # Explained variance
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(1, n_features+1), explained_variance_ratio, color='steelblue')
    axes[0].set_xlabel('PCA Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Variance per Component')

    axes[1].plot(range(1, n_features+1), np.cumsum(explained_variance_ratio), 'ko-')
    axes[1].axhline(0.95, color='red', linestyle='--', label='95%')
    axes[1].axhline(0.99, color='orange', linestyle='--', label='99%')
    axes[1].set_xlabel('Number of PCA Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Cumulative Explained Variance')
    axes[1].legend()
    plt.tight_layout()
    plt.show()