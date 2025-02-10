import numpy as np
import scipy.stats as st


def column_normalize(X):
    """Column normalize X."""
    return (X - X.mean(axis=0)) / (X.std(axis=0) * np.sqrt(X.shape[1]))


def toRanks(A):
    """Convert the columns of A to ranks."""
    AA = np.zeros_like(A)
    for i in range(A.shape[1]):
        AA[:, i] = st.rankdata(A[:, i])
    AA = np.array(np.around(AA), dtype="int") - 1
    return AA


def gaussianize(Y):
    """Gaussianize X: [samples x phenotypes].

    each phentoype is converted to ranks and transformed back to normal using the inverse CDF
    """
    N, P = Y.shape

    YY = toRanks(Y)
    quantiles = (np.arange(N) + 0.5) / N
    gauss = st.norm.isf(quantiles)
    Y_gauss = np.zeros((N, P))
    for i in range(P):
        Y_gauss[:, i] = gauss[YY[:, i]]
    Y_gauss *= -1
    return Y_gauss
