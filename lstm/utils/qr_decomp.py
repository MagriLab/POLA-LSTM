import numpy as np
from typing import Tuple

def qr_factorization(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Decompose matrix A into orthogonal matrix Q and an upper triangular matrix R.

    Args:
        A (np.ndarray): matrix that is to be decomposed

    Returns:
        Tuple[np.ndarray, np.ndarray]: matrices Q and R
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()

        for i in range(j):
            q = Q[:, i]
            R[i, j] = q.dot(v)
            v = v - R[i, j] * q

        norm = np.linalg.norm(v)
        Q[:, j] = v / norm
        R[j, j] = norm
    return Q, R