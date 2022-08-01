import numpy as np


def l96(x, p=8):
    """ DE of the lorenz 96 system.

    Args:
        x (tf.Tensor/np.array): prediction/solution at time t
        p (int, optional): Forcing term F. Defaults to 8.

    Returns:
        tf.Tensor/np.array: rhs at t of l96 equations
    """

    return np.roll(x, 1) * (np.roll(x, -1) - np.roll(x, 2)) - x + p


def l96jac(x, p):
    """returns the Jacobian of the L96 systems

    Args:
        x (Tensor/array): prediction/solution at time t
        p (int, optional): Forcing term F. Defaults to 8.

    Returns:
        J : Jacobian of L96 rhs at time t 
    """
    D = len(x)
    J = np.zeros((D, D), dtype='float')
    for i in range(D):
        J[i, (i-1) % D] = x[(i+1) % D] - x[(i-2) % D]
        J[i, (i+1) % D] = x[(i-1) % D]
        J[i, (i-2) % D] = -x[(i-1) % D]
        J[i, i] = -1.0
    return J
