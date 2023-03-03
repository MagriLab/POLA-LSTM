import numpy as np
import tensorflow as tf

def l96(x, p=8):
    """ DE of the lorenz 96 system.

    Args:
        x (tf.Tensor/np.array): prediction/solution at time t
        p (int, optional): Forcing term F. Defaults to 8.

    Returns:
        tf.Tensor/np.array: rhs at t of l96 equations
    """

    return np.roll(x, 1, axis=None) * (np.roll(x, -1, axis=None) - np.roll(x, 2, axis=None)) - x + p

def l96jac(x, p=8):
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

def l96_batch(x, p=8):
    """ DE of the lorenz 96 system.

    Args:
        x (tf.Tensor/np.array): prediction/solution at time t
        p (int, optional): Forcing term F. Defaults to 8.

    Returns:
        tf.Tensor/np.array: rhs at t of l96 equations
    """

    return tf.roll(x, 1, axis=2) * (tf.roll(x, -1, axis=2) - tf.roll(x, 2, axis=2)) - x + p
    
def backward_diff(y_pred, delta_t=0.01):
    bd = (y_pred[:, 1:, :] - y_pred[:, :-1, :])/delta_t  # y_pred (batch, dim), x_batch (batch, window, dim)
    return bd

def RK4_step_l96(u, delta_t):
    K1 = l96_batch(u)
    K2 = l96_batch(u+ delta_t*K1/2.0)
    K3 = l96_batch(u + delta_t*K2/2.0)
    K4 = l96_batch(u + delta_t*K3)
    u = u + delta_t * (K1/2.0 + K2 + K3 + K4/2.0) / 3.0
    return u
