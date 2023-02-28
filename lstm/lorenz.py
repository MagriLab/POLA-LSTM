import numpy as np
import tensorflow as tf


def l63(u, T, params):
    beta, rho, sigma = params
    x, y, z = u
    return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])


def l63_batch(pred: np.ndarray, sigma: float = 10, beta: float = 2.667, rho: float = 28) -> np.ndarray:
    """
    Function that applies the lorenz equations to a batch of initial conditions.

    Parameters:
    - pred (np.ndarray):  prediction of the LSTM (batch_size, time_steps, 3) 
    - sigma (float): sigma parameter in lorenz equations (default = 10)
    - beta (float): beta parameter in lorenz equations (default = 2.667)
    - rho (float): rho parameter in lorenz equations (default = 28)

    Returns:
    - np.ndarray: a numpy array of shape (batch_size, time_steps, 3) representing the result of the lorenz equations applied to each point in the pred array.
    """
    x = pred[:, :, 0]
    y = pred[:, :, 1]
    z = pred[:, :, 2]

    x_t = -sigma * (x - y)
    y_t = rho * x - y - x * z
    z_t = -beta * z + x * y
    return tf.stack((x_t, y_t, z_t), axis=-1)


def l63jac(u, params):
    beta, rho, sigma = params  # Unpack the constants vector
    x, y, z = u  # Unpack the state vector

    # Jacobian
    J = np.array([[-sigma, sigma,     0],
                  [rho-z,    -1,    -x],
                  [y,     x, -beta]])
    return J


def backward_diff(y_pred, delta_t=0.01):
    bd = (y_pred[:, 1:, :] - y_pred[:, :-1, :])/delta_t  # y_pred (batch, dim), x_batch (batch, window, dim)
    return tf.stack((bd[:, :, 0], bd[:, :, 1], bd[:, :, 2]), axis=-1)
