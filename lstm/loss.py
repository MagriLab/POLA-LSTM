import numpy as np
import tensorflow as tf

from .lorenz import lorenz, norm_lorenz

x_max = 19.619508366918392
y_max = 27.317051197038307
z_max = 48.05371246231375


@tf.function
def loss_oloop(y_true, y_pred, washout=0):
    mse = tf.keras.losses.MeanSquaredError()
    loss = mse(y_true[washout:, :], y_pred[washout:, :])
    return loss


@tf.function
def loss_oloop_l2_reg(y_true, y_pred, washout=10, reg_weight=0.001):
    mse = tf.keras.losses.MeanSquaredError()
    loss = mse(y_true[washout:, :], y_pred[washout:, :]) + reg_weight * tf.nn.l2_loss(
        y_pred[washout:, :]
    )
    return loss


def norm_backward_diff(y_pred, x_batch_train, delta_t=0.01, norm=True):
    bd = (y_pred[:, :] - x_batch_train[:, -1, :])/delta_t  # y_pred (batch, dim), x_batch (batch, window, dim)
    if norm == False:
        return bd[:, 0], bd[:, 1], bd[:, 2]
    else:
        return bd[:, 0]/y_max, bd[:, 1]/z_max/x_max, bd[:, 2]/x_max/y_max


@tf.function
def pi_loss(y_pred, x_batch_train, washout=0):
    """_summary_

    Args:
        y_pred (Tensor): network prediction
        x_batch_train: one batch of training windows
        washout (int, optional): to attenuate initialisation. Defaults to 0.
    Returns:
        _type_: _description_
    """
    mse = tf.keras.losses.MeanSquaredError()
    x_t, y_t, z_t = lorenz(y_pred)  # generate rhs of Lorenz equations
    # compute backward diff for all the predictions in a batch
    x_fd, y_fd, z_fd = norm_backward_diff(y_pred, x_batch_train, norm=False)
    pi_loss = mse(x_t, x_fd) + mse(y_t, y_fd) + mse(z_t, z_fd)  # compute mse for each dimension
    return pi_loss


def norm_pi_loss(y_pred, x_batch_train, washout=0, total_points=10000):
    """_summary_

    Args:
        y_pred (Tensor): network prediction
        x_batch_train: one batch of training windows
        washout (int, optional): to attenuate initialisation. Defaults to 0.
    Returns:
        _type_: _description_
    """
    # max_from_norm(total_points=total_points)
    mse = tf.keras.losses.MeanSquaredError()
    x_t, y_t, z_t = norm_lorenz(y_pred)  # generate rhs of Lorenz equations
    # compute backward diff for all the predictions in a batch
    x_fd, y_fd, z_fd = norm_backward_diff(y_pred, x_batch_train)
    pi_loss = mse(x_t, x_fd) + mse(y_t, y_fd) + mse(z_t, z_fd)  # compute mse for each dimension
    return pi_loss
