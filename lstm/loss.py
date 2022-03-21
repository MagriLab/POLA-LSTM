import numpy as np
import tensorflow as tf


# def max_from_norm(total_points = 100000):
#     global x_max, y_max, z_max
#     if total_points == 100000:
#         x_max = 19.62036351364186
#         y_max = 27.31708182056948
#         z_max = 48.05263683702385
#     elif total_points == 10000:
        # x_max = 19.61996107472211
        # y_max = 27.317071267968995
        # z_max = 48.05315303164703
x_max = 19.619508366918392 
y_max = 27.317051197038307
z_max = 48.05371246231375



@tf.function
def loss_oloop(y_true, y_pred, washout=0):
    mse = tf.keras.losses.MeanSquaredError()  # reduction=tf.keras.losses.Reduction.SUM
    loss = mse(y_true[washout:, :], y_pred[washout:, :])
    return loss


@tf.function
def loss_oloop_reg(y_true, y_pred, washout=10, reg_weight=0.001):
    mse = tf.keras.losses.MeanSquaredError()  # reduction=tf.keras.losses.Reduction.SUM
    loss = mse(y_true[washout:, :], y_pred[washout:, :]) + reg_weight * tf.nn.l2_loss(
        y_pred[washout:, :]
    )
    return loss


def backward_diff(y_pred, x_batch_train, delta_t=0.01):
    """ compute the backward difference using the network
        prediction and the last time snapshot from the training window.

    Args:
        y_pred (Tensor): network prediction
        x_batch_train: batch of training windows
        delta_t (float): time step. Defaults to 0.01.

    Returns:
        _type_: the values of backward difference for each component
    """
    bd = (y_pred[:, :] - x_batch_train[:, -1, :])/delta_t  # y_pred (batch, dim), x_batch (batch, window, dim)
    return bd[:, 0], bd[:, 1], bd[:, 2]

def norm_backward_diff(y_pred, x_batch_train, delta_t=0.01):
    bd = (y_pred[:, :] - x_batch_train[:, -1, :])/delta_t  # y_pred (batch, dim), x_batch (batch, window, dim)
    return bd[:, 0]/y_max, bd[:, 1]/z_max/x_max, bd[:, 2]/x_max/y_max


def lorenz(pred, sigma=10, beta=2.667, rho=28):
    """ use chaotic Lorenz system to generate right hand side

    Args:
        pred (3D Tensor): network prediction
        sigma (int):  Defaults to 10.
        beta (float): Defaults to 2.667.
        rho (int): Defaults to 28.

    Returns:
        right hand side
    """
    x = pred[:, 0]
    y = pred[:, 1]
    z = pred[:, 2]
    x_t = -sigma * (x - y)
    y_t = rho * x - y - x * z
    z_t = -beta * z + x * y
    return x_t, y_t, z_t


def norm_lorenz(pred, sigma=10, beta=2.667, rho=28):

    x = pred[:, 0]
    y = pred[:, 1]
    z = pred[:, 2]

    x_t = -sigma * (x/y_max - y/x_max)
    y_t = rho * x/y_max/z_max - y/x_max/z_max - x * z/y_max
    z_t = -beta * z/x_max/y_max + x * y/z_max
    return x_t, y_t, z_t


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
    x_fd, y_fd, z_fd = backward_diff(y_pred, x_batch_train)  # compute backward diff for all the predictions in a batch
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
    #max_from_norm(total_points=total_points)
    mse = tf.keras.losses.MeanSquaredError()
    x_t, y_t, z_t = norm_lorenz(y_pred)  # generate rhs of Lorenz equations
    x_fd, y_fd, z_fd = norm_backward_diff(y_pred, x_batch_train)  # compute backward diff for all the predictions in a batch
    pi_loss = mse(x_t, x_fd) + mse(y_t, y_fd) + mse(z_t, z_fd)  # compute mse for each dimension
    return pi_loss

@tf.function
def bd_loss(y_pred, x_batch_train, y_batch_train, washout=0):
    """_summary_

    Args:
        y_pred (Tensor): network prediction
        x_batch_train: one batch of training windows
        washout (int, optional): to attenuate initialisation. Defaults to 0.
    Returns:
        _type_: _description_
    """
    mse = tf.keras.losses.MeanSquaredError()
    x_t, y_t, z_t = backward_diff(y_batch_train, x_batch_train)  # generate rhs of Lorenz equations
    x_fd, y_fd, z_fd = backward_diff(y_pred, x_batch_train)  # compute backward diff for all the predictions in a batch
    pi_loss = mse(x_t, x_fd) + mse(y_t, y_fd) + mse(z_t, z_fd)  # compute mse for each dimension
    return pi_loss
