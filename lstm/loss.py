import numpy as np
import tensorflow as tf


def loss_oloop(y_true, y_pred, washout=0):
    mse = tf.keras.losses.MeanSquaredError()  # reduction=tf.keras.losses.Reduction.SUM
    loss = mse(y_true[washout:, :], y_pred[washout:, :])
    return loss


def loss_oloop_reg(y_true, y_pred, washout=10, reg_weight=0.001):
    mse = tf.keras.losses.MeanSquaredError()  # reduction=tf.keras.losses.Reduction.SUM
    loss = mse(y_true[washout:, :], y_pred[washout:, :]) + reg_weight * tf.nn.l2_loss(
        y_pred[washout:, :]
    )
    return loss


def forward_diff(prediction, delta_t=0.01):
    der = (prediction[1:] - prediction[:-1])/delta_t


def pi_loss(y_true, y_pred, washout=10, reg_weight=0.001):
    mse = tf.keras.losses.MeanSquaredError()
    # reduction=tf.keras.losses.Reduction.SUM
    loss = mse(y_true[washout:, :], y_pred[washout:, :]) + mse(forward_diff(y_pred) - der)
    return loss
