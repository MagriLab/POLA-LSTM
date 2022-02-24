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
