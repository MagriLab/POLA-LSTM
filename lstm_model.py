import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tensorflow_datasets as tfds
import tensorflow as tf
from data_processing import (
    create_training_split,
    df_training_split,
    create_df_3d,
    create_window_closed_loop,
    add_new_pred,
    compute_lyapunov_time_arr,
    train_valid_test_split,
)


def custom_loss(y_true, y_pred, washout=0):
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    loss = mse(y_true[washout:, :], y_pred[washout:, :])  # (batchsize, dimensions)
    return loss


def build_open_loop_lstm():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.LSTM(50, activation="relu", name="LSTM_1"),
            tf.keras.layers.Dense(3, name="Dense_1"),
        ]
    )
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(loss=custom_loss, optimizer="adam")
    return model


def load_open_loop_lstm(model_checkpoint):
    # Create a new model instance
    model = build_open_loop_lstm()
    # Restore the weights
    model.load_weights(model_checkpoint)
    return model


def prediction_closed_loop(model, time_test, df_test, n_length):
    lyapunov_time = compute_lyapunov_time_arr(time_test)
    test_window = create_window_closed_loop(df_test.transpose(), 0)
    predictions = model.predict(test_window)
    for iteration in range(1, n_length):
        test_window = create_window_closed_loop(
            df_test.transpose(), iteration, predictions
        )
        new_pred = model.predict(test_window)
        predictions = add_new_pred(predictions, new_pred)
    return lyapunov_time, predictions
