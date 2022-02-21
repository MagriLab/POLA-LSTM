import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
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


def select_random_window_with_label(df_transposed, n_windows):
    idx = random.sample(range(len(df_transposed) - 50 - 1), n_windows)
    # window_list =[]
    window_list = [df_transposed[i : i + 50, :].reshape(1, 50, 3) for i in idx]
    label_list = [df_transposed[i + 51, :].reshape(1, 3) for i in idx]
    return window_list, label_list, idx


def select_random_batch_with_label(df_transposed):
    idx_start = random.randint(0, len(df_transposed) - 50 - 1)
    idx = np.arange(start=idx_start, stop=idx_start + 32)
    # window_list =[]
    window_list = [df_transposed[i : i + 50, :].reshape(1, 50, 3) for i in idx]
    label_list = [df_transposed[i + 51, :].reshape(1, 3) for i in idx]
    return window_list, label_list, idx


def select_random_batches_with_label(
    df_transposed, n_int, window_size=50, batch_size=32
):
    idx = random_int_window_dist(
        df_transposed, n_int, window_size=window_size, batch_size=batch_size
    )
    idx_list = np.array(
        [np.arange(start=idx_start, stop=idx_start + 32) for idx_start in idx]
    ).flatten()
    # for i in idx_list:
    #     print(i, len(df_transposed[i : i + 50, :]))
    window_list = [df_transposed[i : i + 50, :].reshape(1, 50, 3) for i in idx_list]
    label_list = [df_transposed[i + 51, :].reshape(1, 3) for i in idx_list]
    return np.array(window_list), np.array(label_list), idx_list


def random_int_window_dist(df_transposed, n_int, window_size=50, batch_size=32):
    idx = [batch_size * i for i in sorted(random.sample(range(185), n_int))]
    return idx


def add_cloop_prediction(df_transposed, idx, predictions):
    window_list = np.array(
        [
            np.append(
                df_transposed[idx[i] + 1 : idx[i] + 50, :],
                predictions[i, :].reshape(1, 3),
                axis=0,
            ).reshape(50, 3)
            for i in range(0, len(idx))
        ]
    )
    label_list = np.array([df_transposed[i + 52, :] for i in idx])
    return window_list, label_list


def loss_oloop(y_true, y_pred, washout=10):
    mse = tf.keras.losses.MeanSquaredError()  # reduction=tf.keras.losses.Reduction.SUM
    loss = mse(y_true[washout:, :], y_pred[washout:, :])  # (batchsize, dimensions)
    return loss


def loss_oloop_reg(y_true, y_pred, washout=10):
    mse = tf.keras.losses.MeanSquaredError()  # reduction=tf.keras.losses.Reduction.SUM
    loss = mse(y_true[washout:, :], y_pred[washout:, :]) + 0.001 * tf.nn.l2_loss(
        y_pred[washout:, :]
    )  # (batchsize, dimensions)
    return loss


def build_open_loop_lstm(cells=100):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.LSTM(cells, activation="relu", name="LSTM_1"),
            tf.keras.layers.Dense(3, name="Dense_1"),
        ]
    )
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss=loss_oloop, optimizer="adam", metrics=["mse"])
    return model


def train_oloop(model, epochs, train_dataset, batch_size):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(
                    x_batch_train, training=True
                )  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = loss_oloop(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))


def load_open_loop_lstm(model_checkpoint):
    # Create a new model instance
    model = build_open_loop_lstm()
    # Restore the weights
    model.load_weights(model_checkpoint)
    return model


def prediction_closed_loop(model, time_test, df_test, n_length, window_size=50):
    lyapunov_time = compute_lyapunov_time_arr(time_test, window_size=window_size)
    test_window = create_window_closed_loop(
        df_test.transpose(), 0, window_size=window_size
    )

    predictions = model.predict(test_window)
    predictions = np.array(predictions).reshape(1, 3)
    for iteration in range(1, n_length):
        test_window = create_window_closed_loop(
            df_test.transpose(), iteration, window_size=window_size, pred=predictions
        )
        new_pred = model.predict(test_window)
        new_pred = np.array(new_pred).reshape(1, 3)
        predictions = add_new_pred(predictions, new_pred)
    return lyapunov_time, predictions
