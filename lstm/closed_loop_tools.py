import random

import numpy as np
import tensorflow as tf

lorenz_dim = 3


def create_window_closed_loop(test_data, iteration, window_size=50, pred=np.array([])):
    if iteration == 0:
        return test_data[:window_size, :].reshape(1, window_size, lorenz_dim)
    if iteration < window_size:
        n_pred = pred.shape[0]
        idx_test_entries = (
            iteration + window_size - n_pred
        )  # end index of entries from the test data,
        test_data = test_data[iteration:idx_test_entries, :]
        return np.append(test_data, pred, axis=0).reshape(1, window_size, lorenz_dim)
    else:
        return pred[-window_size:, :].reshape(1, window_size, lorenz_dim)


def add_new_pred(pred_old, pred_new):
    return np.append(pred_old, pred_new, axis=0)


def compute_lyapunov_time_arr(time_vector, c_lyapunov=0.90566, window_size=50):
    t_lyapunov = 1 / c_lyapunov
    lyapunov_time = (time_vector[window_size:] -
                     time_vector[window_size]) / t_lyapunov
    return lyapunov_time


def select_random_window_with_label(df_transposed, n_windows, window_size=50):
    idx = random.sample(range(len(df_transposed) - window_size - 1), n_windows)
    # window_list =[]
    window_list = [
        df_transposed[i: i + window_size,
                      :].reshape(1, window_size, lorenz_dim)
        for i in idx
    ]
    label_list = [
        df_transposed[i + window_size + 1, :].reshape(1, lorenz_dim) for i in idx
    ]
    return window_list, label_list, idx


def select_random_batch_with_label(df_transposed, window_size=50, batch_size=32):
    idx_start = random.randint(0, len(df_transposed) - window_size - 1)
    idx = np.arange(start=idx_start, stop=idx_start + batch_size)
    # window_list =[]
    window_list = [
        df_transposed[i: i + window_size,
                      :].reshape(1, window_size, lorenz_dim)
        for i in idx
    ]
    label_list = [
        df_transposed[i + window_size + 1, :].reshape(1, lorenz_dim) for i in idx
    ]
    return window_list, label_list, idx


def select_random_batches_with_label(
    df_transposed, n_int, window_size=50, batch_size=32
):
    idx = random_int_window_dist(
        df_transposed, n_int, window_size=window_size, batch_size=batch_size
    )
    idx_list = np.array(
        [np.arange(start=idx_start, stop=idx_start + batch_size)
         for idx_start in idx]
    ).flatten()
    # for i in idx_list:
    #     print(i, len(df_transposed[i : i + 50, :]))
    window_list = [
        df_transposed[i: i + window_size,
                      :].reshape(1, window_size, lorenz_dim)
        for i in idx_list
    ]
    label_list = [
        df_transposed[i + window_size + 1, :].reshape(1, lorenz_dim) for i in idx_list
    ]
    return np.array(window_list), np.array(label_list), idx_list


def random_int_window_dist(df_transposed, n_int, window_size=50, batch_size=32):
    n_total_windows = int((len(df_transposed) - window_size) / batch_size)
    print(len(df_transposed))
    idx = [batch_size *
           i for i in sorted(random.sample(range(n_total_windows), n_int))]
    return idx


def add_cloop_prediction(df_transposed, idx, predictions, window_size=50):
    window_list = np.array(
        [
            np.append(
                df_transposed[idx[i] + 1: idx[i] + window_size, :],
                predictions[i, :].reshape(1, lorenz_dim),
                axis=0,
            ).reshape(window_size, lorenz_dim)
            for i in range(0, len(idx))
        ]
    )
    label_list = np.array([df_transposed[i + window_size + 2, :] for i in idx])
    return window_list, label_list


def prediction_closed_loop(model, time_test, df_test, n_length, window_size=50):
    lyapunov_time = compute_lyapunov_time_arr(
        time_test, window_size=window_size)
    test_window = create_window_closed_loop(
        df_test.transpose(), 0, window_size=window_size
    )

    predictions = model.predict(test_window)
    predictions = np.array(predictions).reshape(1, lorenz_dim)
    for iteration in range(1, n_length):
        test_window = create_window_closed_loop(
            df_test.transpose(), iteration, window_size=window_size, pred=predictions
        )
        new_pred = model.predict(test_window)
        new_pred = np.array(new_pred).reshape(1, lorenz_dim)
        predictions = add_new_pred(predictions, new_pred)
    return lyapunov_time, predictions
