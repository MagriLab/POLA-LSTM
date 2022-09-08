import einops
import numpy as np
import tensorflow as tf


def compute_lyapunov_time_arr(time_vector, c_lyapunov=0.90566, window_size=50):
    t_lyapunov = 1 / c_lyapunov
    lyapunov_time = (time_vector[window_size:] -
                     time_vector[window_size]) / t_lyapunov
    return lyapunov_time


def create_df_3d(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(size=window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(
        lambda window: (window[:-1], window[1:])
    )
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, 3], [None, 3]))
    return dataset


def append_label_to_window(window, label):
    corr_label = einops.rearrange(label[:, -1, :], "i j -> i 1 j")
    return tf.concat((window, corr_label), axis=1)


def split_window_label(window_label_tensor, window_size=100):
    window = window_label_tensor[:, -(window_size):, :]
    return window


def create_test_window(df_test, window_size=100):
    test_window = tf.convert_to_tensor(df_test[:, :window_size].T)
    test_window = einops.rearrange(test_window, "i j -> 1 i j")
    return test_window


def prediction_closed_loop(model, time_test, df_test, n_length, window_size=100, c_lyapunov=0.90566):
    dim = df_test.shape[0]
    lyapunov_time = compute_lyapunov_time_arr(
        time_test, window_size=window_size, c_lyapunov=c_lyapunov)
    predictions = np.zeros((n_length, dim))
    test_window = create_test_window(df_test, window_size=window_size)
    for i in range(len(predictions)):
        pred = model.predict(test_window)
        predictions[i, :] = pred[0, -1, :]
        test_window = split_window_label(append_label_to_window(test_window, pred), window_size=window_size)
    return lyapunov_time, predictions


def lstm_step(window_input, h, c, model, i, dim=3):
    if i > 100:
        window_input = tf.reshape(tf.matmul(h, model.layers[1].get_weights()[
                                  0]) + model.layers[1].get_weights()[1], shape=(1, dim))
    z = tf.keras.backend.dot(window_input, model.layers[0].cell.kernel)
    z += tf.keras.backend.dot(h, model.layers[0].cell.recurrent_kernel)
    z = tf.keras.backend.bias_add(z, model.layers[0].cell.bias)

    z0, z1, z2, z3 = tf.split(z, 4, axis=1)

    i = tf.sigmoid(z0)
    f = tf.sigmoid(z1)
    c_new = f * c + i * tf.tanh(z2)
    o = tf.sigmoid(z3)

    h_new = o * tf.tanh(c_new)

    out = tf.matmul(h_new, model.layers[1].get_weights()[0]) + model.layers[1].get_weights()[1]
    return out, h_new, c_new