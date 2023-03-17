import numpy as np
import tensorflow as tf
import random
import einops


def df_train_valid_test_split(df, train_ratio=0.6, valid_ratio=0.2):
    len_df_col = df.shape[1]
    train = np.array(df[:, 0 : int(len_df_col * train_ratio)])
    valid = np.array(
        df[
            :,
            int(len_df_col * train_ratio) : int(
                len_df_col * (train_ratio + valid_ratio)
            ),
        ]
    )
    test = np.array(df[:, int(len_df_col * (train_ratio + valid_ratio)) :])
    return train, valid, test


def train_valid_test_split(df, train_ratio=0.6, valid_ratio=0.2):
    len_df_col = len(df)
    train = np.array(df[0 : int(len_df_col * train_ratio)])
    valid = np.array(
        df[
            int(len_df_col * train_ratio) : int(
                len_df_col * (train_ratio + valid_ratio)
            )
        ]
    )
    test = np.array(df[int(len_df_col * (train_ratio + valid_ratio)) :])
    return train, valid, test


def create_test_window(df_test, window_size=100):
    test_window = tf.convert_to_tensor(df_test[:, :window_size].T)
    test_window = einops.rearrange(test_window, "i j -> 1 i j")
    return test_window


def create_df_3d(series, window_size, batch_size, shuffle_buffer):
    n = series.shape[1]
    m = series.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(size=window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(
        lambda window: (window[:-1], window[-1])
    )
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, n], [None]))
    return dataset


def create_df_nd_mtm(
    series, window_size, batch_size, shuffle_buffer, shuffle_window=10
):
    n = series.shape[1]
    m = series.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(size=window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.shuffle(m * shuffle_window)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(
        lambda window: (window[:-1], window[1:])
    )
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, n], [None, n]))
    return dataset


def create_df_nd_even_md_mtm(
    series, window_size, batch_size, shuffle_buffer, n_random_idx=5, shuffle_window=10
):
    n = series.shape[1]
    m = series.shape[0]
    random.seed(0)
    idx_skip = int(n/n_random_idx)
    idx_lst = list(range(0, n))[::int(n/n_random_idx)]
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(size=window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.shuffle(m * shuffle_window)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(
        lambda window: (window[:-1, ::idx_skip], window[1:])
    )
    dataset = dataset.padded_batch(
        batch_size, padded_shapes=([None, n_random_idx], [None, n])
    )
    return idx_lst, dataset


def create_df_nd_random_md_mtm_idx(
    series, window_size, batch_size, shuffle_buffer, n_random_idx=15, shuffle_window=10
):
    n = series.shape[1]
    m = series.shape[0]
    random.seed(0)
    idx_lst = random.sample(range(n), n_random_idx)
    idx_lst.sort()
    print(idx_lst)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(size=window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.shuffle(m * shuffle_window)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(
        lambda window: (tf.gather(window[:-1, :], idx_lst, axis=1), window[1:])
    )
    dataset = dataset.padded_batch(
        batch_size, padded_shapes=([None, n_random_idx], [None, n])
    )
    return idx_lst, dataset
