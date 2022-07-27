import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def create_training_split(df, ratio=0.7):
    len_df = len(df)
    train = np.array(df[0: int(len_df * ratio)])
    test = np.array(df[int(len_df * ratio):])
    return train, test


def df_training_split(df, ratio=0.7):
    len_df_col = df.shape[1]
    train = np.array(df[:, 0: int(len_df_col * ratio)])
    test = np.array(df[:, int(len_df_col * ratio):])
    return train, test


def train_valid_test_split(df, train_ratio=0.6, valid_ratio=0.2):
    len_df_col = len(df)
    train = np.array(df[0: int(len_df_col * train_ratio)])
    valid = np.array(
        df[
            int(len_df_col * train_ratio): int(
                len_df_col * (train_ratio + valid_ratio)
            )
        ]
    )
    test = np.array(df[int(len_df_col * (train_ratio + valid_ratio)):])
    return train, valid, test


def df_train_valid_test_split(df, train_ratio=0.6, valid_ratio=0.2):
    len_df_col = df.shape[1]
    train = np.array(df[:, 0: int(len_df_col * train_ratio)])
    valid = np.array(
        df[
            :,
            int(len_df_col * train_ratio): int(
                len_df_col * (train_ratio + valid_ratio)
            ),
        ]
    )
    test = np.array(df[:, int(len_df_col * (train_ratio + valid_ratio)):])
    return train, valid, test


def create_df_3d(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(size=window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    # dataset = dataset.shuffle(7).map(lambda window: (window[:-1], window[-1]))#separates each window into features and label (next/last value)
    dataset = dataset.shuffle(shuffle_buffer).map(
        lambda window: (window[:-1], window[-1])
    )
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, 3], [None]))
    return dataset


def create_df_3d_mtm(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(size=window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    # dataset = dataset.shuffle(7).map(lambda window: (window[:-1], window[-1]))#separates each window into features and label (next/last value)
    dataset = dataset.shuffle(shuffle_buffer).map(
        lambda window: (window[:-1], window[1:])
    )
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, 3], [None, 3]))
    return dataset


def create_df_nd_mtm(series, window_size, batch_size, shuffle_buffer):
    n = series.shape[1]
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(size=window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    # dataset = dataset.shuffle(7).map(lambda window: (window[:-1], window[-1]))#separates each window into features and label (next/last value)
    dataset = dataset.shuffle(shuffle_buffer).map(
        lambda window: (window[:-1], window[1:])
    )
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, n], [None, n]))
    return dataset

def create_df_3d_mtm_random(series, window_size, batch_size, shuffle_buffer, new_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(size=window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    # dataset = dataset.shuffle(7).map(lambda window: (window[:-1], window[-1]))#separates each window into features and label (next/last value)
    dataset = dataset.shuffle(shuffle_buffer).map(
        lambda window: (window[:-1], window[1:])
    )
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, 3], [None, 3]))
    tf.data.Dataset.random(seed=0).take(new_size)
    return dataset
