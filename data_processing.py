import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tensorflow_datasets as tfds
import tensorflow as tf


def create_training_split(df, ratio=0.7):
    len_df = len(df)
    train = np.array(df[0:int(len_df*ratio)])
    test = np.array(df[int(len_df*ratio):])
    return train, test

def df_training_split(df, ratio=0.7):
    len_df_col = df.shape[1]
    train = np.array(df[:, 0:int(len_df_col*ratio)])
    test = np.array(df[:, int(len_df_col*ratio):])
    return train, test

def create_df_3d(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(size=window_size+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    #dataset = dataset.shuffle(7).map(lambda window: (window[:-1], window[-1]))#separates each window into features and label (next/last value)
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, 3],[None]))
    return dataset

def create_window_closed_loop(test_data, iteration, pred=np.array([])):
    if iteration == 0:
        print("no network prediction yet")
        return test_data[:50, :].reshape(1, 50, 3)
    if iteration < 50:
        n_pred = pred.shape[0]
        idx_test_entries = iteration + 50 - n_pred  # end index of entries from the test data, 
        test_data = test_data[iteration: idx_test_entries , :]
        return np.append(test_data, pred, axis=0).reshape(1, 50, 3)
    else:
        return pred[-50:,:].reshape(1, 50, 3)

def add_new_pred(pred_old, pred_new):
    return np.append(pred_old,pred_new, axis=0)

def compute_lyapunov_time_arr(time_vector, c_lyapunov=0.90566):
    t_lyapunov = 1/c_lyapunov
    lyapunov_time = (time_vector[50:]-time_vector[0])/t_lyapunov
    return lyapunov_time