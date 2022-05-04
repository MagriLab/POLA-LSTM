
import os
import sys
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import matplotlib.pyplot as plt

plt.rcParams["figure.facecolor"] = "w"
import datetime
import importlib
import random
import time

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

sys.path.append('../')
from lstm.lstm_model import build_open_loop_lstm, load_open_loop_lstm
from lstm.postprocessing import plots
from lstm.preprocessing.data_processing import (create_df_3d,
                                                df_train_valid_test_split,
                                                train_valid_test_split)
from lstm.utils.random_seed import reset_random_seeds


def return_idx_reoc(input_array):
    idx_sort = np.argsort(input_array)
    sorted_input_arrayy = input_array[idx_sort]
    vals, idx_start, count = np.unique(sorted_input_arrayy, return_counts=True, return_index=True)
    res = np.split(idx_sort, idx_start[1:])    #filter them with respect to their size, keeping only items occurring more than once
    vals = vals[count > 1]
    res_filter = filter(lambda x: x.size > 1, res)
    print(res[res_filter])
    return res


    
# Data imports
mydf = np.genfromtxt("Lorenz_Data/CSV/Lorenz_trans_01_norm_100000_.csv", delimiter=",").astype(np.float64)
time_train, time_valid, time_test = train_valid_test_split(mydf[0, :])
df_train, df_valid, df_test = df_train_valid_test_split(mydf[1:, :])


# Windowing
window_size = 100
batch_size = 32
cells = 10
shuffle_buffer_size = df_train.shape[0]
train_dataset = create_df_3d(
    df_train.transpose(), window_size, batch_size, shuffle_buffer_size
)
valid_dataset = create_df_3d(df_valid.transpose(), window_size, batch_size, 1)
test_dataset = create_df_3d(df_test.transpose(), window_size, batch_size, 1)

reset_random_seeds()

model = build_open_loop_lstm(cells)


n_epochs = 400
model_checkpoint = "/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/models/100_window_10LSTM_trans/Images/LSTM_0_washout/run_2102/long_lya100000/model/" + str(n_epochs)
model.load_weights(model_checkpoint)


print("Model loaded", df_test.shape)
img_filepath = "/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/models/100_window_10LSTM_trans/Images/LSTM_0_washout/run_2102/"
if not os.path.exists(img_filepath):
    os.makedirs(img_filepath)
lya_filepath = img_filepath + "oloop_" +str(n_epochs) + "_wide.png"


predictions = plots.plot_closed_loop_lya(
    model,
    n_epochs,
    time_test,
    df_test,
    n_length=5,
    window_size=window_size,
    img_filepath=lya_filepath,
)