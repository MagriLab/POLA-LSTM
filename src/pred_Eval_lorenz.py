import argparse
import datetime
import importlib
import os
import random
import sys
import time
import warnings
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import math
import seaborn as sns
import pandas as pd
sys.path.append('../')
from lstm.utils.random_seed import reset_random_seeds
from lstm.utils.config import generate_config
from lstm.preprocessing.data_processing import (create_df_nd_mtm,
                                                df_train_valid_test_split,
                                                train_valid_test_split)
from lstm.postprocessing.loss_saver import loss_arr_to_tensorboard
from lstm.postprocessing import plots_mtm, prediction_horizon, plots
from lstm.lstm_model import build_pi_model
from lstm.lorenz import fixpoints
from lstm.loss import loss_oloop, norm_loss_pi_many
from lstm.closed_loop_tools_mto import append_label_to_batch
from lstm.closed_loop_tools_mtm import append_label_to_window, split_window_label, create_test_window, prediction_closed_loop
from lstm.utils.supress_tf_warning import tensorflow_shutup
tensorflow_shutup()
warnings.simplefilter(action="ignore", category=FutureWarning)
tf.keras.backend.set_floatx('float64')

def build_model(cells=100):
    model = tf.keras.Sequential()
    kernel_init = tf.keras.initializers.GlorotUniform(seed=123)
    recurrent_init = tf.keras.initializers.Orthogonal(seed=123)
    model.add(tf.keras.layers.LSTM(cells, activation="tanh", name="LSTM_1", return_sequences=True))
    model.add(tf.keras.layers.Dense(lorenz_dim, name="Dense_1"))
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, metrics=["mse"], loss=loss_oloop)
    return model


delta_t = 0.1
c_lyapunov=0.033791
delta_t_lya = delta_t*c_lyapunov
N_1_lya = math.ceil(1/delta_t/c_lyapunov)

# Windowing
lorenz_dim=6
window_size = 100
batch_size = 128
cells = 10

def nrmse(pred, df_test, window_size=100, n_length=None):
    """ normalized root mean square error """
    if n_length==None:
        n_length = len(pred)
    std = np.std(df_test[:, window_size:window_size+n_length])
    diff = pred[:n_length, :] - df_test[:, window_size:window_size+n_length].T
    return np.sqrt(np.mean(diff**2/std, axis=1))
def vpt(pred, df_test, threshold, window_size=100):
    """valid prediction time"""
    nrmse_vec = nrmse(pred, df_test, window_size=window_size)
    for i in range(1, len(pred)):
        if nrmse_vec[i] > threshold:
            return i-1
    return len(pred)


lambda_folder = [ 'atomic-sweep-1' , 'lunar-sweep-2', 'genial-sweep-3', 'woven-sweep-4' ]

mydf = np.genfromtxt("/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/cdv_data/CSV/euler_27500_trans_snr80.csv", delimiter=",").astype(np.float64)
time = mydf[0, :]
mydf = mydf[1:, :]
df_train, df_valid, df_test = df_train_valid_test_split(mydf, train_ratio=0.5, valid_ratio=0.25)
time_train, time_valid, time_test = train_valid_test_split(time, train_ratio=0.5, valid_ratio=0.25)
vpt_threshold = 0.4

for i, lambda_val in enumerate(lambda_folder):
    print('---: ' + lambda_val  + '---')
    model_path =  "/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/models/cdv/27500-snr80/"  + lambda_val +"/" 
    epoch = 5000

    model = build_model(10)
    model.load_weights(model_path + "model/"+ str(epoch) +"/weights").expect_partial()
    
    n_length = 2000

    pred_lt_test = np.zeros(((int((df_train.shape[1]-window_size-n_length)/N_1_lya)),))
    pred_lt_valid = np.zeros(((int((df_train.shape[1]-window_size-n_length)/N_1_lya)),))
    
    for i in range(int((df_valid.shape[1]-window_size-n_length)/N_1_lya)):
        lyapunov_time, prediction = prediction_closed_loop(
        model, time_valid, df_valid[:, i*N_1_lya:], n_length, window_size=window_size, c_lyapunov=c_lyapunov
        )
        vpt_valid = lyapunov_time[vpt(prediction, df_valid[:, i*N_1_lya:], threshold=vpt_threshold, window_size=window_size)]
        pred_lt_valid[i] = vpt_valid
        print("VPT Validation", vpt_valid)

    for i in range(int((df_test.shape[1]-window_size-n_length)/N_1_lya)):
        lyapunov_time, prediction = prediction_closed_loop(
        model, time_test, df_test[:, i*N_1_lya:], n_length, window_size=window_size, c_lyapunov=c_lyapunov
        )
        vpt_test = lyapunov_time[vpt(prediction, df_test[:, i*N_1_lya:], threshold=vpt_threshold, window_size=window_size)]
        pred_lt_test[i] = vpt_test
    print('---LAMBDA: '+lambda_val+' ---')
    print('TEST VPT quantiles:', np.quantile(pred_lt_test,.75), np.median(pred_lt_test), np.quantile(pred_lt_test,.25))
    print('TEST VPT mean     :', np.mean(pred_lt_test))
    print('VALID VPT quantiles:', np.quantile(pred_lt_valid,.75), np.median(pred_lt_valid), np.quantile(pred_lt_valid,.25))
    print('VALID VPT mean     :', np.mean(pred_lt_valid))
    df = pd.DataFrame(list(zip(pred_lt_valid, pred_lt_test)), columns =['Valid', 'Test'])
    df.to_csv(model_path+"recycle_val.csv", encoding='utf-8', index=False)
    print("Saved at ", model_path+"recycle_val.csv")