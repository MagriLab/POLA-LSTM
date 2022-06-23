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
import scipy
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
import torch


sys.path.append('../../')
from lstm.closed_loop_tools_mtm import (create_test_window,
                                        prediction_closed_loop)

from lstm.loss import loss_oloop
from lstm.lstm_model import build_pi_model
from lstm.postprocessing import plots, plots_mtm, prediction_horizon
from lstm.preprocessing.data_processing import (create_df_nd_mtm,
                                                df_train_valid_test_split,
                                                train_valid_test_split)
from lstm.utils.supress_tf_warning import tensorflow_shutup
warnings.simplefilter(action="ignore", category=FutureWarning)
tf.keras.backend.set_floatx('float64')
tensorflow_shutup()

def build_pi_model(cells=100):
    model = tf.keras.Sequential()
    kernel_init = tf.keras.initializers.GlorotUniform(seed=123)
    recurrent_init = tf.keras.initializers.Orthogonal(seed=123)
    model.add(tf.keras.layers.LSTM(cells, activation="tanh", name="LSTM_1", return_sequences=True))
    model.add(tf.keras.layers.Dense(dim, name="Dense_1"))
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, metrics=["mse"], loss=loss_oloop)
    return model


def lstm_step(window_input, h, c, model):

    window_input = tf.reshape(tf.matmul(h, model.layers[1].get_weights()[
                              0]) + model.layers[1].get_weights()[1], shape=(1, 3))
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


def step_and_jac(window_input, h, c, model, Jac_h_c=None, Jac_x_c=None):
    with tf.GradientTape(persistent=True) as tape_window:
        tape_window.watch(window_input)
        with tf.GradientTape(persistent=True) as tape_c:
            tape_c.watch(c)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(h)
                out, h_new, c_new = lstm_step(window_input, h, c, model)
            Jac_h_new_h = tf.reshape(tape.jacobian(h_new, h), shape=(10, 10))
            Jac_c_new_h = tf.reshape(tape.jacobian(c_new, h), shape=(10, 10))
        Jac_c_new_c = tf.reshape(tape_c.jacobian(c_new, c), shape=(10, 10))
        Jac_h_new_c = tf.reshape(tape_c.jacobian(h_new, c), shape=(10, 10))

    Jac = tf.concat([tf.concat([Jac_c_new_c, Jac_c_new_h], axis=1),
                    tf.concat([Jac_h_new_c, Jac_h_new_h], axis=1)], axis=0)

    return Jac, out, h_new, c_new


mydf = np.genfromtxt(
    '/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/lorenz_data/CSV/10000/Lorenz_trans_001_norm_10000.csv',
    delimiter=",").astype(
    np.float64)
df_train, df_valid, df_test = df_train_valid_test_split(mydf[1:, :], train_ratio=0.5, valid_ratio=0.25)
time_train, time_valid, time_test = train_valid_test_split(mydf[0, :], train_ratio=0.5, valid_ratio=0.25)

# Windowing
dim = 3
window_size = 100
batch_size = 256
cells = 10
shuffle_buffer_size = df_train.shape[0]
train_dataset = create_df_nd_mtm(df_train.transpose(), window_size, batch_size, df_train.shape[0])
valid_dataset = create_df_nd_mtm(df_valid.transpose(), window_size, batch_size, 1)
test_dataset = create_df_nd_mtm(df_test.transpose(), window_size, batch_size, 1)

model_path = '/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/models/lorenz/euler/10000-many-diff_loss/'
img_filepath = model_path + "images/time_dev/"

if not os.path.exists(img_filepath):
    os.makedirs(img_filepath)
epochs = 10000

model = build_pi_model(cells)
model.load_weights(model_path + "model/" + str(epochs) + "/weights").expect_partial()
n_length = 150
lyapunov_time, prediction = prediction_closed_loop(
    model, time_test, df_test, n_length, window_size=window_size, c_lyapunov=0.9
)

test_window = create_test_window(df_test, window_size=window_size)
N_units = 10
dt        = 0.01  # time step
t_lyap    = 0.9**(-1)   
norm_time = 1
N=100

Ntransient = int(N/10)
N_test = N - Ntransient
print('N',N,'Ntran',Ntransient,'N_test',N_test)
Ttot  = np.arange(int(N_test/norm_time)) * dt * norm_time

N_test_norm = int(N_test/norm_time)
print('N_test_norm',N_test_norm)
# Lyapunov Exponents timeseries
LE = np.zeros((N_test_norm, dim))
# Q matrix recorded in time    
QQ_t = np.zeros((N_units+N_units, dim, N_test_norm))
# R matrix recorded in time
RR_t = np.zeros((dim, dim, N_test_norm))
window = test_window[:, 0, :]    
h = tf.Variable(model.layers[0].get_initial_state(test_window)[0], trainable=False)
c = tf.Variable(model.layers[0].get_initial_state(test_window)[1], trainable=False)
pred = np.zeros(shape=(N, 3))
pred[0, :] = window
delta = scipy.linalg.orth(np.random.rand(N_units+N_units,dim))
Q, R = np.linalg.qr(delta)
delta  = Q[:,:dim]    

jacobian, window, h, c  = step_and_jac(window, h, c, model)   
for i in np.arange(1,Ntransient):
    if i < window_size:
        window = test_window[:, i, :]   
    jacobian, window, h, c = step_and_jac(window, h, c, model)
    pred[i, :] = window
    delta = np.matmul(jacobian, delta)

    if i % norm_time == 0:
        Q, R   = np.linalg.qr(delta)
        delta = Q[:,:dim]

indx = 0
for i in np.arange(Ntransient,N):
    if i < window_size:
        window = test_window[:, i, :]
    jacobian, window, h, c = step_and_jac(window, h, c, model)
    pred[i, :] = window
    jacobian = tf.transpose(jacobian)
    delta = np.matmul(jacobian, delta)
    if i % norm_time == 0:
        Q, R   = np.linalg.qr(delta)
        delta = Q[:,:dim]

        RR_t[:,:,indx] = R
        QQ_t[:,:,indx] = Q
        LE[indx] = np.abs(np.diag(R[:dim,:dim])) 
        
        indx+=1
        
        if i%100==0:
            print('Inside closed loop i=',i)
LEs = np.cumsum(np.log(LE[1:]),axis=0)/ np.tile(Ttot[1:],(dim,1)).T
print('Lyapunov exponents: ',LEs[-1])