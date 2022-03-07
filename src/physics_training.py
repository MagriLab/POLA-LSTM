import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import time
import random
import importlib
import datetime
import matplotlib.pyplot as plt
import os
import sys
import warnings
sys.path.append('../')
from lstm.loss import loss_oloop
from lstm.lstm_model import build_open_loop_lstm, load_open_loop_lstm
from lstm.postprocessing import plots
from lstm.preprocessing.data_processing import (create_df_3d,
                                                df_train_valid_test_split,
                                                train_valid_test_split)
from lstm.utils.random_seed import reset_random_seeds

warnings.simplefilter(action="ignore", category=FutureWarning)


plt.rcParams["figure.facecolor"] = "w"

reset_random_seeds()

filepath = "models/pi-model/200"
if not os.path.exists(filepath):
    os.makedirs(filepath)


def forward_diff(y_pred, delta_t=0.01):
    fd = (y_pred[1:, :] - y_pred[:-1, :])/delta_t
    return fd


@tf.function
def pi_loss(y_pred, der, washout=10, reg_weight=0.001):
    mse = tf.keras.losses.MeanSquaredError()
    pi_loss = mse(der[:-1, :], forward_diff(y_pred))
    return pi_loss


mydf = np.genfromtxt("lorenz_data/CSV/Lorenz_trans_001_norm_10000.csv", delimiter=",")
df_train, df_valid, df_test = df_train_valid_test_split(mydf[1:, :])
time_train, time_valid, time_test = train_valid_test_split(mydf[0, :])

der_df = np.genfromtxt("lorenz_data/CSV/Lorenz_trans_001_norm_10000_der.csv", delimiter=",")
der_df_train, der_df_valid, der_df_test = df_train_valid_test_split(der_df[1:, :])

# Windowing
window_size = 100
batch_size = 32
cells = 10
shuffle_buffer_size = df_train.shape[0]
lorenz_dim = 3
train_dataset = create_df_3d(
    df_train.transpose(), window_size, batch_size, shuffle_buffer_size
)

train_dataset = create_df_3d(df_train.transpose(), window_size, batch_size, len(df_train))
valid_dataset = create_df_3d(df_valid.transpose(), window_size, batch_size, 1)
test_dataset = create_df_3d(df_test.transpose(), window_size, batch_size, 1)
der_train_dataset = create_df_3d(der_df_train.transpose(), window_size, batch_size, len(df_train))


for example_inputs, example_labels in train_dataset.take(1):
    input_shape = example_inputs.shape
    output_shape = example_labels.shape
print(input_shape, output_shape)


model = tf.keras.Sequential()
kernel_init = tf.keras.initializers.GlorotUniform(seed=1)
recurrent_init = tf.keras.initializers.Orthogonal(seed=1)
model.add(tf.keras.layers.LSTM(cells, activation="relu", name="LSTM_1"))
model.add(tf.keras.layers.Dense(lorenz_dim, name="Dense_1"))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss=pi_loss, optimizer=optimizer, metrics=["mse"])


@tf.function
def train_step(x_batch_train, y_batch_train, der_y_batch_train):
    with tf.GradientTape() as tape:
        pred = model(x_batch_train, training=True)
        loss_dd = loss_oloop(y_batch_train, pred)
        loss_pi = pi_loss(pred, der_y_batch_train)
        loss_value = loss_dd + loss_pi
    grads = tape.gradient(loss_value, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_dd, loss_pi


epochs = 2000
for epoch in range(epochs):
    start_time = time.time()
    for step, ((x_batch_train, y_batch_train),
               (der_x_batch_train, der_y_batch_train)) in enumerate(zip(train_dataset, der_train_dataset)):
        loss_dd, loss_pi = train_step(x_batch_train, y_batch_train, der_y_batch_train)

    # model.valid_step(valid_dataset)
    print("Time of this epoch: ", time.time() - start_time)
    print("Training loss (for one batch) at step %d: %.4f" % (epoch, float(loss_dd+loss_pi)))
    print(loss_dd, loss_pi)

predictions = plots.plot_closed_loop_lya(
    model,
    epochs,
    time_test,
    df_test,
    n_length=500,
    window_size=100,
    img_filepath="models/pi-model/2000/test-pi" + str(epochs)+".png",
)
plots.plot_phase_space(
    predictions,
    epochs,
    df_test,
    window_size=100,
    img_filepath="models/pi-model/2000/test_phase" + str(epochs)+".png"
)


model_checkpoint = filepath + "model/" + str(epochs)
# Save the weights
model.save_weights(model_checkpoint)
