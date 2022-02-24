import os
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import sys
import matplotlib.pyplot as plt
plt.rcParams["figure.facecolor"] = "w"
import datetime
import random
import tensorflow as tf
import numpy as np
import time
import tensorflow_datasets as tfds
import importlib


sys.path.append('../')
from lstm.preprocessing.data_processing import create_df_3d, train_valid_test_split, df_train_valid_test_split
from lstm.lstm_model import build_open_loop_lstm, load_open_loop_lstm
import lstm.postprocessing.plots


def reset_random_seeds():
    os.environ["PYTHONHASHSEED"] = str(2)
    tf.random.set_seed(2)
    np.random.seed(2)
    random.seed(2)


# Data imports
mydf = np.genfromtxt("Lorenz_Data/CSV/Lorenz_trans_001_norm_100000.csv", delimiter=",")
time = mydf[0, :]
mydf = mydf[1:, :]
df_train, df_valid, df_test = df_train_valid_test_split(mydf)
time_train, time_valid, time_test = train_valid_test_split(time)
x_train, x_valid, x_test = train_valid_test_split(mydf[0, :])
y_train, y_valid, y_test = train_valid_test_split(mydf[1, :])
z_train, z_valid, z_test = train_valid_test_split(mydf[2, :])

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

model = lstm_model.build_open_loop_lstm(cells)

img_filepath = (
    "models/"
    + str(window_size)
    + "_window_10LSTM_trans/Images/LSTM_0_washout/run_2402/t_1000_100000/"
)
model_checkpoint = "/Users/eo821/Documents/PhD_research/Lorenz_LSTM/models/80_window_10LSTM_trans/Images/LSTM_0_washout/run_2402/t_100_100000/model/110/"
model.load_weights(model_checkpoint)
log_dir = (
    "models/"
    + str(window_size)
    + "_window_10LSTM_trans/logs/fit/washout0/run_2402/t_1000_100000/"
    + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)
early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=10, restore_best_weights=True
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
n_epochs = 150

print("--- Continue Open Loop Training ---")
history = model.fit(
    train_dataset,
    epochs=n_epochs,
    batch_size=batch_size,
    initial_epoch=110,
    validation_data=valid_dataset,
    verbose=1,
    callbacks=[tensorboard_callback],  # , early_stop_callback],
)


if not os.path.exists(img_filepath):
    os.makedirs(img_filepath)
lya_filepath = img_filepath + "oloop_" + str(history.params["epochs"]) + ".png"


predictions = plots.plot_closed_loop_lya(
    model,
    history.params["epochs"],
    time_test,
    df_test,
    n_length=500,
    window_size=window_size,
    img_filepath=lya_filepath,
)
phase_filepath = img_filepath + "phase_oloop_" + str(history.params["epochs"]) + ".png"
plots.plot_phase_space(
    predictions,
    history.params["epochs"],
    df_test,
    img_filepath=phase_filepath,
    window_size=window_size,
)
n_epochs_old = n_epochs

for i in range(0, 10):
    print("Open Loop Iteration is ", i)
    n_epochs = n_epochs_old + 50
    history = model.fit(
        train_dataset,
        epochs=n_epochs,
        initial_epoch=n_epochs_old,
        batch_size=32,
        validation_data=valid_dataset,
        callbacks=[tensorboard_callback],
        verbose=2,
    )

    n_epochs_old = n_epochs
    lya_filepath = img_filepath + "oloop" + str(history.params["epochs"]) + ".png"
    predictions = plots.plot_closed_loop_lya(
        model,
        history.params["epochs"],
        time_test,
        df_test,
        window_size=window_size,
        n_length=500,
        img_filepath=lya_filepath,
    )
    phase_filepath = (
        img_filepath + "phase_oloop" + str(history.params["epochs"]) + ".png"
    )
    plots.plot_phase_space(
        predictions,
        history.params["epochs"],
        df_test,
        img_filepath=phase_filepath,
        window_size=window_size,
    )
    model_checkpoint = img_filepath + "model/" + str(n_epochs)
    # Save the weights
    model.save_weights(model_checkpoint)
