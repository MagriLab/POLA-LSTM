import os
import warnings
import seaborn as sns
import datetime
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tensorflow_datasets as tfds
import plot_file
import importlib
import lstm_model
from lstm_model import build_open_loop_lstm, load_open_loop_lstm
from data_processing import (
    create_training_split,
    df_training_split,
    create_df_3d,
    create_window_closed_loop,
    add_new_pred,
    compute_lyapunov_time_arr,
    train_valid_test_split,
    df_train_valid_test_split,
)

plt.rcParams["figure.facecolor"] = "w"
warnings.simplefilter(action="ignore", category=FutureWarning)


def reset_random_seeds():
    os.environ["PYTHONHASHSEED"] = str(2)
    tf.random.set_seed(2)
    np.random.seed(2)
    random.seed(2)


# Tensorboard
# %load_ext tensorboard


# Data imports
mydf = np.genfromtxt("CSV/Lorenz_trans_001_norm_100000.csv", delimiter=",")
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
shuffle_buffer_size = df_train.shape[0]
train_dataset = create_df_3d(
    df_train.transpose(), window_size, batch_size, shuffle_buffer_size
)
valid_dataset = create_df_3d(df_valid.transpose(), window_size, batch_size, 1)
test_dataset = create_df_3d(df_test.transpose(), window_size, batch_size, 1)

reset_random_seeds()
model = lstm_model.build_open_loop_lstm(cells=10)
log_dir = (
    "models/100_window_10LSTM_trans/logs/fit/washout0/run_2002/long_lya/"
    + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)
early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=10, restore_best_weights=True
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
n_epochs = 10

print("--- Begin Open Loop Training ---")
history = model.fit(
    train_dataset,
    epochs=n_epochs,
    batch_size=batch_size,
    validation_data=valid_dataset,
    verbose=1,
    callbacks=[tensorboard_callback],  # , early_stop_callback],
)


img_filepath = "models/100_window_10LSTM_trans/Images/LSTM_0_washout/run_2002/long_lya/"

if not os.path.exists(img_filepath):
    os.makedirs(img_filepath)
lya_filepath = img_filepath + "oloop_" + str(history.params["epochs"]) + ".png"


predictions = plot_file.plot_closed_loop_lya(
    model,
    history.params["epochs"],
    time_test,
    df_test,
    n_length=1900,
    window_size=window_size,
    img_filepath=lya_filepath,
)
phase_filepath = img_filepath + "phase_oloop_" + str(history.params["epochs"]) + ".png"
plot_file.plot_phase_space(
    predictions,
    history.params["epochs"],
    df_test,
    img_filepath=phase_filepath,
    window_size=window_size,
)
n_epochs_old = n_epochs
model_checkpoint = img_filepath + "model/"
# Save the weights
model.save_weights(model_checkpoint)
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
    predictions = plot_file.plot_closed_loop_lya(
        model,
        history.params["epochs"],
        time_test,
        df_test,
        window_size=window_size,
        n_length=1900,
        img_filepath=lya_filepath,
    )
    phase_filepath = (
        img_filepath + "phase_oloop" + str(history.params["epochs"]) + ".png"
    )
    plot_file.plot_phase_space(
        predictions,
        history.params["epochs"],
        df_test,
        img_filepath=phase_filepath,
        window_size=window_size,
    )
model_checkpoint = img_filepath + "model/"
# Save the weights
model.save_weights(model_checkpoint)

# print("--- Open Loop Training Complete ---")
# print("--- Begin Closed Loop Training ---")
# n_epochs_old = n_epochs
# new_learning_rate = 0.0001
# model.optimizer.lr.assign(new_learning_rate)
# for i in range(0, 5):
#     print("Closed Loop Iteration is ", i)
#     n_epochs = n_epochs_old + 200
#     n_batches = 100
#     cloop_size = 32 * n_batches
#     test_window, labels, idx = lstm_model.select_random_batches_with_label(
#         df_train.transpose(), n_batches
#     )
#     predictions = model.predict(
#         np.array(test_window).reshape(cloop_size, window_size, 3)
#     )

#     cloop_windows, cloop_label = lstm_model.add_cloop_prediction(
#         df_train.transpose(), idx, predictions
#     )
#     print(cloop_windows.shape)
#     history = model.fit(
#         cloop_windows.reshape(cloop_size, window_size, 3),
#         cloop_label,
#         epochs=n_epochs,
#         initial_epoch=n_epochs_old,
#         batch_size=32,
#         validation_data=valid_dataset,
#         callbacks=[tensorboard_callback],
#         verbose=2,
#     )

#     n_epochs_old = n_epochs
#     lya_filepath = img_filepath + "cloop_" + str(history.params["epochs"]) + ".png"
#     predictions = plot_file.plot_closed_loop_lya(
#         model,
#         history.params["epochs"],
#         time_test,
#         df_test,
#         n_length=500,
#           window_size=window_size,
#         img_filepath=lya_filepath,
#     )
#     phase_filepath = (
#         img_filepath + "phase_cloop" + str(history.params["epochs"]) + ".png"
#     )
#     plot_file.plot_phase_space(
#         predictions, history.params["epochs"], df_test, img_filepath=phase_filepath,
# window_size=window_size
#     )

# print("--- Closed Loop Training Complete ---")
