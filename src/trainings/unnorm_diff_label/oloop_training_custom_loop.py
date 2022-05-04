
import argparse
import datetime
import importlib
import os
import random
import sys
import time
import warnings
from pathlib import Path
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import einops
sys.path.append('../../..')

from lstm.closed_loop_tools_mto import compute_lyapunov_time_arr
from lstm.loss import loss_oloop, norm_pi_loss, norm_pi_loss_two_step
from lstm.lstm_model import build_pi_model
from lstm.postprocessing import plots
from lstm.postprocessing.tensorboard_converter import loss_arr_to_tensorboard
from lstm.preprocessing.data_processing import (df_train_valid_test_split,
                                                train_valid_test_split)
from lstm.utils.config import generate_config
from lstm.utils.random_seed import reset_random_seeds


tf.keras.backend.set_floatx('float64')
warnings.simplefilter(action="ignore", category=FutureWarning)


plt.rcParams["figure.facecolor"] = "w"

def fixpoints(total_points=5000, beta=2.667, rho=28, sigma=10, unnorm=False):
    if unnorm==True:
        x_max = 1
        y_max = 1
        z_max = 1
    elif total_points == 100000:
        x_max = 19.62036351364186
        y_max = 27.31708182056948
        z_max = 48.05263683702385
    elif total_points == 10000:
        x_max = 19.61996107472211
        y_max = 27.317071267968995
        z_max = 48.05315303164703
    elif total_points == 5000:
        x_max = 19.619508366918392 
        y_max = 27.317051197038307
        z_max = 48.05371246231375


    x_fix = np.sqrt(beta*(rho-1))
    y_fix = np.sqrt(beta*(rho-1))
    z_fix = rho-1
    return x_fix/x_max, y_fix/y_max, z_fix/z_max


x_fix, y_fix, z_fix = fixpoints(total_points=10000, unnorm=True)

def plot_closed_loop_lya(
    model,
    n_epochs,
    time_test,
    df_test,
    img_filepath=None,
    n_length=6000,
    window_size=50,
):
    lyapunov_time, pred_closed_loop = prediction_closed_loop(
        model, time_test, df_test, n_length, window_size=window_size
    )
    test_time_end = len(pred_closed_loop)

    fig, axs = plt.subplots(3, 2, sharey=True, facecolor="white")  # , figsize=(15, 14))
    fig.suptitle("Closed Loop LSTM Prediction at epoch " + str(n_epochs))
    axs[0, 0].plot(
        lyapunov_time[:test_time_end],
        df_test[0, window_size: window_size + test_time_end],
        label="True Data",
    )
    axs[0, 0].plot(
        lyapunov_time[:test_time_end],
        pred_closed_loop[:, 0],
        "--",
        label="RNN Prediction",
    )
    axs[0, 0].axhline(y=x_fix, color="lightcoral", linestyle=":")
    axs[0, 0].axhline(y=-x_fix, color="lightcoral", linestyle=":")
    axs[0, 0].set_ylabel("x")
    sns.kdeplot(
        df_test[0, window_size:test_time_end],
        vertical=True,
        color="tab:blue",
        ax=axs[0, 1],
    )
    sns.kdeplot(pred_closed_loop[:, 0], vertical=True, color="tab:orange", ax=axs[0, 1])
    axs[1, 0].plot(
        lyapunov_time[:test_time_end],
        df_test[1, window_size: window_size + test_time_end],
        label="data",
    )
    axs[1, 0].plot(
        lyapunov_time[:test_time_end],
        pred_closed_loop[:, 1],
        "--",
        label="RNN prediction on test data",
    )
    axs[1, 0].set_ylabel("y")
    axs[1, 0].axhline(y=y_fix, color="lightcoral", linestyle=":")
    axs[1, 0].axhline(y=-y_fix, color="lightcoral", linestyle=":")
    sns.kdeplot(
        df_test[1, window_size:test_time_end],
        vertical=True,
        color="tab:blue",
        ax=axs[1, 1],
    )
    sns.kdeplot(pred_closed_loop[:, 1], vertical=True, color="tab:orange", ax=axs[1, 1])
    axs[2, 0].plot(
        lyapunov_time[:test_time_end],
        df_test[2, window_size: window_size + test_time_end],
        label="Numerical Solution",
    )
    axs[2, 0].plot(
        lyapunov_time[:test_time_end],
        pred_closed_loop[:, 2],
        "--",
        label="LSTM prediction",
    )
    axs[2, 0].set_ylabel("z")
    axs[2, 0].axhline(y=z_fix, color="lightcoral", linestyle=":", label="Fixpoint")
    sns.kdeplot(
        df_test[2, 0:test_time_end], vertical=True, color="tab:blue", ax=axs[2, 1]
    )
    sns.kdeplot(pred_closed_loop[:, 2], vertical=True, color="tab:orange", ax=axs[2, 1])
    axs[2, 0].legend(loc="center left", bbox_to_anchor=(2.3, 2.0))
    axs[0, 0].set_xticklabels([])
    axs[1, 0].set_xticklabels([])
    axs[0, 1].get_shared_x_axes().join(axs[0, 1], axs[1, 1], axs[2, 1])
    axs[1, 1].get_shared_x_axes().join(axs[0, 1], axs[1, 1], axs[2, 1])
    axs[2, 1].get_shared_x_axes().join(axs[0, 1], axs[1, 1], axs[2, 1])
    axs[0, 1].set_xticklabels([])
    axs[1, 1].set_xticklabels([])
    if img_filepath != None:
        fig.savefig(img_filepath, dpi=200, facecolor="w", bbox_inches="tight")
        print("Closed Loop prediction saved at ", img_filepath)
    return pred_closed_loop


def plot_phase_space(predictions, n_epochs, df_test, img_filepath=None, window_size=50):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig.suptitle("Phase Space Comparison at Epoch " + str(n_epochs))
    test_time_end = len(predictions)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(
        df_test[0, window_size: window_size + test_time_end],
        df_test[1, window_size: window_size + test_time_end],
        df_test[2, window_size: window_size + test_time_end],
        color="tab:blue",
        alpha=0.7,
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    # ax1.set_xlim(-1.1, 1.1)
    # ax1.set_ylim(-1.1, 1.1)
    # ax1.set_zlim(-1.1, 1.1)
    ax1.set_title("Numerical Solution")
    ax1.plot(x_fix, y_fix, z_fix, "x", color="tab:red", alpha=0.7)
    ax1.plot(-x_fix, -y_fix, z_fix, "x", color="tab:red", alpha=0.7)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot(
        predictions[:, 0],
        predictions[:, 1],
        predictions[:, 2],
        color="tab:orange",
        alpha=0.7,
    )
    ax2.plot(x_fix, y_fix, z_fix, "x", color="tab:red", alpha=0.7)
    ax2.plot(-x_fix, -y_fix, z_fix, "x", color="tab:red", alpha=0.7)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_title("LSTM Prediction")
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_zlim(ax1.get_zlim())
    if img_filepath != None:
        fig.savefig(img_filepath, dpi=200, facecolor="w", bbox_inches="tight")
        print("Phase Space prediction saved at ", img_filepath)

lorenz_dim=3
def create_df_3d_diff(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(size=window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    # dataset = dataset.shuffle(7).map(lambda window: (window[:-1], window[-1]))#separates each window into features and label (next/last value)
    dataset = dataset.shuffle(shuffle_buffer).map(
        lambda window: (window[:-1], (window[-1]-window[-2]))
    )
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, 3], [None]))
    return dataset

def append_label_to_window(window, label):
    corr_label = window[:, -1,:]+ label
    return tf.concat((window, einops.rearrange(corr_label,"i j -> 1 i j")), axis=1)

def split_window_label(window_label_tensor, window_size=100, batch_size=32):
    window = window_label_tensor[:,-(window_size):, :] 
    return window

def create_test_window(df_test, window_size=100):
    test_window = tf.convert_to_tensor(df_test[:, :window_size].T)
    test_window = einops.rearrange(test_window, "i j -> 1 i j")
    return test_window

def prediction_closed_loop(model, time_test, df_test, n_length, window_size=50):
    lyapunov_time = compute_lyapunov_time_arr(
        time_test, window_size=window_size)
    predictions = np.zeros((n_length, lorenz_dim))
    test_window = create_test_window(df_test, window_size=window_size)
    for i in range(len(predictions)):
        pred = model.predict(test_window)
        predictions[i, :] = test_window[:, -1, :] + pred
        test_window = split_window_label(append_label_to_window(test_window, pred))
    return lyapunov_time, predictions

def append_label_to_batch(window, label):
    corr_label = window[:, -1,:]+ label
    return tf.concat((window, einops.rearrange(corr_label,"i j -> i 1 j")), axis=1)




def run_lstm(args: argparse.Namespace):

    reset_random_seeds()

    filepath = args.data_path
    if not os.path.exists(filepath / "images"):
        os.makedirs(filepath / "images")

    mydf = np.genfromtxt(args.config_path, delimiter=",").astype(np.float64)
    df_train, df_valid, df_test = df_train_valid_test_split(mydf[1:, :], train_ratio=0.3334, valid_ratio=0.3334)
    time_train, time_valid, time_test = train_valid_test_split(mydf[0, :], train_ratio=0.3334, valid_ratio=0.3334)

    # Windowing
    lorenz_dim = 3
    train_dataset = create_df_3d_diff(df_train.transpose(), args.window_size, args.batch_size, df_train.shape[0])
    valid_dataset = create_df_3d_diff(df_valid.transpose(), args.window_size, args.batch_size, 1)
    test_dataset = create_df_3d_diff(df_test.transpose(), args.window_size, args.batch_size, 1)

    model = build_pi_model(args.n_cells)
    # model.load_weights(args.input_data_path)

    def decayed_learning_rate(step):
        initial_learning_rate = 0.001
        decay_steps = 1000
        decay_rate = 0.75
        # careful here! step includes batch steps in the tf framework
        return initial_learning_rate * decay_rate ** (step / decay_steps)

    @tf.function
    def train_step_pi(x_batch_train, y_batch_train, weight=1, normalised=True):
        with tf.GradientTape() as tape:
            one_step_pred = model(x_batch_train, training=True)
            new_batch = split_window_label(append_label_to_batch(x_batch_train, one_step_pred))
            two_step_pred = model(new_batch, training=True)
            loss_dd = loss_oloop(y_batch_train, one_step_pred)
            loss_pi = norm_pi_loss_two_step(one_step_pred, two_step_pred, norm=normalised)

            loss_value = loss_dd + weight*loss_pi
        grads = tape.gradient(loss_value, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_dd, loss_pi

    @tf.function
    def valid_step_pi(x_batch_valid, y_batch_valid, normalised=True):
        val_logit = model(x_batch_valid, training=False)
        loss_dd = loss_oloop(y_batch_valid, val_logit)
        new_batch = split_window_label(append_label_to_batch(x_batch_valid, val_logit))
        two_step_pred = model(new_batch, training=False)
        loss_pi = norm_pi_loss_two_step(val_logit, two_step_pred, norm=normalised)
        return loss_dd, loss_pi

    train_loss_dd_tracker = np.array([])
    train_loss_pi_tracker = np.array([])
    valid_loss_dd_tracker = np.array([])
    valid_loss_pi_tracker = np.array([])
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=args.learning_rate, decay_steps=1000, decay_rate=0.5)
    # tf.keras.backend.set_value(model.optimizer.learning_rate, lr_schedule)

    for epoch in range(args.n_epochs+1):
        model.optimizer.learning_rate = decayed_learning_rate(epoch)
        start_time = time.time()
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_dd, loss_pi = train_step_pi(x_batch_train, y_batch_train,
                                             weight=args.physics_weighing, normalised=args.normalised)
        train_loss_dd_tracker = np.append(train_loss_dd_tracker, loss_dd)
        train_loss_pi_tracker = np.append(train_loss_pi_tracker, loss_pi)

        print("Epoch: %d, Time: %.1fs , Batch: %d" % (epoch, time.time() - start_time, step))
        print("TRAINING: Data-driven loss: %4E; Physics-informed loss at epoch: %.4E" % (loss_dd, loss_pi))

        valid_loss_dd = 0
        valid_loss_pi = 0
        for val_step, (x_batch_valid, y_batch_valid) in enumerate(valid_dataset):
            val_loss_dd, val_loss_pi = valid_step_pi(x_batch_valid, y_batch_valid, normalised=args.normalised)
            valid_loss_dd += val_loss_dd
            valid_loss_pi += val_loss_pi
        valid_loss_dd_tracker = np.append(valid_loss_dd_tracker, valid_loss_dd/val_step)
        valid_loss_pi_tracker = np.append(valid_loss_pi_tracker, valid_loss_pi/val_step)
        print("VALIDATION: Data-driven loss: %4E; Physics-informed loss at epoch: %.4E" %
              (valid_loss_dd / val_step, valid_loss_pi / val_step))

        if epoch % args.epoch_steps == 0:
            print("LEARNING RATE:%.2e" % model.optimizer.learning_rate)
            predictions = plot_closed_loop_lya(
                model,
                epoch,
                time_test,
                df_test,
                n_length=500,
                window_size=args.window_size,
                img_filepath=filepath / "images" / f"pred_{epoch}.png",
            )
            plot_phase_space(
                predictions,
                epoch,
                df_test,
                window_size=args.window_size,
                img_filepath=filepath / "images" / f"phase_{epoch}.png",
            )

            model_checkpoint = filepath / "model" / f"{epoch}" / "weights"
            model.save_weights(model_checkpoint)
            logs_checkpoint = filepath / "logs"
    if not os.path.exists(logs_checkpoint):
        os.makedirs(logs_checkpoint)
    np.savetxt(logs_checkpoint/f"training_loss_dd.txt", train_loss_dd_tracker)
    np.savetxt(logs_checkpoint/f"training_loss_pi.txt", train_loss_pi_tracker)
    np.savetxt(logs_checkpoint/f"valid_loss_dd.txt", valid_loss_dd_tracker)
    np.savetxt(logs_checkpoint/f"valid_loss_pi.txt", valid_loss_pi_tracker)
    loss_arr_to_tensorboard(logs_checkpoint, train_loss_dd_tracker, train_loss_pi_tracker,
                            valid_loss_dd_tracker, valid_loss_pi_tracker)


parser = argparse.ArgumentParser(description='Open Loop')
# arguments for configuration parameters
parser.add_argument('--n_epochs', type=int, default=10000)
parser.add_argument('--epoch_steps', type=int, default=500)
parser.add_argument('--epoch_iter', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_cells', type=int, default=10)
parser.add_argument('--oloop_train', default=True, action='store_true')
parser.add_argument('--cloop_train', default=False, action='store_true')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--activation', type=str, default='Tanh')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--l2_regularisation', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--early_stop', default=False, action='store_true')
parser.add_argument('--early_stop_patience', type=int, default=10)
parser.add_argument('--physics_informed', default=True, action='store_true')
parser.add_argument('--physics_weighing', type=float, default=0)

parser.add_argument('--normalised', default=False, action='store_true')
parser.add_argument('--t_0', type=int, default=0)
parser.add_argument('--t_trans', type=int, default=20)
parser.add_argument('--t_end', type=int, default=100)
parser.add_argument('--delta_t', type=int, default=0.01)
parser.add_argument('--total_n', type=float, default=8000)
parser.add_argument('--window_size', type=int, default=100)
parser.add_argument('--hidden_units', type=int, default=10)
parser.add_argument('--signal_noise_ratio', type=int, default=0)
# arguments to define paths
# parser.add_argument( '--experiment_path', type=Path, required=True)
# parser.add_argument('-idp', '--input_data_path', type=Path, required=True)
# parser.add_argument('--log-board_path', type=Path, required=True)
parser.add_argument('-dp', '--data_path', type=Path, required=True)
parser.add_argument('-cp', '--config_path', type=Path, required=True)

parsed_args = parser.parse_args()


yaml_config_path = parsed_args.data_path / f'config.yml'


generate_config(yaml_config_path, parsed_args)

run_lstm(parsed_args)
# python oloop_training_custom_loop.py -dp models/euler/10000/ -cp ../../lorenz_data/CSV/10000/trans_euler_10000.csv 
# -idp /Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/trainings/unnorm_diff_label/models/euler/10000/model/10000/weights
