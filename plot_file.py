import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams["figure.facecolor"] = "white"
import time
import tensorflow_datasets as tfds
import seaborn as sns

from data_processing import (
    create_training_split,
    df_training_split,
    create_df_3d,
    create_window_closed_loop,
    add_new_pred,
    compute_lyapunov_time_arr,
    train_valid_test_split,
)
from lstm_model import build_open_loop_lstm, load_open_loop_lstm, prediction_closed_loop


def plot_closed_loop_lya(
    model, n_epochs, time_test, df_test, img_filepath=None, n_length=6000
):
    lyapunov_time, pred_closed_loop = prediction_closed_loop(
        model, time_test, df_test, n_length
    )
    test_time_end = len(pred_closed_loop)

    fig, axs = plt.subplots(3, 2, sharey=True, facecolor="white")  # , figsize=(15, 14))
    fig.suptitle("Closed Loop LSTM Prediction at epoch " + str(n_epochs))
    axs[0, 0].plot(
        lyapunov_time[:test_time_end],
        df_test[0, 50 : 50 + test_time_end],
        label="True Data",
    )
    axs[0, 0].plot(
        lyapunov_time[:test_time_end],
        pred_closed_loop[:, 0],
        "--",
        label="RNN Prediction",
    )
    axs[0, 0].set_ylabel("x")
    sns.kdeplot(
        df_test[0, 50:test_time_end], vertical=True, color="tab:blue", ax=axs[0, 1]
    )
    sns.kdeplot(pred_closed_loop[:, 0], vertical=True, color="tab:orange", ax=axs[0, 1])
    axs[1, 0].plot(
        lyapunov_time[:test_time_end], df_test[1, 50 : 50 + test_time_end], label="data"
    )
    axs[1, 0].plot(
        lyapunov_time[:test_time_end],
        pred_closed_loop[:, 1],
        "--",
        label="RNN prediction on test data",
    )
    axs[1, 0].set_ylabel("y")
    sns.kdeplot(
        df_test[1, 50:test_time_end], vertical=True, color="tab:blue", ax=axs[1, 1]
    )
    sns.kdeplot(pred_closed_loop[:, 1], vertical=True, color="tab:orange", ax=axs[1, 1])
    axs[2, 0].plot(
        lyapunov_time[:test_time_end],
        df_test[2, 50 : 50 + test_time_end],
        label="Numerical Solution",
    )
    axs[2, 0].plot(
        lyapunov_time[:test_time_end],
        pred_closed_loop[:, 2],
        "--",
        label="LSTM prediction",
    )
    axs[2, 0].set_ylabel("z")
    axs[2, 0].set_ylim(-1, 1)
    sns.kdeplot(
        df_test[2, 0:test_time_end], vertical=True, color="tab:blue", ax=axs[2, 1]
    )
    sns.kdeplot(pred_closed_loop[:, 2], vertical=True, color="tab:orange", ax=axs[2, 1])
    # ax3.set_xlim(5,10)
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
    plt.show()
    return pred_closed_loop


def plot_phase_space(predictions, n_epochs, df_test, img_filepath=None):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig.suptitle("Phase Space Comparison at Epoch " + str(n_epochs))
    test_time_end = len(predictions)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(
        df_test[0, 50 : 50 + test_time_end],
        df_test[1, 50 : 50 + test_time_end],
        df_test[2, 50 : 50 + test_time_end],
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

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot(
        predictions[:, 0],
        predictions[:, 1],
        predictions[:, 2],
        color="tab:orange",
        alpha=0.7,
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_title("LSTM Prediction")
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_zlim(ax1.get_zlim())
    if img_filepath != None:
        if not os.path.exists(img_filepath):
            os.makedirs(img_filepath)
        fig.savefig(img_filepath, dpi=200, facecolor="w", bbox_inches="tight")
    plt.show()


def plot_closed_loop_lya_with_phase(
    model, n_epochs, time_test, df_test, img_filepath=None, n_length=6000
):
    lyapunov_time, pred_closed_loop = prediction_closed_loop(
        model, time_test, df_test, n_length
    )
    test_time_end = len(pred_closed_loop)

    fig, axs = plt.subplots(4, 2, sharey=True, facecolor="white")  # , figsize=(15, 14))
    fig.suptitle("Closed Loop LSTM Prediction at Epoch " + str(n_epochs))
    axs[0, 0].plot(
        lyapunov_time[:test_time_end],
        df_test[0, 50 : 50 + test_time_end],
        label="True Data",
    )
    axs[0, 0].plot(
        lyapunov_time[:test_time_end],
        pred_closed_loop[:, 0],
        "--",
        label="RNN Prediction",
    )
    axs[0, 0].set_ylabel("x")
    sns.kdeplot(
        df_test[0, 50:test_time_end], vertical=True, color="tab:blue", ax=axs[0, 1]
    )
    sns.kdeplot(pred_closed_loop[:, 0], vertical=True, color="tab:orange", ax=axs[0, 1])
    axs[1, 0].plot(
        lyapunov_time[:test_time_end], df_test[1, 50 : 50 + test_time_end], label="data"
    )
    axs[1, 0].plot(
        lyapunov_time[:test_time_end],
        pred_closed_loop[:, 1],
        "--",
        label="RNN prediction on test data",
    )
    axs[1, 0].set_ylabel("y")
    sns.kdeplot(
        df_test[1, 50:test_time_end], vertical=True, color="tab:blue", ax=axs[1, 1]
    )
    sns.kdeplot(pred_closed_loop[:, 1], vertical=True, color="tab:orange", ax=axs[1, 1])
    axs[2, 0].plot(
        lyapunov_time[:test_time_end],
        df_test[2, 50 : 50 + test_time_end],
        label="Numerical Solution",
    )
    axs[2, 0].plot(
        lyapunov_time[:test_time_end],
        pred_closed_loop[:, 2],
        "--",
        label="LSTM prediction",
    )
    axs[2, 0].set_ylabel("z")
    axs[2, 0].set_ylim(-1, 1)
    sns.kdeplot(
        df_test[2, 0:test_time_end], vertical=True, color="tab:blue", ax=axs[2, 1]
    )
    sns.kdeplot(pred_closed_loop[:, 2], vertical=True, color="tab:orange", ax=axs[2, 1])
    # ax3.set_xlim(5,10)
    axs[2, 0].legend(loc="center left", bbox_to_anchor=(2.3, 2.0))
    axs[0, 0].set_xticklabels([])
    axs[1, 0].set_xticklabels([])
    axs[0, 1].get_shared_x_axes().join(axs[0, 1], axs[1, 1], axs[2, 1])
    axs[1, 1].get_shared_x_axes().join(axs[0, 1], axs[1, 1], axs[2, 1])
    axs[2, 1].get_shared_x_axes().join(axs[0, 1], axs[1, 1], axs[2, 1])
    axs[0, 1].set_xticklabels([])
    axs[1, 1].set_xticklabels([])

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(
        df_test[0, 50 : 50 + test_time_end],
        df_test[1, 50 : 50 + test_time_end],
        df_test[2, 50 : 50 + test_time_end],
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

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot(
        pred_closed_loop[:, 0],
        pred_closed_loop[:, 1],
        pred_closed_loop[:, 2],
        color="tab:orange",
        alpha=0.7,
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_title("LSTM Prediction")
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_zlim(ax1.get_zlim())

    if img_filepath != None:
        fig.savefig(img_filepath, dpi=200, facecolor="w", bbox_inches="tight")
    plt.show()

    return pred_closed_loop
