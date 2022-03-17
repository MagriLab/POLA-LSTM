import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..closed_loop_tools import (compute_lyapunov_time_arr,
                                 prediction_closed_loop)

plt.rcParams["figure.facecolor"] = "white"


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
    # axs[0, 0].axhline(y=0.43, color="lightcoral", linestyle=":")
    # axs[0, 0].axhline(y=-0.43, color="lightcoral", linestyle=":")
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
    # axs[1, 0].axhline(y=0.31, color="lightcoral", linestyle=":")
    # axs[1, 0].axhline(y=-0.31, color="lightcoral", linestyle=":")
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
    # axs[2, 0].axhline(y=0.56, color="lightcoral", linestyle=":", label="Fixpoint")
    # axs[2, 0].set_ylim(-1, 1)
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
        print("Closed Loop prediction saved at ", img_filepath)
    # plt.show()
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
    # ax1.plot(0.43, 0.31, 0.56, "x", color="tab:red", alpha=0.7)
    # ax1.plot(-0.43, -0.36, 0.56, "x", color="tab:red", alpha=0.7)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot(
        predictions[:, 0],
        predictions[:, 1],
        predictions[:, 2],
        color="tab:orange",
        alpha=0.7,
    )
    # ax2.plot(0.43, 0.31, 0.56, "x", color="tab:red", alpha=0.7)
    # ax2.plot(-0.43, -0.36, 0.56, "x", color="tab:red", alpha=0.7)
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
    # plt.show()


def plot_closed_loop_lya_lim(
    pred_closed_loop,
    time_test,
    df_test,
    t_lim_min=0,
    t_lim_max=15,
    img_filepath=None,
    window_size=50,
    n_epoch=0,
):
    lyapunov_time = compute_lyapunov_time_arr(time_test)
    test_time_end = len(pred_closed_loop)

    fig, axs = plt.subplots(3, 2, sharey=True, facecolor="white")  # , figsize=(15, 14))
    fig.suptitle("Closed Loop LSTM Prediction at Epoch " + str(n_epoch))
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
    sns.kdeplot(
        df_test[2, 0:test_time_end], vertical=True, color="tab:blue", ax=axs[2, 1]
    )
    sns.kdeplot(pred_closed_loop[:, 2], vertical=True, color="tab:orange", ax=axs[2, 1])
    # ax3.set_xlim(5,10)
    axs[2, 0].legend(loc="center left", bbox_to_anchor=(2.3, 2.0))
    axs[2, 0].set_xlabel("LT")
    axs[0, 0].set_xticklabels([])
    axs[1, 0].set_xticklabels([])
    axs[0, 0].set_xlim(t_lim_min, t_lim_max)
    axs[1, 0].set_xlim(t_lim_min, t_lim_max)
    axs[2, 0].set_xlim(t_lim_min, t_lim_max)
    axs[0, 1].get_shared_x_axes().join(axs[0, 1], axs[1, 1], axs[2, 1])
    axs[1, 1].get_shared_x_axes().join(axs[0, 1], axs[1, 1], axs[2, 1])
    axs[2, 1].get_shared_x_axes().join(axs[0, 1], axs[1, 1], axs[2, 1])
    axs[0, 1].set_xticklabels([])
    axs[1, 1].set_xticklabels([])
    if img_filepath != None:
        fig.savefig(img_filepath, dpi=200, facecolor="w", bbox_inches="tight")
        plt.close(fig)
    # plt.show()


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
    # ax1.plot(0.43, 0.31, 0.56, "x", color="tab:red", alpha=0.7)
    # ax1.plot(-0.43, -0.36, 0.56, "x", color="tab:red", alpha=0.7)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot(
        predictions[:, 0],
        predictions[:, 1],
        predictions[:, 2],
        color="tab:orange",
        alpha=0.7,
    )
    # ax2.plot(0.43, 0.31, 0.56, "x", color="tab:red", alpha=0.7)
    # ax2.plot(-0.43, -0.36, 0.56, "x", color="tab:red", alpha=0.7)
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
        plt.close(fig)
    # plt.show()



def plot_error_closed_loop_lya_lim(
    pred_closed_loop,
    time_test,
    df_test,
    t_lim_min=0,
    t_lim_max=15,
    img_filepath=None,
    window_size=50,
    n_epoch=0,
):
    lyapunov_time = compute_lyapunov_time_arr(time_test)
    test_time_end = len(pred_closed_loop)

    fig, axs = plt.subplots(3, 2, sharex=True, facecolor="white")  # , figsize=(10, 12))
    fig.suptitle("Pointwise Error of LSTM Prediction at Epoch " + str(n_epoch))
    x_error = np.abs(
        df_test[0, window_size: window_size + test_time_end] - pred_closed_loop[:, 0]
    )
    y_error = np.abs(
        df_test[1, window_size: window_size + test_time_end] - pred_closed_loop[:, 1]
    )
    z_error = np.abs(
        df_test[2, window_size: window_size + test_time_end] - pred_closed_loop[:, 2]
    )
    axs[0, 1].plot(lyapunov_time[:test_time_end], x_error, "r", label="Error")

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
    axs[0, 0].set_ylabel("x")
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
    # ax3.set_xlim(5,10)

    axs[0, 0].set_xlim(t_lim_min, t_lim_max)
    axs[1, 0].set_xlim(t_lim_min, t_lim_max)
    axs[2, 0].set_xlim(t_lim_min, t_lim_max)

    axs[0, 1].set_ylabel("$|x_{pred} - x_{num}|$")

    axs[1, 1].plot(lyapunov_time[:test_time_end], y_error, "r", label="Error")
    axs[1, 1].set_ylabel("|$y_{pred} - y_{num}|$")

    axs[2, 1].plot(lyapunov_time[:test_time_end], z_error, "r")
    axs[2, 1].set_ylabel("$|z_{pred} - z_{num}|$")

    axs[2, 1].set_yscale("log")
    axs[1, 1].set_yscale("log")
    axs[0, 1].set_yscale("log")
    axs[2, 1].set_xlim(t_lim_min, t_lim_max)
    axs[0, 1].get_shared_y_axes().join(axs[0, 1], axs[1, 1], axs[2, 1])
    axs[1, 1].get_shared_y_axes().join(axs[0, 1], axs[1, 1], axs[2, 1])
    axs[2, 1].get_shared_y_axes().join(axs[0, 1], axs[1, 1], axs[2, 1])
    axs[2, 0].set_xlabel("LT")
    fig.subplots_adjust(wspace=0.5)
    axs[2, 0].legend(loc="center left", bbox_to_anchor=(2.5, 2.0))
    # axs[0, 1].set_xticklabels([])
    # axs[1, 1].set_xticklabels([])
    if img_filepath != None:
        fig.savefig(img_filepath, dpi=200, facecolor="w", bbox_inches="tight")
        plt.close(fig)
    # plt.show()



def plot_open_loop_lya(
    model,
    n_epochs,
    time_test,
    test_dataset,
    img_filepath=None,
    n_length=6000,
    window_size=50,
):
    lyapunov_time = compute_lyapunov_time_arr(time_test, window_size=100)
    prediction = model.predict(test_dataset)
    test_time_end = len(prediction)
    fig, axs = plt.subplots(3, 2, sharey=True, facecolor="white")  # , figsize=(15, 14))
    fig.suptitle("Open Loop LSTM Prediction at epoch " + str(n_epochs))
    axs[0, 0].plot(
        lyapunov_time[:test_time_end],
        test_dataset[0, window_size: window_size + test_time_end],
        label="True Data",
    )
    axs[0, 0].plot(
        lyapunov_time[:test_time_end],
        prediction[:, 0],
        "--",
        label="RNN Prediction",
    )
    # axs[0, 0].axhline(y=0.43, color="lightcoral", linestyle=":")
    # axs[0, 0].axhline(y=-0.43, color="lightcoral", linestyle=":")
    axs[0, 0].set_ylabel("x")
    sns.kdeplot(
        test_dataset[0, window_size:test_time_end],
        vertical=True,
        color="tab:blue",
        ax=axs[0, 1],
    )
    sns.kdeplot(prediction[:, 0], vertical=True, color="tab:orange", ax=axs[0, 1])
    axs[1, 0].plot(
        lyapunov_time[:test_time_end],
        test_dataset[1, window_size: window_size + test_time_end],
        label="data",
    )
    axs[1, 0].plot(
        lyapunov_time[:test_time_end],
        prediction[:, 1],
        "--",
        label="RNN prediction on test data",
    )
    axs[1, 0].set_ylabel("y")
    # axs[1, 0].axhline(y=0.31, color="lightcoral", linestyle=":")
    # axs[1, 0].axhline(y=-0.31, color="lightcoral", linestyle=":")
    sns.kdeplot(
        test_dataset[1, window_size:test_time_end],
        vertical=True,
        color="tab:blue",
        ax=axs[1, 1],
    )
    sns.kdeplot(prediction[:, 1], vertical=True, color="tab:orange", ax=axs[1, 1])
    axs[2, 0].plot(
        lyapunov_time[:test_time_end],
        test_dataset[2, window_size: window_size + test_time_end],
        label="Numerical Solution",
    )
    axs[2, 0].plot(
        lyapunov_time[:test_time_end],
        prediction[:, 2],
        "--",
        label="LSTM prediction",
    )
    axs[2, 0].set_ylabel("z")
    # axs[2, 0].axhline(y=0.56, color="lightcoral", linestyle=":", label="Fixpoint")
    # axs[2, 0].set_ylim(-1, 1)
    sns.kdeplot(
        test_dataset[2, 0:test_time_end], vertical=True, color="tab:blue", ax=axs[2, 1]
    )
    sns.kdeplot(prediction[:, 2], vertical=True, color="tab:orange", ax=axs[2, 1])
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
        print("Open Loop prediction saved at ", img_filepath)
        plt.close(fig)
    # plt.show()
    return prediction

