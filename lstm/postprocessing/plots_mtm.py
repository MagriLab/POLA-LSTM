
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from lstm.lorenz import fixpoints

from ..closed_loop_tools_mtm import prediction_closed_loop

plt.rcParams["figure.facecolor"] = "white"

x_fix, y_fix, z_fix = fixpoints(total_points=10000, unnorm=False)


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


def plot_prediction(
    model,
    n_epochs,
    time_test,
    df_test,
    img_filepath=None,
    n_length=6000,
    window_size=50,
    c_lyapunov=0.90566
):
    # test_time_end = len(pred_closed_loop)
    lyapunov_time, pred_closed_loop = prediction_closed_loop(
        model, time_test, df_test, n_length, window_size=window_size, c_lyapunov=c_lyapunov
    )
    test_time_end = len(pred_closed_loop)
    dim = df_test.shape[0]
    fig, axs = plt.subplots(dim, 1, sharey=True, facecolor="white", edgecolor='k')  # , figsize=(15, 14))
    rel_l2_err = np.linalg.norm(df_test[:, window_size: window_size + test_time_end].T -
                                pred_closed_loop[: test_time_end]) / np.linalg.norm(pred_closed_loop[: test_time_end])
    fig.suptitle("Relative L2 Error for %d LT: %.2e" % (lyapunov_time[test_time_end], rel_l2_err))

    fig.subplots_adjust(hspace=.5, wspace=.001)

    axs = axs.ravel()

    for i in range(dim):
        axs[i].plot(
            lyapunov_time[:test_time_end],
            df_test[i, window_size: window_size + test_time_end],
            label="Numerical Solution",
        )
        axs[i].plot(
            lyapunov_time[:test_time_end],
            pred_closed_loop[:test_time_end, i],
            "--",
            label="RNN Prediction",
        )
        comp = "u_"+str(i+1)
        axs[i].set_ylabel(r"$"+comp+"$")
        if i < dim-1:
            axs[i].set_xticklabels([])

    axs[dim-1].set_xlabel('LT')
    axs[dim-1].legend(loc="center left", bbox_to_anchor=(1.3, 2.0))
    # axs[0].set_xticklabels([])
    # axs[1].set_xticklabels([])
    if img_filepath != None:
        fig.savefig(img_filepath, dpi=200, facecolor="w", bbox_inches="tight")
        print("Closed Loop prediction saved at ", img_filepath)
    return pred_closed_loop


def plot_cdv(
    model,
    n_epochs,
    time_test,
    df_test,
    img_filepath=None,
    n_length=6000,
    window_size=50,
    c_lyapunov=0.033791
):

    lyapunov_time, pred_closed_loop = prediction_closed_loop(
        model, time_test, df_test, n_length, window_size=window_size, c_lyapunov=c_lyapunov
    )
    rel_l2_err = np.linalg.norm(df_test[:, window_size: window_size + n_length].T -
                                pred_closed_loop[: n_length]) / np.linalg.norm(pred_closed_loop[: n_length])
    fig, axs = plt.subplots(2, 1, facecolor='w', edgecolor='k', sharey=True, sharex=True)
    fig.subplots_adjust(hspace=.5, wspace=.001)
    axs = axs.ravel()
    color = plt.cm.tab10(np.linspace(0, 1, 10))
    for i in range(0, 3):
        axs[0].plot(lyapunov_time[:n_length], df_test[i, window_size:window_size+n_length],
                    label='$u'+str(i+1)+'$', c=color[i])  # (U/np.max(np.abs(U), axis=0))[cutoff:, i])
        axs[0].plot(lyapunov_time[:n_length], (pred_closed_loop.T)[i, :n_length],
                    ':', c=color[i])  # (U/np.max(np.abs(U), axis=0))[cutoff:, i])

    for i in range(3, 6):
        axs[1].plot(
            lyapunov_time[: n_length],
            df_test[i, window_size: window_size + n_length],
            label='$u' + str(i + 1) + '$', c=color[i])
        axs[1].plot(lyapunov_time[:n_length], (pred_closed_loop.T)[i, :n_length], ':', c=color[i])
    axs[1].set_xlabel('LT')
    axs[0].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    axs[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

    axs[0].set_title("Relative L2 Error for %.1f LT: %.2e" % (lyapunov_time[n_length], rel_l2_err))
    fig.suptitle('Network Prediction at Epoch '+str(n_epochs))

    if img_filepath != None:
        fig.savefig(img_filepath, dpi=120, facecolor="w", bbox_inches="tight")
        plt.close()
    print("Prediction saved at ", img_filepath)
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


def plot_de(
    pred_closed_loop,
    n_epochs,
    lyapunov_time,
    df_test,
    img_filepath=None,
    n_length=6000,
    window_size=50
):

    test_time_end = len(pred_closed_loop)

    fig, axs = plt.subplots(3, 1, sharey=True, facecolor="white")  # , figsize=(15, 14))
    # + "\n KL Divergence: %.2f"% prediction_horizon.kl_divergence(pred_closed_loop, df_test.T, window_size=window_size))
    fig.suptitle("Closed Loop LSTM Prediction at epoch " + str(n_epochs))

    sns.kdeplot(
        df_test[0, window_size:test_time_end],
        vertical=True,
        color="tab:blue",
        ax=axs[0],
    )
    sns.kdeplot(pred_closed_loop[:, 0], vertical=True, color="tab:orange", ax=axs[0])
    sns.kdeplot(
        df_test[1, window_size:test_time_end],
        vertical=True,
        color="tab:blue",
        ax=axs[1],
    )
    sns.kdeplot(pred_closed_loop[:, 1], vertical=True, color="tab:orange", ax=axs[1])
    sns.kdeplot(
        df_test[2, 0:test_time_end], vertical=True, color="tab:blue", ax=axs[2]
    )
    sns.kdeplot(pred_closed_loop[:, 2], vertical=True, color="tab:orange", ax=axs[2])
    if img_filepath != None:
        fig.savefig(img_filepath, dpi=200, facecolor="w", bbox_inches="tight")
        print("Closed Loop prediction saved at ", img_filepath)
    return pred_closed_loop



def plot_pred_save(pred, df_valid, img_filepath=None):
    dim = df_valid.shape[0]
    N_plot = min(pred.shape[0], df_valid.shape[1])
    fig, axs = plt.subplots(dim, 1, sharex=True, facecolor="white")  # , figsize=(15, 14))
    rel_l2_err = np.linalg.norm(df_valid[:, :N_plot].T -
                                pred[: N_plot]) / np.linalg.norm(pred[: N_plot])
    fig.suptitle(f"Relative L2 Error for {rel_l2_err}")
    fig.subplots_adjust(hspace=.5, wspace=.001)

    axs = axs.ravel()

    for i in range(dim):
        axs[i].plot(
            df_valid[i, :N_plot],
            label="Numerical Solution",
        )
        axs[i].plot(
            pred[:N_plot, i],
            "--",
            label="RNN Prediction",
        )
        comp = "u_"+str(i+1)
        axs[i].set_ylabel(r"$"+comp+"$")
        if i < dim-1:
            axs[i].set_xticklabels([])
    
    if img_filepath != None:
        plt.savefig(img_filepath, dpi=100, facecolor="w", bbox_inches="tight")
        print("prediction saved at ", img_filepath)
    plt.close()

# def plot_prediction(
#     pred_closed_loop,
#     n_epochs,
#     lyapunov_time,
#     df_test,
#     img_filepath=None,
#     n_length=6000,
#     window_size=50,
#     test_time_end=100
# ):
#     # test_time_end = len(pred_closed_loop)

#     fig, axs = plt.subplots(3, 1, sharey=True, facecolor="white")  # , figsize=(15, 14))
#     rel_l2_err = np.linalg.norm(df_test[:, window_size: window_size + test_time_end].T - pred_closed_loop[:test_time_end])/np.linalg.norm(pred_closed_loop[:test_time_end])
#     fig.suptitle("Relative L2 Error for %d LT: %.2e"%(lyapunov_time[test_time_end], rel_l2_err))
#     axs[0].plot(
#         lyapunov_time[:test_time_end],
#         df_test[0, window_size: window_size + test_time_end],
#         label="True Data",
#     )
#     axs[0].plot(
#         lyapunov_time[:test_time_end],
#         pred_closed_loop[:test_time_end, 0],
#         "--",
#         label="RNN Prediction",
#     )
#     axs[0].axhline(y=x_fix, color="lightcoral", linestyle=":")
#     axs[0].axhline(y=-x_fix, color="lightcoral", linestyle=":")
#     axs[0].set_ylabel("x")
#     axs[1].plot(
#         lyapunov_time[:test_time_end],
#         df_test[1, window_size: window_size + test_time_end],
#         label="data",
#     )
#     axs[1].plot(
#         lyapunov_time[:test_time_end],
#         pred_closed_loop[:test_time_end, 1],
#         "--",
#         label="RNN prediction on test data",
#     )
#     axs[1].set_ylabel("y")
#     axs[1].axhline(y=y_fix, color="lightcoral", linestyle=":")
#     axs[1].axhline(y=-y_fix, color="lightcoral", linestyle=":")
#     axs[2].plot(
#         lyapunov_time[:test_time_end],
#         df_test[2, window_size: window_size + test_time_end],
#         label="Numerical Solution",
#     )
#     axs[2].plot(
#         lyapunov_time[:test_time_end],
#         pred_closed_loop[:test_time_end, 2],
#         "--",
#         label="LSTM prediction",
#     )
#     axs[2].set_ylabel("z")
#     axs[2].axhline(y=z_fix, color="lightcoral", linestyle=":", label="Fixpoint")
#     axs[2].set_xlabel('LT')
#     axs[2].legend(loc="center left", bbox_to_anchor=(1.3, 2.0))
#     axs[0].set_xticklabels([])
#     axs[1].set_xticklabels([])
#     if img_filepath != None:
#         fig.savefig(img_filepath, dpi=200, facecolor="w", bbox_inches="tight")
#         print("Closed Loop prediction saved at ", img_filepath)
#     return pred_closed_loop
