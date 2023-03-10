
import sys
import os
from pathlib import Path
import warnings
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import tensorflow as tf
sys.path.append('../../../')

from lstm.utils.config import load_config_to_argparse
from lstm.utils.create_paths import make_folder_filepath
from lstm.utils.supress_tf_warning import tensorflow_shutup
from lstm.postprocessing.nrmse import vpt, nrmse_array
from lstm.preprocessing.data_processing import (df_train_valid_test_split,
                                                train_valid_test_split, create_df_nd_random_md_mtm_idx)
from lstm.lstm_model import load_model
from lstm.lstm import LSTMRunner
from lstm.closed_loop_tools_mtm import (prediction)
warnings.simplefilter(action="ignore", category=FutureWarning)
tf.keras.backend.set_floatx('float64')
tensorflow_shutup()

lorenz_normal = np.array([21.29, 29.01, 53.75])

mydf = np.genfromtxt(
    '/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/trainings/Yael_CSV/L63/l63_euler_168500_0.01_stand21.29_29.01_53.75_trans.csv',
    # '/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/trainings/Yael_CSV/L96/dim_10_rk4_42500_0.01_standlorenz_normal[i]_trans.csv',
    delimiter=",").astype(
    np.float64)

sweep_path = Path('/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/trainings/l63-pi/')


for folder_name in ['pi-1']:  # ,'D10-10' next(os.walk(sweep_path))[1]:
    sweep_models = list(filter(lambda x: x != 'images', next(os.walk(sweep_path/folder_name))[1]))
    img_filepath_folder = make_folder_filepath(sweep_path / folder_name,  'images')
    for model_name in sweep_models:
        print(model_name)
        model_path = sweep_path / folder_name / model_name
        args = load_config_to_argparse(model_path)
        dim = 3  # df_train.shape[0]
        n_random_idx = 1  # int(folder_name[-1])
        # dim = n_random_idx
        epochs = max([int(i) for i in next(os.walk(model_path / 'model'))[1]])
        img_filepath = make_folder_filepath(model_path, 'images')
        train_ratio = args.train_ratio*(57500/168500)
        valid_ratio = args.valid_ratio*(57500/168500)
        random.seed(0)
        # idx_lst = random.sample(range(1, 10+1), n_random_idx)
        # idx_lst.sort()
        # print(idx_lst)
        df_train, df_valid, df_test = df_train_valid_test_split(
            mydf[1:, :: args.upsampling],
            train_ratio=train_ratio, valid_ratio=valid_ratio)
        time_train, time_valid, time_test = train_valid_test_split(
            mydf[0, ::args.upsampling], train_ratio=train_ratio, valid_ratio=valid_ratio)
        # Compare this prediction with the LE prediction

        t_lyap = 0.9**(-1)
        N_lyap = int(t_lyap / (args.delta_t*args.upsampling))
        idx_lst, train_dataset = create_df_nd_random_md_mtm_idx(
            df_train.transpose(),
            args.window_size, args.batch_size, df_train.shape[0],
            n_random_idx=n_random_idx)
        missing_idx = list(set(range(0, 3)).difference(idx_lst))
        for batch, label in train_dataset.take(1):
            print(f'Shape of batch: {batch.shape} \n Shape of Label {label.shape}')
            
        runner = LSTMRunner(args, 'l96', idx_lst)
        runner.load_model(model_path, epochs)
        model = runner.model
        batch_pred = model(batch)

        print('--- model successfully loaded---')

        N = 1000*N_lyap
        pred = prediction(model, df_test, args.window_size, dim, n_random_idx, N=N)
        # /short Pred
        lyapunov_time = np.arange(0, N/N_lyap, args.delta_t/t_lyap)
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)  # , figsize=(15, 15))
        N_plot = 10*N_lyap
        ax1.plot(lyapunov_time[:N_plot-args.window_size], lorenz_normal[0]*df_test[0, args.window_size:N_plot].T)
        ax1.plot(lyapunov_time[:N_plot-args.window_size], lorenz_normal[0]*pred[args.window_size:N_plot, 0])
        ax1.set_ylabel('x')
        ax2.plot(lyapunov_time[:N_plot-args.window_size], lorenz_normal[1]*df_test[1, args.window_size:N_plot].T, )
        ax2.plot(lyapunov_time[:N_plot-args.window_size], lorenz_normal[1]*pred[args.window_size:N_plot, 1])
        ax2.set_ylabel('y')
        ax3.plot(lyapunov_time[:N_plot-args.window_size], lorenz_normal[2]*df_test[2, args.window_size:N_plot].T, label='Target')
        ax3.plot(lyapunov_time[:N_plot-args.window_size], lorenz_normal[2]*pred[args.window_size:N_plot, 2], label='LSTM')
        ax3.set_ylabel('z')
        ax3.set_xlabel('LT')
        # ax3.set_xlim(0, 50)
        ax3.legend(loc='center left', bbox_to_anchor=(1, 1.65))
        plt.suptitle(
            f'Prediction Horizon: {lyapunov_time[vpt(pred[args.window_size:, :], df_test[:, args.window_size:], 0.4)]:.2f} LT')
        plt.savefig(f'{img_filepath}/{epochs}_{int(N_plot/N_lyap)}_LT_short_pred.png',
                    dpi=100, facecolor="w", bbox_inches="tight")
        print(f'Images saved at {img_filepath}/{epochs}_{int(N_plot/N_lyap)}_LT_short_pred.png')
        plt.close()

        # nrmse
        model_nrmse_time = nrmse_array(
            pred[args.window_size: args.window_size + N_plot, :],
            df_test[:, args.window_size: args.window_size + N_plot])
        N_plot = min(5*N_lyap, len(lyapunov_time), len(model_nrmse_time))
        plt.plot(lyapunov_time[:N_plot], model_nrmse_time[:N_plot])
        plt.hlines(y=0.4, xmin=lyapunov_time[0], xmax=lyapunov_time[N_plot])
        plt.xlabel('LT')
        plt.ylabel('NRMSE')
        plt.yscale('log')
        plt.savefig(f'{img_filepath}/nrmse.png', dpi=100, facecolor="w", bbox_inches="tight")
        plt.close()

        # Statistics

        # Statistics
        bin_size = 50
        lw = 3
        fs = 12

        fig = plt.figure()
        fig.tight_layout()
        N_plot = min(df_test.shape[1], pred.shape[0])
        # fig.suptitle(f'PDF over {int(N/N_lyap)} LT', fontsize=fs+5)
        for i in range(0, dim):
            ax = plt.subplot(1, 3, i+1)
            truth = df_test[i, :N_plot]
            lstm_sol = pred[:N_plot, i]
            density = stats.gaussian_kde(truth)
            n, x, _ = plt.hist(truth, bins=bin_size, color='white', histtype=u'step',
                               density=True, orientation='horizontal')
            density_100z = stats.gaussian_kde(lstm_sol)
            n, x_100z, _ = plt.hist(lstm_sol, bins=bin_size, color='white',
                                    histtype=u'step', density=True, orientation='horizontal')
            ax.plot(x*lorenz_normal[i], density(x), 'k', linewidth=lw, label='Reference')
            ax.plot(x_100z*lorenz_normal[i], density_100z(x_100z), 'r--', linewidth=lw, label='LSTM')
            ax.set(adjustable='box', aspect='auto')
            ax.set_xlabel('$u_{'+str(i+1)+' }$', fontsize=fs+2)
            if i % 5 == 0:
                ax.set_ylabel('PDF', fontsize=fs+2)
            plt.axis('on')
            # ax.set_xticklabels([])
            if i in missing_idx:
                print(i)
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
                    spine.set_color('blue')
                    if i == missing_idx[-1]:
                        ax.legend([spine], ['Missing \nobservation'], bbox_to_anchor=(5.75, 0.65))
            ax.set_xlim(min(x*lorenz_normal[i])*1.1, max(x*lorenz_normal[i])*1.1)
            ax.set_ylim(-0.1, (max(density(x).max(), density_100z(x_100z).max())*1.1))
            ax.set_aspect(1./ax.get_data_ratio())
            # ax.set_yticklabels([])
            ax.grid(True, c='lightgray', linestyle='--', linewidth=0.5)
        ax.legend(loc='lower left', bbox_to_anchor=(1.5, 0.65))
        fig.subplots_adjust(wspace=0.5, hspace=-0.3)
        plt.savefig(f'{img_filepath}/{epochs}_{int(N_plot/N_lyap)}_LT_PDF.png',
                    dpi=100, facecolor="w", bbox_inches="tight")
        print(f'Images saved at {model_path}{int(N/N_lyap)}_LT_PDF.png')
        plt.close()

        # Phase Plot

        fig = plt.figure(figsize=plt.figaspect(0.5))
        test_time_end = 100*N_lyap
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot(
            lorenz_normal[0]*df_train[0, args.window_size: args.window_size + test_time_end],
            lorenz_normal[1]*df_train[1, args.window_size: args.window_size + test_time_end],
            lorenz_normal[2]*df_train[2, args.window_size: args.window_size + test_time_end],
            color="tab:blue",
            alpha=0.7,
        )
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
        ax1.set_title("Reference")

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.plot(
            lorenz_normal[0]*pred[:, 0],
            lorenz_normal[1]*pred[:, 1],
            lorenz_normal[2]*pred[:, 2],
            color="tab:orange",
            alpha=0.7,
        )
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
        ax2.set_title("LSTM")
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_zlim(ax1.get_zlim())
        plt.savefig(f'{img_filepath}/Phase_{int(test_time_end/N_lyap)}_LT_PDF.png',
                    dpi=100, facecolor="w", bbox_inches="tight")
        plt.close()
