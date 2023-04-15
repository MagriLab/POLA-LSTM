from scipy.fft import fft
import sys
import os
from pathlib import Path
import warnings
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import tensorflow as tf
gpus =  tf.config.list_physical_devices('GPU')
if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.set_visible_devices(gpus[2], 'GPU')
        tf.config.set_logical_device_configuration(gpus[2], [tf.config.LogicalDeviceConfiguration(memory_limit=3072)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Virtual devices must be set before GPUs have been initialize
      print(e)

sys.path.append('../../../')
from lstm.utils.config import load_config_to_argparse
from lstm.utils.create_paths import make_folder_filepath
from lstm.utils.supress_tf_warning import tensorflow_shutup
from lstm.postprocessing.nrmse import nrmse_array, vpt
from lstm.preprocessing.data_class import Dataclass
from lstm.lstm import LSTMRunner
from lstm.closed_loop_tools_mtm import (prediction)
warnings.simplefilter(action="ignore", category=FutureWarning)
tf.keras.backend.set_floatx('float64')

tensorflow_shutup()
norm = 13.33
sweep_path = Path('../L96_d20_rk4')


for folder_name in ['D-18']: #list(filter(lambda x: x != 'images', next(os.walk(sweep_path))[1])):
    sweep_models = list(filter(lambda x: x != 'images', next(os.walk(sweep_path/folder_name))[1]))
    img_filepath_folder = make_folder_filepath(sweep_path / folder_name,  'images')
    for model_name in sweep_models:
        print(model_name)
        model_path = sweep_path / folder_name / model_name
        args = load_config_to_argparse(model_path)
        dim = 20 # df_train.shape[]
        args.sys_dim = dim
        args.data_path = Path('../Yael_CSV/L96/dim_20_rk4_200000_0.01_stand13.33_trans.csv')
        args.train_ratio = args.train_ratio*42500/200000
        args.valid_ratio = args.train_ratio*42500/200000
        args.lyap_path = Path('../Yael_CSV/L96/dim_20_lyapunov_exponents.txt')
        data = Dataclass(args)
        epochs = max([int(i) for i in next(os.walk(model_path /'model'))[1]])

        img_filepath = make_folder_filepath(model_path, 'images')        

        # Compare this prediction with the LE prediction
        t_lyap = 1.55**(-1)
        N_lyap = int(t_lyap / (args.delta_t*args.upsampling))


        runner = LSTMRunner(args, 'l96', data.idx_lst)
        runner.load_model(model_path, epochs)
        model = runner.model

        for batch, label in data.train_dataset.take(1):
            print(batch.shape)
        batch_pred = model(batch)
        n_random_idx = len(data.idx_lst)
        print('--- model successfully loaded---')

        N =  min(data.df_test.shape[1], 500*N_lyap)
        pred_model = prediction(model, data.df_test, args.window_size, dim, data.idx_lst, N=N)
        # /short Pred
        N_plot = 5*N_lyap
        pred = pred_model*norm
        df_test = data.df_test*norm
        lyapunov_time = np.arange(0, N/N_lyap, args.delta_t*args.upsampling/t_lyap)
        pred_horizon = lyapunov_time[vpt(pred[args.window_size:, :], df_test[:, args.window_size:], 0.5)]
        print(f'Prediction Horizon {pred_horizon}')
        plt.plot(lyapunov_time[:N_plot], pred[args.window_size:args.window_size+N_plot, :])

        plt.plot(lyapunov_time[:N_plot], pred[args.window_size:args.window_size+N_plot, -1], label='LSTM')
        plt.plot(lyapunov_time[:N_plot], df_test[:, args.window_size:args.window_size+N_plot].T, 'k--')
        plt.plot(lyapunov_time[:N_plot], df_test[-1, args.window_size:args.window_size+N_plot].T, 'k--', label='Reference')
        plt.xlabel("LT")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f'Prediction Horizon {pred_horizon:.2f}LT')
        plt.xlabel("LT")
        plt.savefig(f'{img_filepath}/short_pred.png', dpi=100, facecolor="w", bbox_inches="tight")
        plt.close()

        N_plot = 50*N_lyap
        missing_idx = list(set(range(0, dim)).difference(data.idx_lst))
        print(missing_idx)
        lyapunov_time = np.arange(0, N/N_lyap, args.delta_t*args.upsampling/t_lyap)
        plt.plot(lyapunov_time[:N_plot], pred[args.window_size:args.window_size+N_plot, missing_idx], label='LSTM')
        plt.plot(lyapunov_time[:N_plot], df_test[missing_idx,
                 args.window_size:args.window_size+N_plot].T, 'k--', label='Reference')
        plt.xlabel("LT")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f' Missing Observations: {len(missing_idx)}')
        plt.savefig(f'{img_filepath}/missing_observation.png', dpi=100, facecolor="w", bbox_inches="tight")
        plt.close()

        # Energyspectrum
        N_fft = min(pred.shape[0], df_test.shape[1])
        vv = fft(df_test[:, :N_fft].T).T

        plt.plot(np.sum(np.real(np.multiply(vv.conj(), vv))[:dim//2], axis=1), '.', label='Reference')
        print(pred.shape)
        vv_pred = fft(pred[:N_fft, :]).T
        plt.plot(np.sum(np.real(np.multiply(vv_pred.conj(), vv_pred))[:dim//2], axis=1), '.', label='LSTM')
        # plt.loglog()
        plt.yscale('log')
        plt.xlabel('k')
        plt.ylabel('E(k)')
        plt.title("Energy Spectrum")
        plt.legend()
        plt.savefig(img_filepath/'energyspectrum.png', dpi=100, facecolor="w", bbox_inches="tight")
        plt.close()

        # kinetic Energy
        plt.legend()
        N_plot = min(df_test.shape[1], pred.shape[0], 25*N_lyap)
        time_axis = data.time_test[:N_plot]
        plt.title('Kinetic Energy on Test Data')
        plt.plot(time_axis, np.sum(pred[:N_plot, :]**2, axis=1), label='LSTM')
        plt.plot(time_axis, np.sum(df_test[:, :N_plot].T**2, axis=1), label='Reference')
        plt.ylabel('$\sum_i^N u_i^2(t)$')
        plt.xlabel('T')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(f'{img_filepath}/kin_energy_test.png', dpi=100, facecolor="w", bbox_inches="tight")

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
        bin_size = 100
        lw = 3
        fs = 12

        fig = plt.figure()
        fig.tight_layout()
        N_plot = min(df_test.shape[1], pred.shape[0])
        # fig.suptitle(f'PDF over {int(N/N_lyap)} LT', fontsize=fs+5)
        for i in range(0, dim):
            ax = plt.subplot(dim//5, 5, i+1)
            truth = df_test[i, :N_plot]
            lstm_sol = pred[:N_plot, i]
            density = stats.gaussian_kde(truth)
            n, x, _ = plt.hist(truth, bins=bin_size, color='white', histtype=u'step',
                               density=True, orientation='horizontal')
            density_100z = stats.gaussian_kde(lstm_sol)
            n, x_100z, _ = plt.hist(lstm_sol, bins=bin_size, color='white',
                                    histtype=u'step', density=True, orientation='horizontal')
            ax.plot(x, density(x), 'k', linewidth=lw, label='Reference')
            ax.plot(x_100z, density_100z(x_100z), 'r--', linewidth=lw, label='LSTM')
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
                        ax.legend([spine], ['Missing \nobservations'], bbox_to_anchor=(5.25, 1.7))
            ax.set_xlim(min(x)*1.1, max(x)*1.1)
            ax.set_ylim(-0.1, (max(density(x).max(), density_100z(x_100z).max())*1.1))
            ax.set_aspect(1./ax.get_data_ratio())
            # ax.set_yticklabels([])
            ax.grid(True, c='lightgray', linestyle='--', linewidth=0.5)
        ax.legend(loc='lower left', bbox_to_anchor=(1.5, 1.6))
        fig.subplots_adjust(wspace=0.5, hspace=-0.3)
        plt.savefig(f'{img_filepath}/{epochs}_{int(N_plot/N_lyap)}_LT_PDF.png',
                    dpi=100, facecolor="w", bbox_inches="tight")
        print(f'Images saved at {model_path}{int(N/N_lyap)}_LT_PDF.png')
        plt.close()

        # Statistics
        bin_size = 100
        lw = 3
        fs = 12
        print(missing_idx)
        fig = plt.figure()  # figsize=(12, 10)
        # fig.suptitle(f'PDF of Missing States over {int(N_plot/N_lyap)} LT', fontsize=fs+5)
        j = 0
        for i in missing_idx:
            ax = plt.subplot(1, len(missing_idx), j+1)
            truth = df_test[i, :N_plot]
            lstm_sol = pred[:N_plot, i]
            density = stats.gaussian_kde(truth)
            n, x, _ = plt.hist(truth, bins=bin_size, color='white', histtype=u'step',
                               density=True, orientation='horizontal')
            density_100z = stats.gaussian_kde(lstm_sol)
            n, x_100z, _ = plt.hist(lstm_sol, bins=bin_size, color='white',
                                    histtype=u'step', density=True, orientation='horizontal')
            ax.plot(x, density(x), 'k', linewidth=lw, label='Reference')
            ax.plot(x_100z, density_100z(x_100z), 'r--', linewidth=lw, label='LSTM')
            ax.set_xlabel('$u_{'+str(i+1)+' }$', fontsize=fs+2)
            if j == 0:
                ax.set_ylabel('PDF', fontsize=fs+2)

            plt.axis('on')
            ax.set_xlim(min(x)*1.1, max(x)*1.1)
            ax.set_ylim(-0.1, (max(density(x).max(), density_100z(x_100z).max())*1.1))
            ax.grid(True, c='lightgray', linestyle='--', linewidth=0.5)
            ax.set_aspect(1./ax.get_data_ratio())
            j = j + 1
        fig.subplots_adjust(wspace=0.5, hspace=0.8)
        ax.legend(loc='upper right', bbox_to_anchor=(2.5, 0.75))
        plt.savefig(f'{img_filepath}/missing_idx_{epochs}_{int(N_plot/N_lyap)}_LT_PDF.png',
                    dpi=100, facecolor="w", bbox_inches="tight")
        print(f'Images saved at {model_path}{int(N/N_lyap)}_LT_PDF.png')
        plt.close()
