

from scipy.fft import fft
import sys
import os
from pathlib import Path
import warnings
import random
import matplotlib.pyplot as plt
import numpy as np
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
from lstm.postprocessing.nrmse import nrmse_array
from lstm.preprocessing.data_class import Dataclass
from lstm.lstm import LSTMRunner
from lstm.closed_loop_tools_mtm import (prediction)
warnings.simplefilter(action="ignore", category=FutureWarning)
tf.keras.backend.set_floatx('float64')

tensorflow_shutup()

norm = 3.58
sweep_path = Path('/home/eo821/Documents/PI-LSTM/Lorenz_LSTM/src/trainings/KS/128dof_dd')


for folder_name in ["dd-26", "dd-32", "dd-64"]:#next(os.walk(sweep_path))[1]:
    sweep_models = list(filter(lambda x: x != 'images', next(os.walk(sweep_path/folder_name))[1]))
    img_filepath_folder = make_folder_filepath(sweep_path / folder_name,  'images')
    for model_name in sweep_models: #sweep_models:
        print(model_name)
        model_path = sweep_path / folder_name / model_name
        args = load_config_to_argparse(model_path)
        dim = 128 # df_train.shape[0]
        args.sys_dim = dim
        args.data_path = Path('/home/eo821/Documents/PI-LSTM/Lorenz_LSTM/src/trainings/Yael_CSV/KS/KS_128_dx62_99000_stand_3.58_deltat_0.25_M_64_trans.csv')
        args.lyap_path = Path('/home/eo821/Documents/PI-LSTM/Lorenz_LSTM/src/trainings/Yael_CSV/KS/le_128_64_20pi_025_tmax_50000.txt')
        data = Dataclass(args)
        epochs = max([int(i) for i in next(os.walk(model_path /'model'))[1]])

        img_filepath = make_folder_filepath(model_path, 'images')        

        # Compare this prediction with the LE prediction
        t_lyap = 0.08**(-1)
        N_lyap = int(t_lyap / (args.delta_t*args.upsampling))


        runner = LSTMRunner(args, 'KS_dd', data.idx_lst)
        runner.load_model(model_path, epochs)
        model = runner.model

        for batch, label in data.train_dataset.take(1):
            print(batch.shape)
        batch_pred = model(batch)
        n_random_idx = len(data.idx_lst)
        print('--- model successfully loaded---')

        print('--- successfully initialized---')
        random.seed(0)
        print(data.idx_lst)
        img_filepath = make_folder_filepath(model_path, 'images')        

        # Compare this prediction with the LE prediction
        t_lyap = 0.08**(-1)
        N_lyap = int(t_lyap / (args.delta_t*args.upsampling))


        N = 500*N_lyap
        lyapunov_time = np.arange(0, N/N_lyap, args.delta_t/t_lyap)
        pred = prediction(model, data.df_test, args.window_size, dim, data.idx_lst, N=N)
        pred = pred *norm
        df_train = data.df_train*norm
        df_test = data.df_test*norm
        # /short Pred
        N_plot = 5*N_lyap
        fig, axes = plt.subplots(nrows=1, ncols=2)
        data1 = pred[args.window_size:N_plot, :].T
        data2 = df_train[:, args.window_size:N_plot]

        # find minimum of minima & maximum of maxima
        minmin = -1 *norm
        maxmax = 1 *norm
        domain_length=20*np.pi
        im1 = axes[0].imshow(data1, vmin=minmin, vmax=maxmax, aspect='auto', cmap='seismic', extent=[lyapunov_time[0], int(lyapunov_time[N_plot]), 0, domain_length])
        im2 = axes[1].imshow(data2, vmin=minmin, vmax=maxmax, aspect='auto', cmap='seismic', extent=[lyapunov_time[0], int(lyapunov_time[N_plot]), 0, domain_length])
        axes[0].set_xlabel('LT')
        axes[0].set_ylabel('x')
        axes[0].set_title('LSTM')
        axes[1].set_title('Reference')
        axes[1].set_xlabel('LT')
        axes[1].set_ylabel('x')
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im2, cax=cbar_ax)
        plt.savefig(img_filepath/'pred_ks_flow.png', dpi=100, facecolor="w", bbox_inches="tight")
        plt.close()
    
        fig, axes = plt.subplots(nrows=1, ncols=2)
        N_plot=200*args.window_size #pred.shape[0] #50*args.window_size
        data1 = pred[args.window_size:N_plot, :]
        data2 = df_test[:, args.window_size:N_plot].T
        fig.suptitle('u(x)')
        # find minimum of minima & maximum of maxima
        minmin = -1 *norm
        maxmax = 1 *norm
        im1 = axes[0].imshow(data1, vmin=minmin, vmax=maxmax, aspect='auto', cmap='seismic', extent=[0, domain_length, lyapunov_time[0], int(lyapunov_time[N_plot])])
        im2 = axes[1].imshow(data2, vmin=minmin, vmax=maxmax, aspect='auto', cmap='seismic', extent=[0, domain_length, lyapunov_time[0], int(lyapunov_time[N_plot])])
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('LT')
        axes[0].set_title('LSTM')
        axes[1].set_title('Reference')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('LT')
        # add space for colour bar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im2, cax=cbar_ax)
        plt.savefig(img_filepath/'LSTM_pred_ks_flow.png', dpi=100, facecolor="w", bbox_inches="tight")
        plt.close()

        # Energyspectrum
        N_fft = min(pred.shape[0], df_test.shape[1])
        max_k = 45
        vv = fft(df_test[:, :N_fft].T).T
        plt.plot(np.sum(np.real(np.multiply(vv.conj(), vv))[:max_k], axis=1), '.', label='Reference - 128 dof')
        vv_idx = fft(df_test[data.idx_lst, :N_fft].T).T
        plt.plot(np.sum(np.real(np.multiply(vv_idx.conj(), vv_idx))[:n_random_idx//2], axis=1), '.', label=f'Reference - {n_random_idx} dof')
        vv_pred = fft(pred[:N_fft, :]).T
        plt.plot(np.sum(np.real(np.multiply(vv_pred.conj(), vv_pred))[:max_k], axis=1), '.', label='LSTM')
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
        plt.plot(time_axis, 0.5*np.sum(pred[:N_plot, :]**2, axis=1)/pred.shape[1], label='LSTM')
        plt.plot(time_axis, 0.5*np.sum(df_test[:, :N_plot].T**2, axis=1)/df_test.shape[0], label='Reference - 128 dof')
        plt.plot(time_axis, 0.5*np.sum(df_test[data.idx_lst, :N_plot].T**2, axis=1)/n_random_idx, label=f'Reference - {n_random_idx} dof')
        plt.ylabel(r'$ \frac{1}{N} \sum_1^N u_i^2(t)$')
        plt.xlabel('T')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        sub_axes = plt.axes([1.0, 0.75, .5, .5]) 
        N_plot = 1000
        # plot the zoomed portion
        time_axis = data.time_test[2000:2000+N_plot]
        sub_axes.plot(time_axis, 0.5*np.sum(pred[2000:2000+N_plot, :]**2/pred.shape[1], axis=1)) 
        sub_axes.plot(time_axis, 0.5*np.sum(df_test[:, 2000:2000+N_plot].T**2, axis=1)/df_test.shape[0])
        sub_axes.plot(time_axis, 0.5*np.sum(df_test[data.idx_lst, 2000:2000+N_plot].T**2, axis=1)/n_random_idx)
        sub_axes.set_ylabel(r'$ \frac{1}{N} \sum_1^N u_i^2(t)$')
        sub_axes.set_xlabel('T')
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
       