

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
sys.path.append('../../../')
from lstm.utils.config import load_config_to_dict
from lstm.utils.create_paths import make_folder_filepath
from lstm.utils.supress_tf_warning import tensorflow_shutup
from lstm.postprocessing.nrmse import vpt, nrmse_array
from lstm.preprocessing.data_processing import (df_train_valid_test_split,
                                                train_valid_test_split, create_df_nd_random_md_mtm_idx)
from lstm.lstm_model import load_model
from lstm.closed_loop_tools_mtm import (prediction)
warnings.simplefilter(action="ignore", category=FutureWarning)
tf.keras.backend.set_floatx('float64')
tensorflow_shutup()
# mpl.rc('text', usetex = False)
# mpl.rc('font', family = 'serif')

mydf = np.genfromtxt(
    '/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/diff_dyn_sys/KS_flow/CSV/L20_pi/KS_128_dx62_99000_stand_3.58_deltat_0.25_M_64_trans.csv',
    delimiter=",").astype(
    np.float64)

sweep_path = Path('/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/trainings/ks/128dof')


for folder_name in ['pi-64']:  # ,'D10-10' next(os.walk(sweep_path))[1]:
    sweep_models = list(filter(lambda x: x != 'images', next(os.walk(sweep_path/folder_name))[1]))
    img_filepath_folder = make_folder_filepath(sweep_path / folder_name,  'images')
    for model_name in sweep_models:
        print(model_name)
        model_path = sweep_path / folder_name / model_name
        model_dict = load_config_to_dict(model_path)

        dim = 128  # df_train.shape[0]
        n_random_idx = int(folder_name[-2:])
        # dim = n_random_idx
        window_size = model_dict['DATA']['WINDOW_SIZE']
        n_cell = model_dict['ML_CONSTRAINTS']['N_CELLS']
        epochs = max([int(i) for i in next(os.walk(model_path / 'model'))[1]])
        if len(list(filter(lambda x: x == 'images', next(os.walk(model_path))[1])))==1:
            continue
        print(f'Epochs {epochs}')
        dt = model_dict['DATA']['DELTA T']  # time step
        batch_size = model_dict['ML_CONSTRAINTS']['BATCH_SIZE']
        img_filepath = make_folder_filepath(model_path, 'images')
        model = load_model(model_path, epochs, model_dict, dim=dim)
        upsampling = model_dict['DATA']['UPSAMPLING']
        train_ratio = model_dict['DATA']['TRAINING RATIO']*(14400/99000)
        valid_ratio = model_dict['DATA']['VALID RATIO']*(14400/99000)
        domain_length = 20*np.pi
        random.seed(0)
        # idx_lst = random.sample(range(1, 10+1), n_random_idx)
        # idx_lst.sort()
        # print(idx_lst)
        df_train, df_valid, df_test = df_train_valid_test_split(
            mydf[1:, :: upsampling],
            train_ratio=train_ratio, valid_ratio=valid_ratio)
        time_train, time_valid, time_test = train_valid_test_split(
            mydf[0, ::upsampling], train_ratio=train_ratio, valid_ratio=valid_ratio)
        # Compare this prediction with the LE prediction
        t_lyap = 0.08**(-1)
        N_lyap = int(t_lyap / (dt*upsampling))
        print(df_train.shape)
        idx_lst, train_dataset = create_df_nd_random_md_mtm_idx(
            df_train.transpose(),
            window_size, batch_size, df_train.shape[0],
            n_random_idx=n_random_idx)
        print(type(train_dataset))
        for batch, label in train_dataset.take(1):
            print(f'Shape of batch: {batch.shape} \n Shape of Label {label.shape}')
        batch_pred = model(batch)

        print('--- model successfully loaded---')

        N = 1000*N_lyap
        lyapunov_time = np.arange(0, N/N_lyap, dt/t_lyap)
        pred = prediction(model, df_test, window_size, dim, n_random_idx, N=N)
        # /short Pred
        N_plot = 5*N_lyap
        fig, axes = plt.subplots(nrows=1, ncols=2)
        data1 = pred[window_size:N_plot, :].T
        data2 = df_train[:, window_size:N_plot]

        # find minimum of minima & maximum of maxima
        minmin = -1
        maxmax = 1

        im1 = axes[0].imshow(data1, vmin=minmin, vmax=maxmax, aspect='auto', cmap='seismic', extent=[lyapunov_time[0], int(lyapunov_time[N_plot]), 0, 60])
        im2 = axes[1].imshow(data2, vmin=minmin, vmax=maxmax, aspect='auto', cmap='seismic', extent=[lyapunov_time[0], int(lyapunov_time[N_plot]), 0, 60])
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
        N_plot=200*window_size #pred.shape[0] #50*window_size
        data1 = pred[window_size:N_plot, :]
        data2 = df_test[:, window_size:N_plot].T
        fig.suptitle('u(x)')
        # find minimum of minima & maximum of maxima
        minmin = -1
        maxmax = 1
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
        max_k = int(dim/2)-1
        vv = fft(df_test[:, :N_fft].T).T
        plt.plot(np.sum(np.real(np.multiply(vv.conj(), vv))[:max_k], axis=1), '.', label='Reference - 128 dof')
        vv_idx = fft(df_test[idx_lst, :N_fft].T).T
        plt.plot(np.sum(np.real(np.multiply(vv_idx.conj(), vv_idx))[:max_k], axis=1), '.', label='Reference - 64 dof')
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
        time_axis = time_test[:N_plot]
        plt.title('Kinetic Energy on Test Data')
        plt.plot(time_axis, np.sum(pred[:N_plot, :]**2, axis=1), label='LSTM')
        plt.plot(time_axis, np.sum(df_test[:, :N_plot].T**2, axis=1), label='Reference - 128 dof')
        plt.plot(time_axis, np.sum(df_test[idx_lst, :N_plot].T**2, axis=1), label='Reference - 64 dof')
        plt.ylabel('$\sum_i^N u_i^2(t)$')
        plt.xlabel('T')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        sub_axes = plt.axes([1.0, 0.75, .5, .5]) 
        N_plot = 1000
        # plot the zoomed portion
        time_axis = time_test[2000:2000+N_plot]
        sub_axes.plot(time_axis, np.sum(pred[2000:2000+N_plot, :]**2, axis=1)) 
        sub_axes.plot(time_axis, np.sum(df_test[:, 2000:2000+N_plot].T**2, axis=1))
        sub_axes.plot(time_axis, np.sum(df_test[idx_lst, 2000:2000+N_plot].T**2, axis=1))
        sub_axes.set_ylabel('$\sum_i^N u_i^2(t)$')
        sub_axes.set_xlabel('T')
        plt.savefig(f'{img_filepath}/kin_energy_test.png', dpi=100, facecolor="w", bbox_inches="tight")

        plt.close()
        # nrmse
        model_nrmse_time = nrmse_array(
            pred[window_size: window_size + N_plot, :],
            df_test[:, window_size: window_size + N_plot])
        N_plot = min(5*N_lyap, len(lyapunov_time), len(model_nrmse_time))
        plt.plot(lyapunov_time[:N_plot], model_nrmse_time[:N_plot])
        plt.hlines(y=0.4, xmin=lyapunov_time[0], xmax=lyapunov_time[N_plot])
        plt.xlabel('LT')
        plt.ylabel('NRMSE')
        plt.yscale('log')
        plt.savefig(f'{img_filepath}/nrmse.png', dpi=100, facecolor="w", bbox_inches="tight")
        plt.close()
       