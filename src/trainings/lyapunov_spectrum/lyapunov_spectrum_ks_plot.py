
import sys
import os
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.set_visible_devices(gpus[1], 'GPU')
        tf.config.set_logical_device_configuration(gpus[1], [tf.config.LogicalDeviceConfiguration(memory_limit=3072)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Virtual devices must be set before GPUs have been initialize
      print(e)

sys.path.append('../../../')
from lstm.utils.supress_tf_warning import tensorflow_shutup
from lstm.utils.create_paths import make_folder_filepath
from lstm.utils.config import load_config_to_argparse
warnings.simplefilter(action="ignore", category=FutureWarning)
tf.keras.backend.set_floatx('float64')
tensorflow_shutup()

ref_lyap=np.loadtxt('../Yael_CSV/KS/le_128_64.txt')

sweep_path = Path('/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/models/ks/128dof') 


for folder_name in ['pi-013', 'pi-016']:
    sweep_models = list(filter(lambda x: x != 'images', next(os.walk(sweep_path/folder_name))[1]))
    img_filepath_folder = make_folder_filepath(sweep_path / folder_name,  'images')
    for model_name in sweep_models:
        print(model_name)
        model_path = sweep_path / folder_name/ model_name 
        args = load_config_to_argparse(model_path)

        dim = 128
        n_random_idx = int(folder_name[-3:])
        
        epochs = max([int(i) for i in next(os.walk(model_path /'model'))[1]])
        print(f'Epochs {epochs}')
        img_filepath = make_folder_filepath(model_path, 'images')
        t_lyap = 0.08**(-1)
        N_lyap = int(t_lyap / (args.delta_t*args.upsampling))

        norm_time = 1
        N_lyap = int(t_lyap/(args.upsampling*args.delta_t))
        N = 500*N_lyap
        Ntransient = max(int(N/100), args.window_size+2)
        N_test = N - Ntransient
        print(f'N:{N}, Ntran: {Ntransient}, Ntest: {N_test}')
        Ttot = np.arange(int(N_test/norm_time)) * (args.upsampling*args.delta_t) * norm_time
        N_test_norm = int(N_test/norm_time)
        print(f'N_test_norm: {N_test_norm}')
        le_dim = 64
        file=model_path/f'{epochs}_lyapunov_exp_{N_test}.txt'
        if file.exists():
            lyapunov_exp = np.loadtxt(model_path/f'{epochs}_lyapunov_exp_{N_test}.txt')
            n_lyap=20
            fullspace = np.arange(1,n_lyap+1)
            fs=12
            ax = plt.figure().gca()
            # plt.title(r'KS, $26/160 \to 160$ dof')
            plt.rcParams.update({'font.size': fs})
            plt.grid(True,c='lightgray',linestyle='--', linewidth=0.5)
            plt.ylabel(r'$\lambda_k$',fontsize=fs)
            plt.xlabel(r'$k$',fontsize=fs)
                
            plt.plot(fullspace, ref_lyap[ :n_lyap],'k-s', markersize=8,label='target')
            plt.plot(fullspace, lyapunov_exp[-1, :n_lyap],'r-o', markersize=6,label='LSTM')
            # plt.plot(fullspace, np.append(np.append(lyapunov_exp[-1, :8], [0, 0]), lyapunov_exp[-1, 8:n_lyap-2]),'b-^', markersize=6,label='LSTM - 2 shifted like Vlachas')

            plt.legend()
            plt.savefig(img_filepath/f'{args.pi_weighing}_{N_test}_scatterplot_lyapunox_exp.png', dpi=100, facecolor="w", bbox_inches="tight")
            plt.savefig(img_filepath_folder/f'{args.pi_weighing}_{model_name}_scatterplot_lyapunox_exp.png', dpi=100, facecolor="w", bbox_inches="tight")
            plt.close()
            print(f'{model_name} : Lyapunov exponents: {lyapunov_exp[-1] } ')
        # else:
            # os.remove(img_filepath_folder/f'{args.pi_weighing}_{model_name}_scatterplot_lyapunox_exp.png')