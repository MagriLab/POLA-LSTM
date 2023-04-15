
import sys
import os
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.append('../../../')
from lstm.utils.supress_tf_warning import tensorflow_shutup
from lstm.utils.create_paths import make_folder_filepath
from lstm.utils.config import  load_config_to_argparse
warnings.simplefilter(action="ignore", category=FutureWarning)
tf.keras.backend.set_floatx('float64')
tensorflow_shutup()

ref_lyap=np.loadtxt('../Yael_CSV/L96/dim_20_lyapunov_exponents.txt')

le_sweep_models = ["generous-sweep-8", "misunderstood-sweep-12", "divine-sweep-2", "woven-sweep-4", "lively-sweep-12", "blooming-sweep-10", "celestial-sweep-8", "icy-sweep-9", "legendary-sweep-4", "tough-sweep-5", "dazzling-sweep-3"]

norm = 13.33
sweep_path = Path('../L96_d20_rk4')
for folder_name in ['D-10', 'D-12']: #list(filter(lambda x: x != 'images', next(os.walk(sweep_path))[1])):
    sweep_models = list(filter(lambda x: x != 'images', next(os.walk(sweep_path/folder_name))[1]))
    img_filepath_folder = make_folder_filepath(sweep_path / folder_name,  'images')
    for model_name in sweep_models:
        if model_name in le_sweep_models:
            print(model_name)
            model_path = sweep_path / folder_name/ model_name 
            args = load_config_to_argparse(model_path)
            
            epochs = max([int(i) for i in next(os.walk(model_path /'model'))[1]])
            
            img_filepath = make_folder_filepath(model_path, 'images')        
            # Compare this prediction with the LE prediction
            t_lyap = 1.55**(-1)
            N_lyap = int(t_lyap / (args.delta_t*args.upsampling))
            
            # Set up parameters for LE computation
            norm_time = 1
            N_lyap = int(t_lyap/(args.upsampling*args.delta_t))
            N = 1000* N_lyap
            Ntransient = max(int(N/100), args.window_size+2)
            N_test = N - Ntransient
            print(f'N:{N}, Ntran: {Ntransient}, Ntest: {N_test}')
            Ttot = np.arange(int(N_test/norm_time)) * (args.upsampling*args.delta_t) * norm_time
            N_test_norm = int(N_test/norm_time)
            print(f'N_test_norm: {N_test_norm}')
            le_dim = 20
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
                    
                plt.plot(fullspace, ref_lyap[-1,  :n_lyap],'k-s', markersize=8,label='target')
                plt.plot(fullspace, lyapunov_exp[-1, :n_lyap],'r-o', markersize=6,label='LSTM')
                # plt.plot(fullspace, np.append(np.append(lyapunov_exp[-1, :8], [0, 0]), lyapunov_exp[-1, 8:n_lyap-2]),'b-^', markersize=6,label='LSTM - 2 shifted like Vlachas')

                plt.legend()
                plt.savefig(img_filepath/f'{args.pi_weighing}_{N_test}_scatterplot_lyapunox_exp.png', dpi=100, facecolor="w", bbox_inches="tight")
                plt.savefig(img_filepath_folder/f'{args.pi_weighing}_{model_name}_scatterplot_lyapunox_exp.png', dpi=100, facecolor="w", bbox_inches="tight")
                plt.close()
                print(f'{model_name} : Lyapunov exponents: {lyapunov_exp[-1] } ')
            # else:
                # os.remove(img_filepath_folder/f'{args.pi_weighing}_{model_name}_scatterplot_lyapunox_exp.png')