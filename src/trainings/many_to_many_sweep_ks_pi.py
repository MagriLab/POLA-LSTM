import argparse
import sys
import random
import time
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#     try:
#         tf.config.set_visible_devices(gpus[1], 'GPU')
#         tf.config.set_logical_device_configuration(gpus[1], [tf.config.LogicalDeviceConfiguration(memory_limit=3072)])
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#      # Virtual devices must be set before GPUs have been initialize
#         print(e)
sys.path.append('../..')
from lstm.preprocessing.data_class import Dataclass
from lstm.preprocessing.data_processing import create_test_window
from lstm.utils.create_paths import make_folder_filepath
from lstm.utils.random_seed import reset_random_seeds
from lstm.utils.config import generate_config_ks
from lstm.utils.learning_rates import decayed_learning_rate
from lstm.closed_loop_tools_mtm import prediction
from lstm.postprocessing.nrmse import vpt
from lstm.lstm import LSTMRunner
from lstm.utils.loss_tracker import LossTracker
from lstm.postprocessing.lyapunov_spectrum import compute_lyapunov_exp, return_lyap_err_ks
from lstm.utils.early_stopping import EarlyStopper
from lstm.closed_loop_tools_mtm import create_test_window
plt.rcParams["figure.facecolor"] = "w"

tf.keras.backend.set_floatx('float64')

warnings.simplefilter(action="ignore", category=FutureWarning)



def run_lstm(args: argparse.Namespace):

    filepath = args.model_path / f"dd-{args.n_random_idx}" 
    reset_random_seeds()
    logs_checkpoint = make_folder_filepath(filepath, "logs")
    yaml_config_path = filepath / f'config.yml'
    generate_config_ks(yaml_config_path, args)

    t_lyap = args.lyap**(-1)
    N_lyap = int(t_lyap/(args.delta_t*args.upsampling))
    data = Dataclass(args)
    runner = LSTMRunner(args, system_name="KS_dd")
    model = runner.model
    loss_tracker = LossTracker(logs_checkpoint)
    early_stopper = EarlyStopper(patience=args.early_stop_patience, min_delta=1e-6)
    sys_dim = data.df_train.shape[0]
    print(f'Dimension of system {sys_dim}')

    for epoch in range(1, args.n_epochs+1):
        model.optimizer.learning_rate = decayed_learning_rate(epoch, args.learning_rate)
        start_time = time.time()
        train_loss_dd = 0
        train_loss_pi = 0
        for step, (x_batch_train, y_batch_train) in enumerate(data.train_dataset):
            loss_dd, loss_reg, loss_pi = runner.train_step_pi(x_batch_train, y_batch_train)
            train_loss_dd += loss_dd
            train_loss_pi += loss_pi
        loss_tracker.append_loss_to_tracker('train', train_loss_dd, loss_reg, train_loss_pi, step)

        print("Epoch: %d, Time: %.1fs , Batch: %d" % (epoch, time.time() - start_time, step))
        print("TRAINING: Data-driven loss: %4E; Physics-informed loss at epoch: %.4E" % (loss_dd/step, loss_pi/step))

        valid_loss_dd = 0
        valid_loss_pi = 0
        for val_step, (x_batch_valid, y_batch_valid) in enumerate(data.valid_dataset):
            val_loss_dd, valid_loss_reg, val_loss_pi = runner.valid_step_pi(x_batch_valid, y_batch_valid)
            valid_loss_dd += val_loss_dd
            valid_loss_pi += val_loss_pi
        loss_tracker.append_loss_to_tracker('valid', valid_loss_dd, valid_loss_reg, valid_loss_pi, val_step)
        print("VALIDATION: Data-driven loss: %4E; Physics-informed loss at epoch: %.4E; Full loss at epoch: %.4E" %
                (valid_loss_dd / val_step, valid_loss_pi / val_step, valid_loss_dd/val_step))
        
        loss_tracker.save_and_update_loss_txt(logs_checkpoint)
        early_stopper.early_stop(valid_loss_dd/val_step + args.pi_weighing*valid_loss_pi/val_step)
        wandb.log({'epochs': epoch,
                    'train_dd_loss': float(train_loss_dd/step),
                    'train_physics_loss': float(train_loss_pi/step),
                    'valid_dd_loss': float(valid_loss_dd/val_step),
                    'valid_full_dd_loss': float(valid_loss_dd/val_step),
                    'valid_physics_loss': float(valid_loss_pi/val_step)})

        if epoch % args.epoch_steps == 0 or early_stopper.stop:
            print("LEARNING RATE:%.2e" % model.optimizer.learning_rate)
            N = 10*N_lyap

            pred = prediction(model, data.df_valid, args.window_size, sys_dim, args.n_random_idx, N=N)
            lyapunov_time = np.arange(0, N/N_lyap, args.delta_t*args.upsampling/t_lyap)
            pred_horizon = lyapunov_time[vpt(pred[args.window_size:], data.df_valid[:, args.window_size:], 0.4)]
            lyapunov_exponents = compute_lyapunov_exp(
                create_test_window(data.df_valid, window_size=args.window_size),
                model, args, 10 * N_lyap, sys_dim, le_dim=10, idx_lst=data.idx_lst)

            max_lyap_percent_error, l_2_error = return_lyap_err_ks(data.ref_lyap, lyapunov_exponents)
            print(max_lyap_percent_error, l_2_error)
            model_checkpoint = filepath / "model" / f"{epoch}" / "weights"
            model.save_weights(model_checkpoint)


            wandb.log({'epochs': epoch,
                'pi_weighing': float(pi_weighing),
                'pred_horizon': float(pred_horizon),
                'max_lyap_err': float(max_lyap_percent_error),
                'lyap_l2_error': float(l_2_error)
                })
            if early_stopper.stop:
                print('EARLY STOPPING')
                early_stopper.reset_counter()
                break
                
        

    loss_tracker.loss_arr_to_tensorboard(logs_checkpoint)
    model_checkpoint = filepath / "model" / f"{epoch}" / "weights"
    model.save_weights(model_checkpoint)
    tf.keras.backend.clear_session()    

parser = argparse.ArgumentParser(description='Open Loop')
parser.add_argument('--dd_loss_label', type=str, default="full")
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--epoch_steps', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--n_cells', type=int, default=200)
parser.add_argument('--oloop_train', default=True, action='store_true')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--activation', type=str, default='Tanh')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--washout', type=int, default=2)

parser.add_argument('--pi_weighing', type=float, default=0.0)
parser.add_argument('--early_stop_patience', type=int, default=100)
parser.add_argument('--reg_weighing', type=float, default=1e-9)
parser.add_argument('--normalised', default=False, action='store_true')
parser.add_argument('--t_0', type=int, default=0)
parser.add_argument('--t_trans', type=int, default=250)
parser.add_argument('--t_end', type=int, default=25000)
parser.add_argument('--delta_t', type=float, default=0.25)
parser.add_argument('--total_n', type=float, default=99000)
parser.add_argument('--window_size', type=int, default=25)
parser.add_argument('--sys_dim', type=float, default=128)

parser.add_argument('--upsampling', type=int, default=1)
parser.add_argument('--n_random_idx', type=int, default=10)
parser.add_argument('--lyap', type=float, default=0.08)
parser.add_argument('--signal_noise_ratio', type=int, default=0)
parser.add_argument('--train_ratio', type=float, default=0.25)
parser.add_argument('--valid_ratio', type=float, default=0.05)
parser.add_argument('--standard_norm', type=float, default=3.58)
parser.add_argument('--M', type=int, default=64)
parser.add_argument('--N', type=int, default=128)
parser.add_argument('--h', type=float, default=0.25)
parser.add_argument('--d', type=int, default=62.8319)


# arguments to define paths
parser.add_argument('-lyp', '--lyap_path', type=Path, required=True)
parser.add_argument('-mp', '--model_path', type=Path, required=True)
parser.add_argument('-dp', '--data_path', type=Path, required=True)

parsed_args = parser.parse_args()


yaml_config_path = parsed_args.model_path / f'config.yml'

generate_config_ks(yaml_config_path, parsed_args)
print(f'REG weight {parsed_args.reg_weighing}')
run_lstm(parsed_args)
# python many_to_many_sweep_ks_pi.py  -dp Yael_CSV/KS/KS_128_dx62_99000_stand_3.58_deltat_0.25_M_64_trans.csv -mp KS/128dof_dd/ -lyp Yael_CSV/KS/le_128_64.txt

