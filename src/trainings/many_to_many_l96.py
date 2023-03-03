
import argparse
import sys
import time
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
sys.path.append('../..')
from lstm.preprocessing.data_processing import df_train_valid_test_split, create_df_nd_random_md_mtm_idx
from lstm.utils.random_seed import reset_random_seeds
from lstm.utils.config import generate_config
from lstm.postprocessing.nrmse import vpt
from lstm.closed_loop_tools_mtm import prediction
from lstm.utils.early_stopping import EarlyStopper
from lstm.closed_loop_tools_mtm import create_test_window
from lstm.postprocessing.lyapunov_spectrum import compute_lyapunov_exp, return_lyap_err
from lstm.utils.learning_rates import decayed_learning_rate
from lstm.utils.create_paths import make_folder_filepath
from lstm.lstm import LSTMRunner
from lstm.utils.loss_tracker import LossTracker
physical_devices = tf.config.list_physical_devices('GPU')
try:
    # Disable first GPU
    tf.config.set_visible_devices(physical_devices[0:], 'GPU')
    logical_devices = tf.config.list_logical_devices('GPU')
    print('Number of used GPUs: ', len(logical_devices))
    # Logical device was not created for first GPU
    assert len(logical_devices) == len(physical_devices) - 1
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
plt.rcParams["figure.facecolor"] = "w"

tf.keras.backend.set_floatx('float64')

warnings.simplefilter(action="ignore", category=FutureWarning)



def run_lstm(args: argparse.Namespace):

    reset_random_seeds()
    ref_lyap = np.loadtxt(args.lyap_path)
    filepath = args.data_path 
    logs_checkpoint = make_folder_filepath(filepath, "logs") 
    yaml_config_path = filepath / f'config.yml'
    generate_config(yaml_config_path, args)
    mydf = np.genfromtxt(args.config_path, delimiter=",").astype(np.float64)

    df_train, df_valid, df_test = df_train_valid_test_split(
        mydf[1:, :: args.upsampling],
        train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)

    sys_dim = df_train.shape[0]
    print(f'Dimension of system {sys_dim}')
    t_lyap = args.lyap**(-1)
    N_lyap = int(t_lyap/(args.delta_t*args.upsampling))
    # Windowing
    idx_lst, train_dataset = create_df_nd_random_md_mtm_idx(
        df_train.transpose(),
        args.window_size, args.batch_size, df_train.shape[0],
        n_random_idx=args.n_random_idx)
    _, valid_dataset = create_df_nd_random_md_mtm_idx(
        df_valid.transpose(),
        args.window_size, args.batch_size, 1, n_random_idx=args.n_random_idx)
    for batch, label in train_dataset.take(1):
        print(f'Shape of batch: {batch.shape} \n Shape of Label {label.shape}')
    runner = LSTMRunner(args, system_name='l96')
    model = runner.model
    loss_tracker = LossTracker(logs_checkpoint)
    early_stopper = EarlyStopper(patience=args.early_stop_patience, min_delta=1e-6)


    for epoch in range(1, args.n_epochs+1):
        model.optimizer.learning_rate = decayed_learning_rate(epoch, args.learning_rate)
        start_time = time.time()
        train_loss_dd = 0
        train_loss_pi = 0
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_dd, loss_reg, loss_pi = runner.train_step_pi(x_batch_train, y_batch_train)
            train_loss_dd += loss_dd
            train_loss_pi += loss_pi
        loss_tracker.append_loss_to_tracker('train', train_loss_dd, loss_reg, train_loss_pi, step)

        print("Epoch: %d, Time: %.1fs , Batch: %d" % (epoch, time.time() - start_time, step))
        print("TRAINING: Data-driven loss: %4E; Physics-informed loss at epoch: %.4E" % (loss_dd, loss_pi))

        valid_loss_dd = 0
        valid_loss_pi = 0
        for val_step, (x_batch_valid, y_batch_valid) in enumerate(valid_dataset):
            val_loss_dd, valid_loss_reg, val_loss_pi = runner.valid_step_pi(x_batch_valid, y_batch_valid)
            valid_loss_dd += val_loss_dd
            valid_loss_pi += val_loss_pi
        loss_tracker.append_loss_to_tracker('valid', valid_loss_dd, valid_loss_reg, valid_loss_pi, val_step)
        print("VALIDATION: Data-driven loss: %4E; Physics-informed loss at epoch: %.4E" %
              (valid_loss_dd / val_step, valid_loss_pi / val_step))

        loss_tracker.save_and_update_loss_txt(logs_checkpoint)
        if epoch % args.epoch_steps == 0 or early_stopper.stop:
            
            model_checkpoint = filepath / "model" / f"{epoch}" / "weights"
            model.save_weights(model_checkpoint)
            N = 10*N_lyap
            pred = prediction(model, df_valid, args.window_size, sys_dim, args.n_random_idx, N=N)
            lyapunov_time = np.arange(0, N/N_lyap, args.delta_t*args.upsampling/t_lyap)
            pred_horizon = lyapunov_time[vpt(pred[args.window_size:], df_valid[:, args.window_size:], 0.4)]
            lyapunov_exponents = compute_lyapunov_exp(
                create_test_window(df_test, window_size=args.window_size),
                model, args, 50 * N_lyap, sys_dim, idx_lst=idx_lst)

            max_lyap_percent_error, l_2_error = return_lyap_err(ref_lyap, lyapunov_exponents)
            print(f"Prediction horizon {pred_horizon} LT, Max Lyap Err: {max_lyap_percent_error}, L2 Error Lyap exp:{l_2_error}")
            if early_stopper.stop:
                print('EARLY STOPPING')
                break
    loss_tracker.loss_arr_to_tensorboard(logs_checkpoint)




parser = argparse.ArgumentParser(description='Open Loop')
parser.add_argument('--n_epochs', type=int, default=5)
parser.add_argument('--epoch_steps', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--n_cells', type=int, default=50)
parser.add_argument('--oloop_train', default=True, action='store_true')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--activation', type=str, default='Tanh')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--sys_dim', type=float, default=10)
parser.add_argument('--pi_weighing', type=float, default=0.0)
parser.add_argument('--early_stop_patience', type=int, default=100)
parser.add_argument('--reg_weighing', type=float, default=0.0)
parser.add_argument('--normalised', default=False, action='store_true')

parser.add_argument('--standard_norm',  type=float, default=13.33)
parser.add_argument('--t_0', type=int, default=0)
parser.add_argument('--t_trans', type=int, default=100)
parser.add_argument('--t_end', type=int, default=425)
parser.add_argument('--upsampling', type=int, default=1)
parser.add_argument('--n_random_idx', type=int, default=10)
parser.add_argument('--lyap', type=float, default=1.2)
parser.add_argument('--delta_t', type=float, default=0.01)
parser.add_argument('--total_n', type=float, default=42500)
parser.add_argument('--window_size', type=int, default=25)
parser.add_argument('--signal_noise_ratio', type=int, default=0)
parser.add_argument('--train_ratio', type=float, default=0.1)
parser.add_argument('--valid_ratio', type=float, default=0.1)
# arguments to define paths
# parser.add_argument('-idp', '--input_data_path', type=Path, required=True)
parser.add_argument('-dp', '--data_path', type=Path, required=True)
parser.add_argument('-cp', '--config_path', type=Path, required=True)
parser.add_argument('-lyp', '--lyap_path', type=Path, required=True)
parsed_args = parser.parse_args()


yaml_config_path = parsed_args.data_path / f'config.yml'


generate_config(yaml_config_path, parsed_args)
print(f'REG weight {parsed_args.reg_weighing}')
run_lstm(parsed_args)

# python many_to_many_l96.py  -cp Yael_CSV/L96/dim_10_rk4_42500_0.01_stand13.33_trans.csv -dp l96/D10/test/ -lyp Yael_CSV/L96/dim_10_lyapunov_exponents.txt
