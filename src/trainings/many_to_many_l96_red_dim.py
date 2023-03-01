
import argparse
import os
import sys
import time
import warnings
from pathlib import Path
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


sys.path.append('../..')
from lstm.closed_loop_tools_mtm import prediction
from lstm.postprocessing.nrmse import vpt
from lstm.preprocessing.data_processing import df_train_valid_test_split
from lstm.utils.random_seed import reset_random_seeds
from lstm.utils.config import generate_config
from lstm.postprocessing.loss_saver import loss_arr_to_tensorboard, save_and_update_loss_txt
from lstm.lstm_model import build_pi_model
from lstm.utils.learning_rates import decayed_learning_rate
from lstm.utils.create_paths import make_folder_filepath
from lstm.lstm import LSTMRunner
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

def create_df_nd_random_md_mtm(series, window_size, batch_size, shuffle_buffer, n_random_idx=5, shuffle_window=10):
    n = series.shape[1]
    m = series.shape[0]
    random.seed(0)
    idx_lst = random.sample(range(n), n_random_idx)
    idx_lst.sort()
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(size=window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.shuffle(m*shuffle_window)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(
        lambda window: (tf.gather(window[:-1, :], idx_lst, axis=1), window[1:])
    )
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, n_random_idx], [None, n]))
    return dataset


def run_lstm(args: argparse.Namespace):

    reset_random_seeds()
    image_filepath = make_folder_filepath(args.data_path, "images") 
    logs_checkpoint = make_folder_filepath(args.data_path, "logs") 

    mydf = np.genfromtxt(args.config_path, delimiter=",").astype(np.float64)
    # mydf[1:,:] = mydf[1:,:]/(np.max(mydf[1:,:]) - np.min(mydf[1:,:]) )
    df_train, df_valid, df_test = df_train_valid_test_split(mydf[1:, ::args.upsampling], train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)
    sys_dim = df_train.shape[0]
    print(f'Dimension of system {sys_dim}')

    t_lyap = 0.93**(-1)
    norm_time = 1
    N_lyap = int(t_lyap/(args.delta_t*args.upsampling))
    # Windowing
    train_dataset = create_df_nd_random_md_mtm(df_train.transpose(), args.window_size, args.batch_size, df_train.shape[0], n_random_idx=4)
    valid_dataset = create_df_nd_random_md_mtm(df_valid.transpose(), args.window_size, args.batch_size, 1, n_random_idx=4)
    for batch, label in train_dataset.take(1):
        print(f'Shape of batch: {batch.shape} \n Shape of Label {label.shape}')
    runner = LSTMRunner(args, system_name='l96')
    model = runner.model

    @tf.function
    def train_step_pi(x_batch_train, y_batch_train, weight=1, normalised=True):
        with tf.GradientTape() as tape:
            one_step_pred = model(x_batch_train, training=True)
            mse = tf.keras.losses.MeanSquaredError()
            loss_dd = mse(y_batch_train, one_step_pred) 
            loss_pi = 0 #mse(tf.math.reduce_sum(y_batch_train, axis=2), tf.math.reduce_sum(one_step_pred, axis=2))  
            loss_value = loss_dd + weight*loss_pi
        grads = tape.gradient(loss_value, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_dd, loss_pi

    @tf.function
    def valid_step_pi(x_batch_valid, y_batch_valid, normalised=True):
        val_logit = model(x_batch_valid, training=False)
        mse = tf.keras.losses.MeanSquaredError()
        loss_dd = mse(y_batch_valid, val_logit) 
        loss_pi = 0 #mse(tf.math.reduce_sum(x_batch_valid, axis=2), tf.math.reduce_sum(val_logit, axis=2))  
        return loss_dd, loss_pi

    train_loss_dd_tracker = np.array([])
    train_loss_pi_tracker = np.array([])
    valid_loss_dd_tracker = np.array([])
    valid_loss_pi_tracker = np.array([])

    for epoch in range(1, args.n_epochs+1):
        model.optimizer.learning_rate = decayed_learning_rate(epoch, args.learning_rate)
        start_time = time.time()
        train_loss_dd = 0
        train_loss_pi = 0
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_dd, loss_pi = train_step_pi(x_batch_train, y_batch_train,
                                            weight=args.reg_weighing, normalised=args.normalised)
            train_loss_dd += loss_dd
            train_loss_pi += loss_pi
        train_loss_dd_tracker = np.append(train_loss_dd_tracker, train_loss_dd/step)
        train_loss_pi_tracker = np.append(train_loss_pi_tracker, train_loss_pi/step)


        print("Epoch: %d, Time: %.1fs , Batch: %d" % (epoch, time.time() - start_time, step))
        print("TRAINING: Data-driven loss: %4E; Physics-informed loss at epoch: %.4E" % (loss_dd, loss_pi))

        valid_loss_dd = 0
        valid_loss_pi = 0
        for val_step, (x_batch_valid, y_batch_valid) in enumerate(valid_dataset):
            val_loss_dd, val_loss_pi = valid_step_pi(x_batch_valid, y_batch_valid)
            valid_loss_dd += val_loss_dd
            valid_loss_pi += val_loss_pi
        valid_loss_dd_tracker = np.append(valid_loss_dd_tracker, valid_loss_dd/val_step)
        valid_loss_pi_tracker = np.append(valid_loss_pi_tracker, valid_loss_pi/val_step)
        print("VALIDATION: Data-driven loss: %4E; Physics-informed loss at epoch: %.4E" %
              (valid_loss_dd / val_step, valid_loss_pi / val_step))

        if epoch % args.epoch_steps == 0:
            print(f"LEARNING RATE:{model.optimizer.learning_rate.numpy()}")
            model_checkpoint = args.data_path / "model" / f"{epoch}" / "weights"
            model.save_weights(model_checkpoint)
            logs_epoch_checkpoint = logs_checkpoint / f"{epoch}"
            loss_arr_to_tensorboard(logs_epoch_checkpoint, train_loss_dd_tracker, train_loss_pi_tracker,
                                    valid_loss_dd_tracker, valid_loss_pi_tracker)
            save_and_update_loss_txt(
                logs_checkpoint, train_loss_dd_tracker[-args.epoch_steps:],
                train_loss_pi_tracker[-args.epoch_steps:],
                valid_loss_dd_tracker[-args.epoch_steps:],
                valid_loss_pi_tracker[-args.epoch_steps:])
            N=10*N_lyap
            pred = prediction(model, df_valid, args.window_size, sys_dim, 4, N=N)
            lyapunov_time = np.arange(0, N/N_lyap, args.delta_t*args.upsampling/t_lyap)
            pred_horizon = lyapunov_time[vpt(pred[args.window_size:], df_valid[:, args.window_size:], 0.4)]
            print(f"Prediction horizon {pred_horizon}")

parser = argparse.ArgumentParser(description='Open Loop')
parser.add_argument('--n_epochs', type=int, default=5000)
parser.add_argument('--epoch_steps', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--n_cells', type=int, default=50)
parser.add_argument('--oloop_train', default=True, action='store_true')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--activation', type=str, default='Tanh')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--sys_dim', type=float, default=6)
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
# parser.add_argument('-lyp', '--lyap_path', type=Path, required=True)
parser.add_argument('-dp', '--data_path', type=Path, required=True)
parser.add_argument('-cp', '--config_path', type=Path, required=True)

parsed_args = parser.parse_args()


yaml_config_path = parsed_args.data_path / f'config.yml'


generate_config(yaml_config_path, parsed_args)
print(f'Physics weight {parsed_args.reg_weighing}')
run_lstm(parsed_args)


# python many_to_many_l96_red_dim.py -dp test/ -cp ../diff_dyn_sys/lorenz96/CSV/D6/dim_6_rk4_42500_0.01_stand13.33_trans.csv