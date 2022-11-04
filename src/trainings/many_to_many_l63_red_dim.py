import argparse
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.append('../..')

from lstm.utils.learning_rates import decayed_learning_rate
from lstm.lstm_model import build_pi_model, train_step_dd, valid_step_dd
from lstm.postprocessing import plots_mtm
from lstm.postprocessing.loss_saver import loss_arr_to_tensorboard, save_and_update_loss_txt
from lstm.utils.config import generate_config
from lstm.utils.random_seed import reset_random_seeds
from lstm.preprocessing.data_processing import (df_train_valid_test_split,
                                                train_valid_test_split)
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


def create_df_nd_md_mtm(series, window_size, batch_size, shuffle_buffer, idx_skip=0, shuffle_window=10):
    n = series.shape[1]
    m = series.shape[0]
    print(n, m)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(size=window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.shuffle(m*shuffle_window)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(
        lambda window: (window[:-1, 1:2], window[1:])
    )
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, 1], [None, n]))
    return dataset


def run_lstm(args: argparse.Namespace):

    reset_random_seeds()

    image_filepath = args.data_path / "images"
    image_filepath.mkdir(parents=True, exist_ok=True)
    logs_checkpoint = args.data_path / "logs"
    logs_checkpoint.mkdir(parents=True, exist_ok=True)

    mydf = np.genfromtxt(args.config_path, delimiter=",").astype(np.float64)
    df_train, df_valid, df_test = df_train_valid_test_split(
        mydf[1:, :], train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)
    time_train, time_valid, time_test = train_valid_test_split(
        mydf[0, :], train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)
    ks_dim = df_train.shape[0]
    print(f'Dimension of system {ks_dim}')
    # Windowing
    train_dataset = create_df_nd_md_mtm(df_train.transpose(), args.window_size,
                                        args.batch_size, df_train.shape[0], idx_skip=0)
    valid_dataset = create_df_nd_md_mtm(df_valid.transpose(), args.window_size, args.batch_size, 1, idx_skip=0)
    for batch, label in train_dataset.take(1):
        print(f'Shape of batch: {batch.shape}; Shape of Label {label.shape}')
    model = build_pi_model(args.n_cells, dim=ks_dim)
    # model.load_weights(args.input_data_path)

    train_loss_dd_tracker = np.array([])
    train_loss_pi_tracker = np.array([])
    valid_loss_dd_tracker = np.array([])
    valid_loss_pi_tracker = np.array([])

    for epoch in range(1, args.n_epochs+1):
        print(f"Epoch: {epoch}")
        model.optimizer.learning_rate = decayed_learning_rate(epoch, args.learning_rate)
        start_time = time.time()
        train_loss_dd = 0
        train_loss_pi = 0
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_dd, loss_pi = train_step_dd(model, x_batch_train, y_batch_train,
                                             weight=args.physics_weighing)
            train_loss_dd += loss_dd
            train_loss_pi += loss_pi
        train_loss_dd_tracker = np.append(train_loss_dd_tracker, train_loss_dd/step)
        train_loss_pi_tracker = np.append(train_loss_pi_tracker, train_loss_pi/step)
        train_step_time = time.time()
        print(f"TRAINING   {train_step_time - start_time :.2f}s: Data-driven loss: {train_loss_dd/step:.4E}; Physics-informed loss at epoch: {train_loss_pi/step:.4E}")

        valid_loss_dd = 0
        valid_loss_pi = 0
        for val_step, (x_batch_valid, y_batch_valid) in enumerate(valid_dataset):
            val_loss_dd, val_loss_pi = valid_step_dd(model, x_batch_valid, y_batch_valid)
            valid_loss_dd += val_loss_dd
            valid_loss_pi += val_loss_pi
        valid_loss_dd_tracker = np.append(valid_loss_dd_tracker, valid_loss_dd/val_step)
        valid_loss_pi_tracker = np.append(valid_loss_pi_tracker, valid_loss_pi/val_step)
        valid_step_time = time.time()
        print(f"VALIDATION {valid_step_time - train_step_time:.2f}s: Data-driven loss: {valid_loss_dd / val_step:.4E}; Physics-informed loss at epoch: {valid_loss_pi / val_step:.4E}")

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


parser = argparse.ArgumentParser(description='Open Loop')

parser.add_argument('--n_epochs', type=int, default=5000)
parser.add_argument('--epoch_steps', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--n_cells', type=int, default=10)
parser.add_argument('--oloop_train', default=True, action='store_true')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--activation', type=str, default='Tanh')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--l2_regularisation', type=float, default=0)
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--early_stop_patience', type=int, default=0)
parser.add_argument('--physics_weighing', type=float, default=0.0)
parser.add_argument('--normalised', default=False, action='store_true')
parser.add_argument('--t_0', type=int, default=0)
parser.add_argument('--t_trans', type=int, default=20)
parser.add_argument('--t_end', type=int, default=100)
parser.add_argument('--delta_t', type=int, default=0.01)
parser.add_argument('--total_n', type=float, default=8000)
parser.add_argument('--window_size', type=int, default=100)
parser.add_argument('--signal_noise_ratio', type=int, default=0)
parser.add_argument('--train_ratio', type=float, default=0.25)
parser.add_argument('--valid_ratio', type=float, default=0.1)

# arguments to define paths
parser.add_argument('-dp', '--data_path', type=Path, required=True)
parser.add_argument('-cp', '--config_path', type=Path, required=True)

parsed_args = parser.parse_args()
yaml_config_path = parsed_args.data_path / f'config.yml'
generate_config(yaml_config_path, parsed_args)

print(f'Physics weight {parsed_args.physics_weighing}')
run_lstm(parsed_args)


# python many_to_many_l63_red_dim.py -dp ../models/l63/some_test/ -cp /Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/diff_dyn_sys/lorenz63/CSV/100000/rk4_100000_norm_trans.csv
