
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

from lstm.preprocessing.data_processing import (create_df_nd_mtm,
                                                df_train_valid_test_split,
                                                train_valid_test_split)
from lstm.utils.random_seed import reset_random_seeds
from lstm.utils.config import generate_config
from lstm.postprocessing.loss_saver import loss_arr_to_tensorboard
from lstm.postprocessing import plots_mtm
from lstm.lstm_model import build_pi_model
from lstm.loss import loss_oloop, norm_loss_pi_many
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

def create_df_nd_random_md_mtm(series, window_size, batch_size, shuffle_buffer, idx_skip=5, shuffle_window=10):
    n = series.shape[1]
    m = series.shape[0]
    random.seed(0)
    batch_shape=series[:, ::idx_skip].shape
    idx_lst = random.sample(range(n), batch_shape[1])
    idx_lst.sort()
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(size=window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.shuffle(m*shuffle_window)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(
        lambda window: (tf.gather(window[:-1, :], idx_lst, axis=1), window[1:])
    )
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, batch_shape[1]], [None, n]))
    return dataset


def run_lstm(args: argparse.Namespace):

    reset_random_seeds()

    filepath = args.data_path
    if not os.path.exists(filepath / "images"):
        os.makedirs(filepath / "images")
    logs_checkpoint = filepath / "logs"
    if not os.path.exists(logs_checkpoint):
        os.makedirs(logs_checkpoint)
    mydf = np.genfromtxt(args.config_path, delimiter=",").astype(np.float64)
    # mydf[1:,:] = mydf[1:,:]/(np.max(mydf[1:,:]) - np.min(mydf[1:,:]) )
    df_train, df_valid, df_test = df_train_valid_test_split(mydf[1:, :], train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)
    time_train, time_valid, time_test = train_valid_test_split(mydf[0, :], train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)
    ks_dim = df_train.shape[0]
    print(f'Dimension of system {ks_dim}')
    # Windowing
    train_dataset = create_df_nd_random_md_mtm(df_train.transpose(), args.window_size, args.batch_size, df_train.shape[0], idx_skip=5)
    valid_dataset = create_df_nd_random_md_mtm(df_valid.transpose(), args.window_size, args.batch_size, 1, idx_skip=5)
    for batch, label in train_dataset.take(1):
        print(f'Shape of batch: {batch.shape} \n Shape of Label {label.shape}')
    model = build_pi_model(args.n_cells, dim=ks_dim)
    # model.load_weights(args.input_data_path)

    def decayed_learning_rate(step):
        decay_steps = 1000
        decay_rate = 0.75
        initial_learning_rate = args.learning_rate
        # careful here! step includes batch steps in the tf framework
        return initial_learning_rate * decay_rate ** (step / decay_steps)

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
        model.optimizer.learning_rate = decayed_learning_rate(epoch)
        start_time = time.time()
        train_loss_dd = 0
        train_loss_pi = 0
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_dd, loss_pi = train_step_pi(x_batch_train, y_batch_train,
                                            weight=args.physics_weighing, normalised=args.normalised)
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
            print("LEARNING RATE:%.2e" % model.optimizer.learning_rate)
                        
            model_checkpoint = filepath / "model" / f"{epoch}" / "weights"
            model.save_weights(model_checkpoint)
            
            model_checkpoint = filepath / "model" / f"{epoch}" / "weights"

            np.savetxt(logs_checkpoint/f"training_loss_dd_{epoch}.txt", train_loss_dd_tracker)
            np.savetxt(logs_checkpoint/f"training_loss_pi_{epoch}.txt", train_loss_pi_tracker)
            np.savetxt(logs_checkpoint/f"valid_loss_dd_{epoch}.txt", valid_loss_dd_tracker)
            np.savetxt(logs_checkpoint/f"valid_loss_pi_{epoch}.txt", valid_loss_pi_tracker)
            logs_epoch_checkpoint = filepath / "logs"/ f"{epoch}"
            loss_arr_to_tensorboard(logs_epoch_checkpoint, train_loss_dd_tracker, train_loss_pi_tracker,
                                    valid_loss_dd_tracker, valid_loss_pi_tracker)


    if not os.path.exists(logs_checkpoint):
        os.makedirs(logs_checkpoint)
    np.savetxt(logs_checkpoint/f"training_loss_dd.txt", train_loss_dd_tracker)
    np.savetxt(logs_checkpoint/f"training_loss_pi.txt", train_loss_pi_tracker)
    np.savetxt(logs_checkpoint/f"valid_loss_dd.txt", valid_loss_dd_tracker)
    np.savetxt(logs_checkpoint/f"valid_loss_pi.txt", valid_loss_pi_tracker)
    loss_arr_to_tensorboard(logs_checkpoint, train_loss_dd_tracker, train_loss_pi_tracker,
                            valid_loss_dd_tracker, valid_loss_pi_tracker)


parser = argparse.ArgumentParser(description='Open Loop')

parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--epoch_steps', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--n_cells', type=int, default=200)
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
parser.add_argument('--t_trans', type=int, default=250)
parser.add_argument('--t_end', type=int, default=25000)
parser.add_argument('--delta_t', type=float, default=0.25)
parser.add_argument('--total_n', type=float, default=99000)
parser.add_argument('--window_size', type=int, default=25)
parser.add_argument('--signal_noise_ratio', type=int, default=0)
parser.add_argument('--train_ratio', type=float, default=0.5)
parser.add_argument('--valid_ratio', type=float, default=0.1)

# arguments to define paths
# parser.add_argument( '--experiment_path', type=Path, required=True)
# parser.add_argument('-idp', '--input_data_path', type=Path, required=True)
# parser.add_argument('--log-board_path', type=Path, required=True)
parser.add_argument('-dp', '--data_path', type=Path, required=True)
parser.add_argument('-cp', '--config_path', type=Path, required=True)

parsed_args = parser.parse_args()


yaml_config_path = parsed_args.data_path / f'config.yml'


generate_config(yaml_config_path, parsed_args)
print(f'Physics weight {parsed_args.physics_weighing}')
run_lstm(parsed_args)

# python many_to_many_ks.py -dp ../models/ks/D3-40_34/60000/50-25/ -cp ../diff_dyn_sys/KS_flow/CSV/KS_40_to_50_dx200_rk4_240000_stand_3.76_trans.csv


# python many_to_many_ks.py -dp ../models/ks/D128-100/40000/20-80/ -cp KS_128_dx100_rk4_50000_stand_3.84_trans.csv


# python many_to_many_ks_red_dim.py -dp ../models/KS/D160-5n-random/42500/25-200/ -cp ../diff_dyn_sys/KS_flow/CSV/KS_160_dx60_rk4_99000_stand_3.52_deltat_0.25_trans.csv