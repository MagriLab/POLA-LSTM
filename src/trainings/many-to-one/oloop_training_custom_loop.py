
import argparse
import datetime
import importlib
import os
import random
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
sys.path.append('../..')

from lstm.loss import loss_oloop, norm_pi_loss, norm_pi_loss_two_step
from lstm.lstm_model import build_pi_model
from lstm.postprocessing import plots
from lstm.postprocessing.tensorboard_converter import loss_arr_to_tensorboard
from lstm.preprocessing.data_processing import (create_df_3d,
                                                df_train_valid_test_split,
                                                train_valid_test_split)

from lstm.closed_loop_tools_mto import append_label_to_batch, split_window_label
from lstm.utils.config import generate_config
from lstm.utils.random_seed import reset_random_seeds


warnings.simplefilter(action="ignore", category=FutureWarning)


plt.rcParams["figure.facecolor"] = "w"

tf.keras.backend.set_floatx('float64')

def run_lstm(args: argparse.Namespace):

    reset_random_seeds()

    filepath = args.data_path
    if not os.path.exists(filepath / "images"):
        os.makedirs(filepath / "images")

    mydf = np.genfromtxt(args.config_path, delimiter=",").astype(np.float64)
    df_train, df_valid, df_test = df_train_valid_test_split(mydf[1:, :], train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)
    time_train, time_valid, time_test = train_valid_test_split(mydf[0, :], train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)

    # Windowing
    lorenz_dim = 3
    train_dataset = create_df_3d(df_train.transpose(), args.window_size, args.batch_size, df_train.shape[0])
    valid_dataset = create_df_3d(df_valid.transpose(), args.window_size, args.batch_size, 1)

    model = build_pi_model(args.n_cells)
    model.load_weights(args.input_data_path)

    def decayed_learning_rate(step):
        initial_learning_rate = args.learning_rate
        decay_steps = 1000
        decay_rate = 0.75
        # careful here! step includes batch steps in the tf framework
        return initial_learning_rate * decay_rate ** (step / decay_steps)

    @tf.function
    def train_step_pi(x_batch_train, y_batch_train, weight=1, normalised=True):
        with tf.GradientTape() as tape:
            one_step_pred = model(x_batch_train, training=True)
            new_batch = split_window_label(append_label_to_batch(x_batch_train, one_step_pred))
            two_step_pred = model(new_batch, training=True)
            loss_dd = loss_oloop(y_batch_train, one_step_pred)
            loss_pi = norm_pi_loss_two_step(one_step_pred, two_step_pred, norm=normalised)
            loss_value = loss_dd + weight*loss_pi
        grads = tape.gradient(loss_value, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_dd, loss_pi

    @tf.function
    def valid_step_pi(x_batch_valid, y_batch_valid, normalised=True):
        val_logit = model(x_batch_valid, training=False)
        loss_dd = loss_oloop(y_batch_valid, val_logit)
        new_batch = split_window_label(append_label_to_batch(x_batch_valid, val_logit))
        two_step_pred = model(new_batch, training=False)
        loss_pi = norm_pi_loss_two_step(val_logit, two_step_pred, norm=normalised)
        return loss_dd, loss_pi

    train_loss_dd_tracker = np.array([])
    train_loss_pi_tracker = np.array([])
    valid_loss_dd_tracker = np.array([])
    valid_loss_pi_tracker = np.array([])
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=args.learning_rate, decay_steps=1000, decay_rate=0.5)
    # tf.keras.backend.set_value(model.optimizer.learning_rate, lr_schedule)

    for epoch in range(args.n_epochs+1):
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
            predictions = plots.plot_closed_loop_lya(
                model,
                epoch,
                time_test,
                df_test,
                n_length=500,
                window_size=args.window_size,
                img_filepath=filepath / "images" / f"pred_{epoch}.png",
            )
            plots.plot_phase_space(
                predictions,
                epoch,
                df_test,
                window_size=args.window_size,
                img_filepath=filepath / "images" / f"phase_{epoch}.png",
            )

            model_checkpoint = filepath / "model" / f"{epoch}" / "weights"
            model.save_weights(model_checkpoint)
            logs_checkpoint = filepath / "logs"
    if not os.path.exists(logs_checkpoint):
        os.makedirs(logs_checkpoint)
    np.savetxt(logs_checkpoint/f"training_loss_dd.txt", train_loss_dd_tracker)
    np.savetxt(logs_checkpoint/f"training_loss_pi.txt", train_loss_pi_tracker)
    np.savetxt(logs_checkpoint/f"valid_loss_dd.txt", valid_loss_dd_tracker)
    np.savetxt(logs_checkpoint/f"valid_loss_pi.txt", valid_loss_pi_tracker)
    loss_arr_to_tensorboard(logs_checkpoint, train_loss_dd_tracker, train_loss_pi_tracker,
                            valid_loss_dd_tracker, valid_loss_pi_tracker)


parser = argparse.ArgumentParser(description='Open Loop')
# arguments for configuration parameters
parser.add_argument('--n_epochs', type=int, default=10000)
parser.add_argument('--epoch_steps', type=int, default=500)
 
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_cells', type=int, default=10)
parser.add_argument('--oloop_train', default=True, action='store_true')
 
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--activation', type=str, default='Tanh')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--l2_regularisation', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
 
parser.add_argument('--early_stop_patience', type=int, default=10)
 
parser.add_argument('--physics_weighing', type=float, default=1.0)

parser.add_argument('--normalised', default=True, action='store_true')
parser.add_argument('--t_0', type=int, default=0)
parser.add_argument('--t_trans', type=int, default=20)
parser.add_argument('--t_end', type=int, default=100)
parser.add_argument('--delta_t', type=int, default=0.01)
parser.add_argument('--total_n', type=float, default=8000)
parser.add_argument('--window_size', type=int, default=100)
parser.add_argument('--hidden_units', type=int, default=10)
parser.add_argument('--signal_noise_ratio', type=int, default=0)
parser.add_argument('--train_ratio', type=float, default=0.5)
parser.add_argument('--valid_ratio', type=float, default=0.1)
# arguments to define paths
# parser.add_argument( '--experiment_path', type=Path, required=True)
parser.add_argument('-idp', '--input_data_path', type=Path, required=True)
# parser.add_argument('--log-board_path', type=Path, required=True)
parser.add_argument('-dp', '--data_path', type=Path, required=True)
parser.add_argument('-cp', '--config_path', type=Path, required=True)

parsed_args = parser.parse_args()


yaml_config_path = parsed_args.data_path / f'config.yml'


generate_config(yaml_config_path, parsed_args)

run_lstm(parsed_args)
# python oloop_training_custom_loop.py -dp ../models/euler/10000/pi-lstm-preloaded/ -cp ../lorenz_data/CSV/10000/euler_10000_norm.csv -idp /Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/models/euler/10000/model/10000/weights
