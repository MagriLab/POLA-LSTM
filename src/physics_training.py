import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import time
import random
import importlib
import datetime
import matplotlib.pyplot as plt
import torch
import os
import sys
import warnings
import argparse
from pathlib import Path
sys.path.append('../')
from lstm.loss import backward_diff, loss_oloop, pi_loss, bd_loss, norm_pi_loss
from lstm.lstm_model import build_pi_model, load_open_loop_lstm
from lstm.postprocessing import plots
from lstm.preprocessing.data_processing import (create_df_3d,
                                                df_train_valid_test_split,
                                                train_valid_test_split)
from lstm.utils.random_seed import reset_random_seeds
from lstm.utils.config import generate_config
from lstm.loss import lorenz

warnings.simplefilter(action="ignore", category=FutureWarning)


plt.rcParams["figure.facecolor"] = "w"


def run_lstm(args: argparse.Namespace):

    reset_random_seeds()

    filepath = args.data_path
    os.makedirs(filepath / "images")


    mydf = np.genfromtxt(args.config_path, delimiter=",").astype(np.float32)
    df_train, df_valid, df_test = df_train_valid_test_split(mydf[1:, :])
    time_train, time_valid, time_test = train_valid_test_split(mydf[0, :])

    # Windowing
    lorenz_dim = 3
    train_dataset = create_df_3d(df_train.transpose(), args.window_size, args.batch_size, df_train.shape[0])
    valid_dataset = create_df_3d(df_valid.transpose(), args.window_size, args.batch_size, 1)
    test_dataset = create_df_3d(df_test.transpose(), args.window_size, args.batch_size, 1)

    model = build_pi_model(args.n_cells)
    #model.load_weights(args.input_data_path)
    @tf.function
    def train_step(x_batch_train, y_batch_train):
        with tf.GradientTape() as tape:
            pred = model(x_batch_train, training=True)
            loss_dd = loss_oloop(y_batch_train, pred)
            loss_pi = pi_loss(pred, x_batch_train)
            loss_value = loss_dd + loss_pi
        grads = tape.gradient(loss_dd, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_dd, loss_pi

    @tf.function
    def train_step_pi(x_batch_train, y_batch_train, weight=1, normalised=True):
        with tf.GradientTape() as tape:
            pred = model(x_batch_train, training=True)
            loss_dd = loss_oloop(y_batch_train, pred)
            if normalised == True:
                loss_pi = norm_pi_loss(pred, x_batch_train)
            else:
                loss_pi = pi_loss(pred, x_batch_train)
            loss_value = loss_dd + weight*loss_pi
        grads = tape.gradient(loss_value, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_dd, loss_pi
    
    @tf.function
    def train_step_der(x_batch_train, y_batch_train, weight=1, normalised=True):
        with tf.GradientTape() as tape:
            pred = model(x_batch_train, training=True)
            loss_dd = loss_oloop(y_batch_train, pred)
            loss_pi = bd_loss(pred, x_batch_train, y_batch_train)
            loss_value = loss_dd + weight*loss_pi
        grads = tape.gradient(loss_value, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_dd, loss_pi

    @tf.function
    def valid_step_pi(x_batch_valid, y_batch_valid):
        val_logit = model(x_batch_valid, training=True)
        loss_dd = loss_oloop(y_batch_valid, val_logit)
        loss_pi = norm_pi_loss(val_logit, x_batch_valid)
        return loss_dd, loss_pi

    train_loss_dd_tracker = np.array([])
    train_loss_pi_tracker = np.array([])
    valid_loss_dd_tracker = np.array([])
    valid_loss_pi_tracker = np.array([])
    tf.keras.backend.set_value(model.optimizer.learning_rate, args.learning_rate)
    for epoch in range(args.n_epochs+1):
        start_time = time.time()
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_dd, loss_pi = train_step_pi(x_batch_train, y_batch_train, weight=args.physics_weighing, normalised=args.normalised)
        train_loss_dd_tracker = np.append(train_loss_dd_tracker, loss_dd)
        train_loss_pi_tracker = np.append(train_loss_pi_tracker, loss_pi)

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
        print("VALIDATION: Data-driven loss: %4E; Physics-informed loss at epoch: %.4E" % (valid_loss_dd/val_step, valid_loss_pi/val_step))

        if epoch % args.epoch_steps == 0:
            predictions = plots.plot_closed_loop_lya(
                model,
                epoch,
                time_test,
                df_test,
                n_length=1000,
                window_size=args.window_size,
                img_filepath=filepath / "images"/ f"pred_{epoch}.png",
            )
            plots.plot_phase_space(
                predictions,
                epoch,
                df_test,
                window_size=args.window_size,
                img_filepath=filepath/ "images"/ f"phase_{epoch}.png",
            )

            model_checkpoint = filepath / "model"/ f"{epoch}"/ "weights"
            model.save_weights(model_checkpoint)
            logs_checkpoint = filepath / "logs"
            if not os.path.exists(logs_checkpoint):
                os.makedirs(logs_checkpoint)
            np.savetxt(logs_checkpoint/f"training_loss_dd.txt", train_loss_dd_tracker)
            np.savetxt(logs_checkpoint/f"training_loss_pi.txt", train_loss_pi_tracker)
            np.savetxt(logs_checkpoint/f"valid_loss_dd.txt", valid_loss_dd_tracker)
            np.savetxt(logs_checkpoint/f"valid_loss_pi.txt", valid_loss_pi_tracker)


parser = argparse.ArgumentParser(description='Open Loop')
# arguments for configuration parameters
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--epoch_steps', type=int, default=100)
parser.add_argument('--epoch_iter', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_cells', type=int, default=10)
parser.add_argument('--oloop_train', default=True, action='store_true')
parser.add_argument('--cloop_train', default=False, action='store_true')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--activation', type=str, default='Tanh')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--l2_regularisation', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--early_stop', default=False, action='store_true')
parser.add_argument('--early_stop_patience', type=int, default=10)
parser.add_argument('--physics_informed', default=True, action='store_true')
parser.add_argument('--physics_weighing', type=float, default=1)

parser.add_argument('--normalised', default=True, action='store_true')
parser.add_argument('--t_0', type=int, default=0)
parser.add_argument('--t_trans', type=int, default=20)
parser.add_argument('--t_end', type=int, default=1000)
parser.add_argument('--delta_t', type=int, default=0.01)
parser.add_argument('--total_n', type=float, default=98000)
parser.add_argument('--window_size', type=int, default=100)
parser.add_argument('--hidden_units', type=int, default=10)

# arguments to define paths
# parser.add_argument( '--experiment_path', type=Path, required=True)
# parser.add_argument('-idp', '--input_data_path', type=Path, required=True)
# parser.add_argument('--log-board_path', type=Path, required=True)
parser.add_argument('-dp', '--data_path', type=Path, required=True)
parser.add_argument('-cp', '--config_path', type=Path, required=True)

parsed_args = parser.parse_args()


yaml_config_path = parsed_args.data_path / f'config.yml'


generate_config(yaml_config_path, parsed_args)

run_lstm(parsed_args)
# /Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/venv/bin/python /Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/physics_training.py -dp ./models/pi-model-100000/1503/ -cp lorenz_data/CSV/Lorenz_trans_001_norm_100000.csv 
# 
# -idp /Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/models/oloop100000/model/1000/weights
