import sys
sys.path.append('..')
from lstm.utils.config import generate_config
from lstm.preprocessing.data_processing import (create_df_3d,
                                                df_train_valid_test_split,
                                                train_valid_test_split)
from lstm.postprocessing import plots
from lstm.lstm_model import build_open_loop_lstm
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import time
import random
import os
import importlib
import datetime
import argparse

plt.rcParams["figure.facecolor"] = "w"
warnings.simplefilter(action="ignore", category=FutureWarning)


def run_lstm(args: argparse.Namespace):

    # first: get the lorenz data ready
    lorenz_df = np.genfromtxt(args.config_path, delimiter=",")
    time_train, time_valid, time_test = train_valid_test_split(lorenz_df[0, :])
    df_train, df_valid, df_test = df_train_valid_test_split(lorenz_df[1:, :])
    print(df_train.shape)
    train_dataset = create_df_3d(
        df_train.transpose(), args.window_size, args.batch_size,  df_train.shape[1]
    )
    valid_dataset = create_df_3d(df_valid.transpose(), args.window_size, args.batch_size, 1)
    test_dataset = create_df_3d(df_test.transpose(),  args.window_size, args.batch_size, 1)

    # Building the model
    model = build_open_loop_lstm(args.n_cells)
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=10, restore_best_weights=True
    )
    log_dir = args.data_path / 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    current_epoch = 0
    print("--- Begin Open Loop Training ---")
    for i in range(0, args.epoch_iter):
        n_epochs_old = current_epoch
        current_epoch = current_epoch + args.epoch_steps
        history = model.fit(
            train_dataset,
            epochs=current_epoch,
            initial_epoch=n_epochs_old,
            batch_size=args.batch_size,
            validation_data=valid_dataset,
            verbose=1,
            callbacks=[tensorboard_callback],  # , early_stop_callback],
        )

        model_checkpoint = args.data_path / "model" / str(current_epoch) / "weights"

        model.save_weights(model_checkpoint)
        lya_filepath = args.data_path / "images" / f"{current_epoch}_oloop.png"
        lya_filepath.parent.mkdir(parents=True, exist_ok=True)

        predictions = plots.plot_closed_loop_lya(
            model,
            history.params["epochs"],
            time_test,
            df_test,
            n_length=500,
            window_size=args.window_size,
            img_filepath=lya_filepath,
        )
        phase_filepath = args.data_path / "images" / f"{current_epoch}_phase.png"
        plots.plot_phase_space(
            predictions,
            history.params["epochs"],
            df_test,
            img_filepath=phase_filepath,
            window_size=args.window_size,
        )


parser = argparse.ArgumentParser(description='Open Loop')
# arguments for configuration parameters
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--epoch_steps', type=int, default=5)
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

parser.add_argument('--normalised', default=True, action='store_true')
parser.add_argument('--t_0', type=int, default=0)
parser.add_argument('--t_trans', type=int, default=20)
parser.add_argument('--t_end', type=int, default=1000)
parser.add_argument('--delta_t', type=int, default=0.01)
parser.add_argument('--total_n', type=float, default=98000)
parser.add_argument('--window_size', type=int, default=100)
parser.add_argument('--hidden_units', type=int, default=100)

# arguments to define paths
# parser.add_argument( '--experiment_path', type=Path, required=True)
# parser.add_argument('--input-data_path', type=Path, required=True)
# parser.add_argument('--log-board_path', type=Path, required=True)
parser.add_argument('-dp', '--data_path', type=Path, required=True)
parser.add_argument('-cp', '--config_path', type=Path, required=True)

parsed_args = parser.parse_args()


yaml_config_path = parsed_args.data_path / f'config.yml'


generate_config(yaml_config_path, parsed_args)

run_lstm(parsed_args)
# /Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/venv/bin/python /Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/oloop_training.py -dp ./here/ -cp lorenz_data/CSV/Lorenz_trans_001_norm_100000.csv
