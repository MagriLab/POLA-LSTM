import argparse
import os
import sys
import time
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.append('../..')

from lstm.utils.random_seed import reset_random_seeds
from lstm.utils.config import generate_config
from lstm.preprocessing.data_processing import (create_df_3d_mtm,
                                                df_train_valid_test_split,
                                                train_valid_test_split)
from lstm.postprocessing.tensorboard_converter import loss_arr_to_tensorboard
from lstm.postprocessing import plots_mtm
from lstm.lstm_model import build_pi_model
from lstm.lorenz import fixpoints
from lstm.loss import loss_oloop, norm_loss_pi_many
from lstm.closed_loop_tools_mto import append_label_to_batch
from lstm.closed_loop_tools_mtm import split_window_label

plt.rcParams["figure.facecolor"] = "w"

tf.keras.backend.set_floatx('float64')

warnings.simplefilter(action="ignore", category=FutureWarning)
lorenz_dim = 3

x_fix, y_fix, z_fix = fixpoints(total_points=10000, unnorm=False)


def build_pi_model(cells=100):
    model = tf.keras.Sequential()
    kernel_init = tf.keras.initializers.GlorotUniform(seed=123)
    recurrent_init = tf.keras.initializers.Orthogonal(seed=123)
    model.add(tf.keras.layers.LSTM(cells, activation="tanh", name="LSTM_1", return_sequences=True))
    model.add(tf.keras.layers.Dense(lorenz_dim, name="Dense_1"))
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, metrics=["mse"], loss=loss_oloop)
    return model


def run_lstm(args: argparse.Namespace):

    reset_random_seeds()

    filepath = args.data_path
    if not os.path.exists(filepath / "images"):
        os.makedirs(filepath / "images")

    mydf = np.genfromtxt(args.config_path, delimiter=",").astype(np.float64)
    df_train, df_valid, df_test = df_train_valid_test_split(mydf[1:, :], train_ratio=0.3334, valid_ratio=0.3334)
    time_train, time_valid, time_test = train_valid_test_split(mydf[0, :], train_ratio=0.3334, valid_ratio=0.3334)

    # Windowing
    train_dataset = create_df_3d_mtm(df_train.transpose(), args.window_size, args.batch_size, df_train.shape[0])
    valid_dataset = create_df_3d_mtm(df_valid.transpose(), args.window_size, args.batch_size, 1)

    model = build_pi_model(args.n_cells)
    model.load_weights(args.input_data_path)

    def decayed_learning_rate(step):
        initial_learning_rate = args.learning_rate
        decay_steps = 1000
        decay_rate = 0.75
        # careful here! step includes batch steps in the tf framework
        return initial_learning_rate * decay_rate ** (step / decay_steps)

    @tf.function
    def train_step_pi_cloop(x_batch_train, y_batch_train, n_cloop, weight=1, normalised=True):
        with tf.GradientTape() as tape:
            pred = model(x_batch_train, training=True)
            loss_dd = loss_oloop(y_batch_train, pred)
            loss_pi = norm_loss_pi_many(pred, norm=normalised)
            loss_pi_cloop = 0
            for i in range(n_cloop):
                new_batch = split_window_label(append_label_to_batch(x_batch_train, pred[:, -1, :]))
                pred = model(new_batch, training=True)
            loss_pi_cloop += norm_loss_pi_many(pred[:, -(n_cloop+1):, :], norm=normalised)
            loss_value = loss_pi_cloop + loss_dd
        grads = tape.gradient(loss_value, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_dd, loss_pi, loss_pi_cloop

    @tf.function
    def valid_step_pi(x_batch_valid, y_batch_valid, normalised=True):
        val_logit = model(x_batch_valid, training=False)
        loss_dd = loss_oloop(y_batch_valid, val_logit)
        # new_batch = split_window_label(append_label_to_window(x_batch_valid, val_logit))
        # two_step_pred = model(new_batch, training=False)
        loss_pi = norm_loss_pi_many(val_logit, norm=normalised)
        return loss_dd, loss_pi

    train_loss_dd_tracker = np.array([])
    train_loss_pi_tracker = np.array([])
    valid_loss_dd_tracker = np.array([])
    valid_loss_pi_tracker = np.array([])
    train_loss_pi_cloop_tracker = np.array([])
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=args.learning_rate, decay_steps=1000, decay_rate=0.5)
    # tf.keras.backend.set_value(model.optimizer.learning_rate, lr_schedule)

    for epoch in range(args.n_epochs+1):
        model.optimizer.learning_rate = decayed_learning_rate(epoch)
        start_time = time.time()
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_dd, loss_pi, loss_pi_cloop = train_step_pi_cloop(
                x_batch_train, y_batch_train, cloop_steps, weight=args.physics_weighing, normalised=args.normalised)
        train_loss_dd_tracker = np.append(train_loss_dd_tracker, loss_dd)
        train_loss_pi_tracker = np.append(train_loss_pi_tracker, loss_pi)
        train_loss_pi_cloop_tracker = np.append(train_loss_pi_cloop_tracker, loss_pi_cloop)

        print("Epoch: %d, Time: %.1fs , Batch: %d" % (epoch, time.time() - start_time, step))
        print(
            "TRAINING: Data-driven loss: %4E; Physics-informed loss at epoch: %.4E; Closed-loop loss at epoch: %.4E" %
            (loss_dd, loss_pi, loss_pi_cloop))

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
            predictions = plots_mtm.plot_prediction(
                model,
                epoch,
                time_test,
                df_test,
                n_length=500,
                window_size=args.window_size,
                img_filepath=filepath / "images" / f"pred_{epoch}.png",
            )
            plots_mtm.plot_phase_space(
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
    np.savetxt(logs_checkpoint/f"training_loss_pi_cloop.txt", train_loss_pi_cloop_tracker)

    loss_arr_to_tensorboard(logs_checkpoint, train_loss_dd_tracker, train_loss_pi_tracker,
                            valid_loss_dd_tracker, valid_loss_pi_tracker)


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
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--l2_regularisation', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--early_stop', default=False, action='store_true')
parser.add_argument('--early_stop_patience', type=int, default=10)
parser.add_argument('--physics_informed', default=True, action='store_true')
parser.add_argument('--physics_weighing', type=float, default=0)

parser.add_argument('--normalised', default=True, action='store_true')
parser.add_argument('--t_0', type=int, default=0)
parser.add_argument('--t_trans', type=int, default=20)
parser.add_argument('--t_end', type=int, default=100)
parser.add_argument('--delta_t', type=int, default=0.01)
parser.add_argument('--total_n', type=float, default=8000)
parser.add_argument('--window_size', type=int, default=100)
parser.add_argument('--hidden_units', type=int, default=10)
parser.add_argument('--signal_noise_ratio', type=int, default=0)
# arguments to define paths
# parser.add_argument( '--experiment_path', type=Path, required=True)
parser.add_argument('-idp', '--input_data_path', type=Path, required=True)
# parser.add_argument('--log-board_path', type=Path, required=True)
parser.add_argument('-dp', '--data_path', type=Path, required=True)
parser.add_argument('-cp', '--config_path', type=Path, required=True)

parsed_args = parser.parse_args()


yaml_config_path = parsed_args.data_path / f'config.yml'


generate_config(yaml_config_path, parsed_args)
cloop_steps = 10
print(cloop_steps)
run_lstm(parsed_args)
# python many_to_many_cloop.py -dp ../models/euler/10000-many-diff_loss/cloop-pilstm-preloaded_10_01/ -cp ../lorenz_data/CSV/10000/euler_10000_norm_trans.csv -idp /Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/models/euler/10000-many-diff_loss/model/10000/weights
