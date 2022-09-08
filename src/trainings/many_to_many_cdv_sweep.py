
import argparse
import os
import sys
import time
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb
# wandb.login()

physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     # Disable first GPU
#     tf.config.set_visible_devices(physical_devices[0:], 'GPU')
#     logical_devices = tf.config.list_logical_devices('GPU')
#     print('Number of used GPUs: ', len(logical_devices))
#     # Logical device was not created for first GPU
#     assert len(logical_devices) == len(physical_devices) - 1
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass
tf.debugging.set_log_device_placement(True)
plt.rcParams["figure.facecolor"] = "w"

tf.keras.backend.set_floatx('float64')
sys.path.append('../..')

from wandb.keras import WandbCallback
from lstm.utils.random_seed import reset_random_seeds
from lstm.utils.config import generate_config
from lstm.preprocessing.data_processing import (create_df_nd_mtm,
                                                df_train_valid_test_split,
                                                train_valid_test_split)
from lstm.postprocessing.tensorboard_converter import loss_arr_to_tensorboard
from lstm.postprocessing import plots_mtm
from lstm.lstm_model import build_pi_model
from lstm.loss import loss_oloop
from lstm.cdv_equations import cdv_system_tensor

warnings.simplefilter(action="ignore", category=FutureWarning)
dim = 6

# x_fix, y_fix, z_fix = fixpoints(total_points=10000, unnorm=False)


def build_pi_model(cells=100):
    model = tf.keras.Sequential()
    kernel_init = tf.keras.initializers.GlorotUniform(seed=123)
    recurrent_init = tf.keras.initializers.Orthogonal(seed=123)
    model.add(tf.keras.layers.LSTM(cells, activation="tanh", name="LSTM_1", return_sequences=True))
    model.add(tf.keras.layers.Dense(dim, name="Dense_1"))
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, metrics=["mse"], loss=loss_oloop)
    return model


def run_lstm():
    reset_random_seeds()
    config_defaults = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "window_size": 100,
        "physics_weighing": 0.0,
        'hidden_units': 10
    }
    # Initialize wandb with a sample project name
    wand = wandb.init(config=config_defaults)

    parsed_args.learning_rate = wandb.config.learning_rate
    parsed_args.batch_size = wandb.config.batch_size
    parsed_args.window_size = wandb.config.window_size
    parsed_args.hidden_units = wandb.config.hidden_units
    parsed_args.physics_weighing = wandb.config.physics_weighing
    print("WANDB Name", wand.name)
    print('Learning rate: ', wandb.config.learning_rate, parsed_args.learning_rate)
    datapath = Path("../models/cdv/sweep_Test/")
    filepath = datapath / str(wand.name)

    if not os.path.exists(filepath / "images"):
        os.makedirs(filepath / "images")

    mydf = np.genfromtxt(parsed_args.config_path, delimiter=",").astype(np.float64)
    df_train, df_valid, df_test = df_train_valid_test_split(mydf[1:, :], train_ratio=wandb.config.train_ratio, valid_ratio=wandb.config.valid_ratio)
    time_train, time_valid, time_test = train_valid_test_split(mydf[0, :], train_ratio=wandb.config.train_ratio, valid_ratio=wandb.config.valid_ratio)
    # Windowing
    dim = 6
    train_dataset = create_df_nd_mtm(df_train.transpose(), parsed_args.window_size,
                                     parsed_args.batch_size, df_train.shape[0])
    valid_dataset = create_df_nd_mtm(df_valid.transpose(), parsed_args.window_size, parsed_args.batch_size, 1)
    test_dataset = create_df_nd_mtm(df_test.transpose(), parsed_args.window_size, parsed_args.batch_size, 1)

    model = build_pi_model(parsed_args.hidden_units)
    # model.load_weights(parsed_args.input_data_path)

    def decayed_learning_rate(step, initial_learning_rate=None):
        decay_steps = 1000
        decay_rate = 0.75
        if initial_learning_rate == None:
            initial_learning_rate == 0.001
        return initial_learning_rate * decay_rate ** (step / decay_steps)

    def norm_loss_pi_many(y_pred, washout=0, total_points=10000, norm=True):
        """_summary_

        parsed_args:
            y_pred (Tensor): network prediction
            x_batch_train: one batch of training windows
            washout (int, optional): to attenuate initialisation. Defaults to 0.
        Returns:
            _type_: _description_
        """
        # max_from_norm(total_points=total_points)
        delta_t = 0.1
        mse = tf.keras.losses.MeanSquaredError()
        u = cdv_system_tensor(y_pred)  # generate rhs of Lorenz equations
        # compute backward diff for all the predictions in a batch
        bd = (y_pred[:, 1:, :] - y_pred[:, :-1, :])/delta_t

        # print("RHS shape: ", u.shape, "Backward Diff shape:", bd.shape)
        return mse(u[:, :-1, :], bd)

    @tf.function
    def train_step_pi(x_batch_train, y_batch_train, weight=1, normalised=True):
        with tf.GradientTape() as tape:
            one_step_pred = model(x_batch_train, training=True)
            # new_batch = split_window_label(append_label_to_window(x_batch_train, one_step_pred))
            # two_step_pred = model(new_batch, training=True)
            loss_dd = loss_oloop(y_batch_train, one_step_pred)
            loss_pi = norm_loss_pi_many(one_step_pred, norm=normalised)
            loss_value = loss_dd + weight*loss_pi
        grads = tape.gradient(loss_value, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_dd, loss_pi

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
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=parsed_args.learning_rate, decay_steps=1000, decay_rate=0.5)
    # tf.keras.backend.set_value(model.optimizer.learning_rate, lr_schedule)

    for epoch in range(1, parsed_args.n_epochs+1):
        model.optimizer.learning_rate = decayed_learning_rate(epoch, initial_learning_rate=parsed_args.learning_rate)
        start_time = time.time()
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_dd, loss_pi = train_step_pi(x_batch_train, y_batch_train,
                                             weight=parsed_args.physics_weighing, normalised=parsed_args.normalised)
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
        print("VALIDATION: Data-driven loss: %4E; Physics-informed loss at epoch: %.4E" %
              (valid_loss_dd / val_step, valid_loss_pi / val_step))

        wandb.log({'epochs': epoch,
                   'train_dd_loss': float(loss_dd),
                   'train_physics_loss': float(loss_pi),
                   'valid_dd_loss': float(valid_loss_dd/val_step),
                   'valid_physics_loss': float(valid_loss_pi/val_step)})

        if epoch % parsed_args.epoch_steps == 0:
            print("LEARNING RATE:%.2e" % model.optimizer.learning_rate)

            predictions = plots_mtm.plot_cdv(
                model,
                epoch,
                time_test,
                df_test,
                c_lyapunov=0.033791,
                n_length=888,
                window_size=parsed_args.window_size,
                img_filepath=filepath / "images" / f"pred_{epoch}.png",
            )

            n_length = len(predictions)
            rel_l2_err = np.linalg.norm(df_test[:, parsed_args.window_size: parsed_args.window_size + n_length].T -
                                        predictions[: n_length]) / np.linalg.norm(predictions[: n_length])
            wandb.log({'epochs': epoch,
                       'dd train loss': float(loss_dd),
                       'dd physics loss': float(loss_pi),
                       'val_loss': float(valid_loss_dd/val_step),
                       'val_physics_loss': float(valid_loss_pi/val_step),
                       'rel_l2_err': float(rel_l2_err)}
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
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--epoch_steps', type=int, default=1000)
 
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_cells', type=int, default=10)
parser.add_argument('--oloop_train', default=True, action='store_true')
 
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--activation', type=str, default='Tanh')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--l2_regularisation', type=float, default=0)
parser.add_argument('--dropout', type=float, default=0.0)
 
parser.add_argument('--early_stop_patience', type=int, default=10)
 
parser.add_argument('--physics_weighing', type=float, default=0.0)
parser.add_argument('--normalised', default=False, action='store_true')
parser.add_argument('--t_0', type=int, default=0)
parser.add_argument('--t_trans', type=int, default=750)
parser.add_argument('--t_end', type=int, default=5750)
parser.add_argument('--delta_t', type=int, default=0.1)
parser.add_argument('--total_n', type=float, default=57500)
parser.add_argument('--window_size', type=int, default=50)
parser.add_argument('--hidden_units', type=int, default=10)
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


sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'valid_dd_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'batch_size': {
            'values': [128]
        },
        'learning_rate': {
            'values': [0.001]
        },
        'window_size': {
            'values': [100]
        },
        'hidden_units': {
            'values': [10]
        },
        'physics_weighing': {
            'values': [0, 1, 0.1, 0.01]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="CDV-27500-lambda")
wandb.agent(sweep_id, function=run_lstm, count=10)


print('Physics weight', parsed_args.physics_weighing)
yaml_config_path = parsed_args.data_path / f'config.yml'
generate_config(yaml_config_path, parsed_args)
# python many_to_many_cdv_sweep.py -dp ../models/cdv/27500-lambda-physics/ -cp ../cdv_data/CSV/euler_27500_trans.csv
# python many_to_many_cdv.py -dp ../models/cdv/test/ -cp ../cdv_data/CSV/euler_17500_trans.csv
