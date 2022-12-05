
import argparse
import os
import sys
import time
import warnings
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb
# wandb.login()
from wandb.keras import WandbCallback
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
# tf.debugging.set_log_device_placement(True)
sys.path.append('../..')

from lstm.preprocessing.data_processing import (create_df_nd_mtm,
                                                df_train_valid_test_split,
                                                train_valid_test_split)
from lstm.utils.random_seed import reset_random_seeds
from lstm.utils.config import generate_config
from lstm.postprocessing.loss_saver import loss_arr_to_tensorboard, save_and_update_loss_txt
from lstm.closed_loop_tools_mtm import prediction
from lstm.postprocessing.nrmse import vpt
from lstm.lstm_model import build_pi_model
physical_devices = tf.config.list_physical_devices('GPU')

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

def plot_pred_save(pred, df_valid, filepath, epoch):
    fig, ax = plt.subplots()
    N_plot = min(pred.shape[0], df_valid.shape[1])
    ax.plot(pred[:N_plot, :])
    ax.plot(df_valid[:, :N_plot].T, 'k')
    img_filepath=filepath / "images" / f"pred_{epoch}.png",
    fig.savefig(img_filepath, dpi=100, facecolor="w", bbox_inches="tight")
    # plt.close()

def main():
    def run_lstm():
        reset_random_seeds()
        config_defaults = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "window_size": 100,
            "upsampling": 1,
            'n_cells': 10
        }
        # Initialize wandb with a sample project name
        wand = wandb.init(config=config_defaults)

        args.learning_rate = wandb.config.learning_rate
        args.batch_size = wandb.config.batch_size
        args.window_size = wandb.config.window_size
        args.n_cells = wandb.config.n_cells
        args.upsampling = wandb.config.upsampling
        print("WANDB Name", wand.name)
        print('Learning rate: ', wandb.config.learning_rate, args.learning_rate)
        datapath = Path("../models/cdv/sweep_Test/")
        filepath = datapath / str(wand.name)

        if not os.path.exists(filepath / "images"):
            os.makedirs(filepath / "images")

        reset_random_seeds()
        image_filepath = args.data_path / "images"
        image_filepath.mkdir(parents=True, exist_ok=True)
        logs_checkpoint = args.data_path / "logs"
        logs_checkpoint.mkdir(parents=True, exist_ok=True)

        mydf = np.genfromtxt(args.config_path, delimiter=",").astype(np.float64)
        # mydf[1:,:] = mydf[1:,:]/(np.max(mydf[1:,:]) - np.min(mydf[1:,:]) )
        df_train, df_valid, df_test = df_train_valid_test_split(mydf[1:, ::args.upsampling], train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)
        time_train, time_valid, time_test = train_valid_test_split(mydf[0, ::args.upsampling], train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)
        sys_dim = df_train.shape[0]
        print(f'Dimension of system {sys_dim}')
        t_lyap = args.lyap**(-1)
        norm_time = 1
        N_lyap = int(t_lyap/(args.delta_t*args.upsampling))
        # Windowing
        train_dataset = create_df_nd_random_md_mtm(df_train.transpose(), args.window_size, args.batch_size, df_train.shape[0], n_random_idx=args.n_random_idx)
        valid_dataset = create_df_nd_random_md_mtm(df_valid.transpose(), args.window_size, args.batch_size, 1, n_random_idx=args.n_random_idx)
        for batch, label in train_dataset.take(1):
            print(f'Shape of batch: {batch.shape} \n Shape of Label {label.shape}')
        model = build_pi_model(args.n_cells, dim=sys_dim)

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
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=args.learning_rate, decay_steps=1000, decay_rate=0.5)
        # tf.keras.backend.set_value(model.optimizer.learning_rate, lr_schedule)

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


            wandb.log({'epochs': epoch,
                    'train_dd_loss': float(train_loss_dd/step),
                    'train_physics_loss': float(train_loss_pi/step),
                    'valid_dd_loss': float(valid_loss_dd/val_step),
                    'valid_physics_loss': float(valid_loss_pi/val_step)})

            if epoch % args.epoch_steps == 0:
                print("LEARNING RATE:%.2e" % model.optimizer.learning_rate)
                model_checkpoint = args.data_path / "model" / f"{epoch}" / "weights"
                model.save_weights(model_checkpoint)
                logs_epoch_checkpoint = logs_checkpoint / f"{epoch}"
                loss_arr_to_tensorboard(logs_epoch_checkpoint, train_loss_dd_tracker, train_loss_pi_tracker,
                                        valid_loss_dd_tracker, valid_loss_pi_tracker)
                save_and_update_loss_txt(
                    logs_checkpoint, 
                    train_loss_dd_tracker[-args.epoch_steps:],
                    train_loss_pi_tracker[-args.epoch_steps:],
                    valid_loss_dd_tracker[-args.epoch_steps:],
                    valid_loss_pi_tracker[-args.epoch_steps:])
                N=10*N_lyap
                pred = prediction(model, df_valid, args.window_size, sys_dim, args.n_random_idx, N=N)
                # plot_pred_save(pred, df_valid, filepath, epoch)
                lyapunov_time = np.arange(0, N/N_lyap, args.delta_t/t_lyap)
                pred_horizon = lyapunov_time[vpt(pred[args.window_size:], df_valid[:, args.window_size:], 0.4)]
                wandb.log({'epochs': epoch,
                            'pred_horizon': float(pred_horizon),
                            'train_dd_loss': float(train_loss_dd/step),
                            'train_physics_loss': float(train_loss_pi/step),
                            'valid_dd_loss': float(valid_loss_dd/val_step),
                            'valid_physics_loss': float(valid_loss_pi/val_step)})

    parser = argparse.ArgumentParser(description='Open Loop')

    parser.add_argument('--n_epochs', type=int, default=3000)
    parser.add_argument('--epoch_steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_cells', type=int, default=50)
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
    parser.add_argument('--t_trans', type=int, default=100)
    parser.add_argument('--t_end', type=int, default=425)
    parser.add_argument('--upsampling', type=int, default=1)
    parser.add_argument('--n_random_idx', type=int, default=4)
    parser.add_argument('--lyap', type=float, default=1.0)
    parser.add_argument('--delta_t', type=float, default=0.01)
    parser.add_argument('--total_n', type=float, default=42500)
    parser.add_argument('--window_size', type=int, default=25)
    parser.add_argument('--signal_noise_ratio', type=int, default=0)
    parser.add_argument('--train_ratio', type=float, default=0.45)
    parser.add_argument('--valid_ratio', type=float, default=0.05)

    # arguments to define paths
    # parser.add_argument( '--experiment_path', type=Path, required=True)
    # parser.add_argument('-idp', '--input_data_path', type=Path, required=True)
    # parser.add_argument('--log-board_path', type=Path, required=True)
    parser.add_argument('-dp', '--data_path', type=Path, required=True)
    parser.add_argument('-cp', '--config_path', type=Path, required=True)


    args = parser.parse_args()


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
                'values': [50]
            },
            'hidden_units': {
                'values': [100]
            },
            'upsampling': {
                'values': [1, 2, 3, 4, 5]
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="L96-Sweep")
    wandb.agent(sweep_id, function=run_lstm, count=20)


    print('Physics weight', args.physics_weighing)
    yaml_config_path = args.data_path / f'config.yml'
    generate_config(yaml_config_path, args)

if __name__ == '__main__':
    main()
    
# python many_to_many_sweep.py  -cp /Yael_CSV/L96/dim_6_rk4_42500_0.01_stand13.33_trans.csv -dp ../models/D6/sweep/