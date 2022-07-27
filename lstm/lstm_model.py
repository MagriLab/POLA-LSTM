import argparse
import random
import time
from pathlib import Path
from pickletools import optimize
from re import L

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from .loss import loss_oloop, loss_oloop_l2_reg, norm_pi_loss
from .postprocessing import plots

lorenz_dim = 3


def build_open_loop_lstm(cells=100):
    model = tf.keras.Sequential()
    kernel_init = tf.keras.initializers.GlorotUniform(seed=1)
    recurrent_init = tf.keras.initializers.Orthogonal(seed=1)
    model.add(tf.keras.layers.LSTM(cells, activation="tanh", name="LSTM_1"))
    model.add(tf.keras.layers.Dense(lorenz_dim, name="Dense_1"))
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001, decay_steps=32000, decay_rate=0.75)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(loss=loss_oloop, optimizer=optimizer, metrics=["mse"])
    return model


def build_pi_model(cells=100, dim=3):
    model = tf.keras.Sequential()
    kernel_init = tf.keras.initializers.GlorotUniform(seed=123)
    recurrent_init = tf.keras.initializers.Orthogonal(seed=123)
    model.add(tf.keras.layers.LSTM(cells, activation="tanh", name="LSTM_1", return_sequences=True,
              kernel_initializer=kernel_init, recurrent_initializer=recurrent_init))
    model.add(tf.keras.layers.Dense(dim, name="Dense_1"))
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, metrics=["mse"], loss=loss_oloop)
    return model


def load_model(model_path, epochs, model_dict, dim=3):
    model = build_pi_model(model_dict['ML_CONSTRAINTS']['N_CELLS'], dim=dim)
    model.load_weights(model_path + "model/" + str(epochs) + "/weights").expect_partial()
    return model

class LorenzLSTM(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, log_board_path: Path):
        super().__init__()
        self.n_cells = args.n_cells
        self.loss = loss_oloop
        self.lstm_cell = tf.keras.layers.LSTMCell(args.n_cells)
        self.lstm_rnn = tf.keras.layers.RNN(
            self.lstm_cell, return_state=True, stateful=False, name="LSTM1"
        )
        self.dense = tf.keras.layers.Dense(3)
        self.optimizer = tf.keras.optimizers.Adam()
        self.compile(loss=self.loss, optimizer=self.optimizer, metrics=["mse"])
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_board_path, histogram_freq=1)
        if args.early_stop == True:
            self.early_stop_callback = tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=args.early_stop_patience, restore_best_weights=True)

    def save_model(self, model_path: Path):
        model_checkpoint = model_path / str(self.current_epoch)
        # Save the weights
        self.save_weights(model_checkpoint)

    def train_oloop(self, args: argparse.Namespace, train_dataset, valid_dataset):
        for iteration in range(args.epoch_iter):
            if args.early_stop == False:
                self.history = self.fit(
                    train_dataset,
                    epochs=args.epochs_steps,
                    inital_epoch=iteration*args.epochs_steps,
                    batch_size=args.batch_size,
                    validation_data=valid_dataset,
                    verbose=1,
                    callbacks=[self.tensorboard_callback],  # , early_stop_callback],
                )
            else:
                self.history = self.fit(
                    train_dataset,
                    epochs=args.epochs_steps,
                    inital_epoch=iteration*args.epochs_steps,
                    batch_size=args.batch_size,
                    validation_data=valid_dataset,
                    verbose=1,
                    callbacks=[self.tensorboard_callback, self.early_stop_callback],
                )
            self.current_epoch = (iteration+1)*args.epochs_steps
