import random
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path

from .loss import loss_oloop, loss_oloop_reg, pi_loss
from .postprocessing import plots

lorenz_dim = 3


def build_open_loop_lstm(cells=100):
    model = tf.keras.Sequential()
    kernel_init = tf.keras.initializers.GlorotUniform(seed=1)
    recurrent_init = tf.keras.initializers.Orthogonal(seed=1)
    model.add(tf.keras.layers.LSTM(cells, activation="relu", name="LSTM_1"))
    model.add(tf.keras.layers.Dense(lorenz_dim, name="Dense_1"))
    model.compile(loss=loss_oloop, optimizer="adam", metrics=["mse"])
    return model


def build_pi_model(cells=100):
    model = tf.keras.Sequential()
    kernel_init = tf.keras.initializers.GlorotUniform(seed=123)
    recurrent_init = tf.keras.initializers.Orthogonal(seed=123)
    model.add(tf.keras.layers.LSTM(cells, activation="relu", name="LSTM_1"))
    model.add(tf.keras.layers.Dense(lorenz_dim, name="Dense_1"))
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, metrics=["mse"])
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


def train_oloop(model, epochs, train_dataset, batch_size):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(
                    x_batch_train, training=True
                )  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = loss_oloop(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            model.optimizer.apply_gradients(
                zip(grads, model.trainable_weights))

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))


def load_open_loop_lstm(model_checkpoint):
    # Create a new model instance
    model = LorenzLSTM()
    # Restore the weights
    model.load_weights(model_checkpoint)
    return model
