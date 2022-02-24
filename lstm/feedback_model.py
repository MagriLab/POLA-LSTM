import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from idna import valid_contextj

from loss import loss_oloop

loss_tracker = tf.keras.metrics.MeanSquaredError(name="loss")
mse_metric_tracker = tf.keras.metrics.MeanSquaredError(name="mse")
val_loss_tracker = tf.keras.metrics.MeanSquaredError(name="val_loss")
val_mse_metric_tracker = tf.keras.metrics.MeanSquaredError(name="val_mse")
cloop_loss_tracker = tf.keras.metrics.MeanSquaredError(name="cloop_loss")


class FeedBack(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(
            self.lstm_cell, return_state=True, stateful=False, name="LSTM1"
        )
        self.dense = tf.keras.layers.Dense(3)
        self.model_compile()

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)
        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs):
        predictions = []
        prediction, state = self.warmup(inputs)
        predictions.append(prediction)
        return predictions

    def model_compile(self, patience=10):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience
        )
        optimizer = tf.keras.optimizers.Adam()
        mse_loss = tf.keras.losses.MeanSquaredError()
        self.loss = mse_loss  # change to custom loss
        self.metric = tf.keras.metrics.MeanSquaredError()
        self.compile(loss=self.loss, optimizer=optimizer, metrics=["mse"])

    @tf.function
    def train_step_oloop(self, data, label):
        # dataset contains ((batch, window, feature), (batch, feature))
        with tf.GradientTape() as tape:
            prediction, state = self.warmup(data)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(label, prediction)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Compute our own metrics
        loss_tracker.update_state(label, prediction)
        mse_metric_tracker.update_state(label, prediction)
        return {"loss": loss_tracker.result(), "mse": mse_metric_tracker.result()}

    @property
    def some_metrics(self):
        return [
            loss_tracker,
            mse_metric_tracker,
            val_loss_tracker,
            val_mse_metric_tracker,
        ]

    def valid_step1(self, dataset):
        # Unpack the data
        for data, label in dataset:
            prediction, state = self.warmup(data)  # Forward pass
        # Update the metrics.
        val_loss_tracker.update_state(label, prediction)
        val_mse_metric_tracker.update_state(label, prediction)
        return {
            "val_loss": val_loss_tracker.result(),
            "val_mse": val_mse_metric_tracker.result(),
        }

    @tf.function
    def train_cloop_batch(self, data_oloop, label_oloop, data_cloop, label_cloop):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.trainable_variables)
            prediction_oloop, state = self.warmup(data_oloop)  # Forward pass open loop
            loss_oloop = self.compiled_loss(label_oloop, prediction_oloop)
            prediction_cloop, state = self.warmup(
                data_cloop
            )  # Forward pass closed loop
            loss_cloop = self.compiled_loss(label_cloop, prediction_cloop)

            loss = loss_cloop + loss_oloop
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return prediction_oloop, prediction_cloop

    def train_cloop(
        self, oloop_dataset, cloop_dataset, n_epochs, n_batches, val_dataset
    ):
        train_loss_results = []
        train_metrics_results = []
        for epoch in range(n_epochs):
            start_time = time.time()

            self.reset_states()
            epoch_loss_avg = self.loss
            epoch_metrics = self.metric
            for batch in range(n_batches):
                data_oloop, label_oloop = np.array(list(oloop_dataset), dtype=object)[
                    batch, :
                ]
                data_cloop, label_cloop = list(cloop_dataset)[batch]
                prediction_oloop, prediction_cloop = self.train_cloop_batch(
                    data_oloop, label_oloop, data_cloop, label_cloop
                )

            loss_tracker.update_state(label_oloop, prediction_oloop)
            mse_metric_tracker.update_state(label_oloop, prediction_oloop)
            cloop_loss_tracker.update_state(label_cloop, prediction_cloop)

            loss_mse_dict = {
                "loss": loss_tracker.result(),
                "mse": mse_metric_tracker.result(),
                "cloop_loss": cloop_loss_tracker.result(),
            }
            val_dict = self.valid_step1(val_dataset)

            train_loss_results.append(loss_mse_dict["loss"])
            train_metrics_results.append(loss_mse_dict["mse"])
            if epoch % 2 == 0:
                print(
                    "Epoch {:.1f}, Loss: {:.2E},  Cloop Loss: {:.2E}, Metrics: {:.2E}, Val_Loss: {:.2E}, Val_Metrics: {:.2E}".format(
                        epoch,
                        train_loss_results[-1],
                        train_metrics_results[-1],
                        loss_mse_dict["cloop_loss"],
                        val_dict["val_loss"],
                        val_dict["val_mse"],
                    )
                )
                print("Time of this epoch: ", time.time() - start_time)
