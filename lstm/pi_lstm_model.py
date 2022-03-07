import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from idna import valid_contextj

from .loss import loss_oloop, pi_loss
lorenz_dim=3


class FeedBack(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        kernel_init = tf.keras.initializers.GlorotUniform(seed=1)
        recurrent_init = tf.keras.initializers.Orthogonal(seed=1)
        self._rnn = tf.keras.layers.LSTM(units, activation="relu", name="LSTM_1")
        self.dense = tf.keras.layers.Dense(lorenz_dim, name="Dense_1")
        
        
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

    # @property
    # def some_metrics(self):
    #     return [
    #         loss_tracker,
    #         mse_metric_tracker,
    #         val_loss_tracker,
    #         val_mse_metric_tracker,
    #     ]

    def model_compile(self, patience=10):
        optimizer = tf.keras.optimizers.Adam()
        mse_loss = tf.keras.losses.MeanSquaredError()
        self.loss = pi_loss + loss_oloop  # change to custom loss
        self.metric = tf.keras.metrics.MeanSquaredError()
        self.compile(loss=self.loss, optimizer=optimizer, metrics=["mse"])



    # def valid_step1(self, dataset):
    #     # Unpack the data
    #     for data, label in dataset:
    #         prediction, state = self.warmup(data)  
        
    #     val_loss_tracker.update_state(label, prediction)
    #     val_mse_metric_tracker.update_state(label, prediction)
    #     return {
    #         "val_loss": val_loss_tracker.result(),
    #     }
    
    @tf.function
    def train_step(self, x_batch_train, y_batch_train, der_y_batch_train):
        with tf.GradientTape() as tape:
            pred = self(x_batch_train, training=True)
            loss_dd = loss_oloop(y_batch_train, pred)
            loss_pi = pi_loss(pred, der_y_batch_train)
            loss_value = loss_dd + loss_pi
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss_dd, loss_pi


    def train_cloop(
        self, train_dataset, der_train_dataset, n_epochs, n_batches, val_dataset
    ):
        train_loss_results = []
        train_metrics_results = []
        for epoch in range(n_epochs):
            start_time = time.time()
            self.reset_states()
            epoch_loss_avg = self.loss
            epoch_metrics = self.metric
            for step, ((x_batch_train, y_batch_train),
                (der_x_batch_train, der_y_batch_train)) in enumerate(zip(train_dataset, der_train_dataset)):
                loss_dd, loss_pi = self.train_step(x_batch_train, y_batch_train, der_y_batch_train)
            loss_mse_dict = {
                "loss": loss_dd + loss_pi,
                "dd-loss": loss_dd.numpy(),
                "pi-loss": loss_pi.numpy(),
            }
            val_dict = self.valid_step1(val_dataset)

            train_loss_results.append(loss_mse_dict["loss"])
            if epoch % 2 == 0:
                print(
                    "Epoch {:.1f}, Loss: {:.2E},  Data-driven Loss: {:.2E}, Pysics Loss: {:.2E}, Val Loss: {:.2E}".format(
                        epoch,
                        loss_mse_dict["loss"],
                        loss_mse_dict["dd-loss"],
                        loss_mse_dict["pi-loss"],
                        0 #val_dict["val_loss"],
                    )
                )
                print("Time of this epoch: ", time.time() - start_time)
