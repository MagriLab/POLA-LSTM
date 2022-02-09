import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tensorflow_datasets as tfds
import tensorflow as tf

loss_tracker = tf.keras.metrics.MeanSquaredError(name="loss")
mse_metric_tracker = tf.keras.metrics.MeanSquaredError(name="mse")
val_loss_tracker = tf.keras.metrics.MeanSquaredError(name="val_loss")
val_mse_metric_tracker = tf.keras.metrics.MeanSquaredError(name="val_mse")


def custom_loss(y_true, y_pred, washout=0):
    mse = tf.keras.losses.MeanSquaredError()
    loss = mse(y_true[washout:, :], y_pred[washout:, :])  # (batchsize, dimensions)
    return loss


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

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)
        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, out_steps=1, training=None):
        self.out_steps = out_steps
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)
        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state, training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)
            # predictions.shape => (time, batch, features)
            predictions = tf.stack(predictions)
            # predictions.shape => (batch, time, features)
            predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions  # np.array(predictions)

    def model_compile(self, patience=10):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience
        )
        optimizer = tf.keras.optimizers.Adam()
        mse_loss = tf.keras.losses.MeanSquaredError()
        self.loss = mse_loss
        self.metric = tf.keras.metrics.MeanSquaredError()
        self.compile(loss=self.loss, optimizer=optimizer, metrics=["mse"])

    def train_step1(self, data, label):
        # dataset contains ((batch, window, feature), (batch, feature))
        # data, label = input
        # for data, label in dataset:

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
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
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

    def train_epochs(self, dataset, n_epochs, val_dataset):
        train_loss_results = []
        train_metrics_results = []
        for epoch in range(n_epochs):
            self.reset_states()
            epoch_loss_avg = self.loss
            epoch_metrics = self.metric
            for data, label in dataset:
                loss_mse_dict = self.train_step1(data, label)

            val_dict = val_loss_dict = self.valid_step1(val_dataset)
            # epoch_loss_avg.update_state(loss_mse_dict['loss'])
            # epoch_metrics.update_state(y, model(x, training=True))
            # End epoch
            train_loss_results.append(loss_mse_dict["loss"])
            train_metrics_results.append(loss_mse_dict["mse"])
            if epoch % 2 == 0:
                print(
                    "Epoch {:03d}: Loss: {:.3f}, Metrics: {:.3f}, Val_Loss: {:.3f}, Val_Metrics: {:.3f}".format(
                        epoch,
                        train_loss_results[-1],
                        train_metrics_results[-1],
                        val_dict["val_loss"],
                        val_dict["val_mse"],
                    )
                )
