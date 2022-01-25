import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tensorflow_datasets as tfds
import tensorflow as tf


def create_open_loop_lstm():
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', name='LSTM_1'),
        tf.keras.layers.Dense(3, name='Dense_1')
        ])
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(loss='mse', optimizer='adam')
    return model

def load_open_loop_lstm(model_checkpoint):
    # Create a new model instance
    model = create_open_loop_lstm()
    # Restore the weights
    model.load_weights(model_checkpoint)
    return model
