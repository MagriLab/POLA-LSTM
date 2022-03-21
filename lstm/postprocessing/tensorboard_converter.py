import tensorflow as tf

import numpy as np

import os
import time


def loss_arr_to_tensorboard(
        logs_checkpoint, train_loss_dd_tracker, train_loss_pi_tracker, valid_loss_pi_tracker, valid_loss_dd_tracker):
    now = time.localtime()
    subdir = time.strftime("%d-%b-%Y_%H.%M.%S", now)
    summary_dir1 = os.path.join(logs_checkpoint, "train")
    summary_writer1 = tf.summary.create_file_writer(summary_dir1)
    for cont in range(0, len(train_loss_dd_tracker)):
        with summary_writer1.as_default():
            tf.summary.scalar(name="epoch:physics_informed_loss", data=train_loss_pi_tracker[cont], step=cont)
            tf.summary.scalar(name="epoch:data_driven_loss", data=train_loss_dd_tracker[cont], step=cont)
        summary_writer1.flush()

    summary_dir1 = os.path.join(logs_checkpoint, "valid")
    summary_writer1 = tf.summary.create_file_writer(summary_dir1)
    for cont in range(0, len(train_loss_dd_tracker)):
        with summary_writer1.as_default():
            tf.summary.scalar(name="epoch:physics_informed_loss", data=valid_loss_pi_tracker[cont], step=cont)
            tf.summary.scalar(name="epoch:data_driven_loss", data=valid_loss_dd_tracker[cont], step=cont)
    summary_writer1.flush()
