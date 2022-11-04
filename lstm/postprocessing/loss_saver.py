import os
import time

import numpy as np
import tensorflow as tf


def loss_arr_to_tensorboard(
        logs_checkpoint, train_loss_dd_tracker, train_loss_pi_tracker, valid_loss_dd_tracker, valid_loss_pi_tracker):
    now = time.localtime()
    summary_dir1 = os.path.join(logs_checkpoint, "train")
    summary_writer1 = tf.summary.create_file_writer(summary_dir1)
    for cont in range(0, len(train_loss_dd_tracker)):
        with summary_writer1.as_default():
            tf.summary.scalar(name="epoch:physics_informed_loss", data=train_loss_pi_tracker[cont], step=cont)
            tf.summary.scalar(name="epoch:data_driven_loss", data=train_loss_dd_tracker[cont], step=cont)
            tf.summary.scalar(name="epoch:total_loss",
                              data=train_loss_dd_tracker[cont]+train_loss_pi_tracker[cont], step=cont)
        summary_writer1.flush()

    summary_dir1 = os.path.join(logs_checkpoint, "valid")
    summary_writer1 = tf.summary.create_file_writer(summary_dir1)
    for cont in range(0, len(train_loss_dd_tracker)):
        with summary_writer1.as_default():
            tf.summary.scalar(name="epoch:physics_informed_loss", data=valid_loss_pi_tracker[cont], step=cont)
            tf.summary.scalar(name="epoch:data_driven_loss", data=valid_loss_dd_tracker[cont], step=cont)
            tf.summary.scalar(name="epoch:total_loss",
                              data=valid_loss_dd_tracker[cont]+valid_loss_pi_tracker[cont], step=cont)
    summary_writer1.flush()


# from dataclasses import dataclass

# @dataclass
# class LossObject:
#     train_dd: np.ndarray
#     valid_dd: np.ndarray
#     train_pi: np.ndarray
#     valid_pi: np.ndarray

# # updated save loss fn

# import csv
# from pathlib import Path 

# def save_loss_to_csv(csv_path: Path, data_loss: LossObject):

#     # csv_path.mkdir(parents=True, exist_ok=True)
#     # this can be done outside the function definition...

#     # need to check if we can do 'a+
#     with open(csv_path, 'a+',  newline='') as f:

#         # might need to check the syntax here
#         csv_writer = csv.writer(f, delimiter=',')

#         row_to_save = [data_loss.train_dd, data_loss.valid_dd, data_loss.train_pi, data_loss.valid_pi]
#         csv_writer.writerow(row_to_save)



def save_and_update_loss_txt(logs_checkpoint, train_loss_dd_unsaved, train_loss_pi_unsaved, valid_loss_dd_unsaved, valid_loss_pi_unsaved):
    ## open the loss logs and append the unseen losses

    if not os.path.exists(logs_checkpoint):
        os.makedirs(logs_checkpoint)

    open_txt_add_arr(logs_checkpoint/f"training_loss_dd.txt", train_loss_dd_unsaved)
    open_txt_add_arr(logs_checkpoint/f"training_loss_pi.txt", train_loss_pi_unsaved)
    open_txt_add_arr(logs_checkpoint/f"valid_loss_dd.txt", valid_loss_dd_unsaved)
    open_txt_add_arr(logs_checkpoint/f"valid_loss_pi.txt", valid_loss_pi_unsaved)


def open_txt_add_arr(filename, array):
    f = open(filename,'a')
    np.savetxt(f, array)
    f.close()