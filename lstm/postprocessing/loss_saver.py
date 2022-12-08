import os
from pathlib import Path
from typing import List
import numpy as np
import tensorflow as tf


def loss_arr_to_tensorboard(
        logs_checkpoint: Path, train_loss_dd_tracker: List[float],
        train_loss_pi_tracker: List[float],
        valid_loss_dd_tracker: List[float],
        valid_loss_pi_tracker: List[float]) -> None:
    """
    Writes the training and validation loss values to TensorBoard.

    Args:
        logs_checkpoint (Path): path to the logs checkpoint directory.
        train_loss_dd_tracker (List[float]): data-driven training loss values for each epoch.
        train_loss_pi_tracker (List[float]): physics-informed training loss values for each epoch.
        valid_loss_dd_tracker (List[float]): data-driven validation loss values for each epoch.
        valid_loss_pi_tracker (List[float]): physics-informed validation loss values for each epoch.

    Returns:    
        None
    """
    summary_dir_train = os.path.join(logs_checkpoint, "train")
    summary_writer_train = tf.summary.create_file_writer(summary_dir_train)
    summary_dir_valid = os.path.join(logs_checkpoint, "valid")
    summary_writer_valid = tf.summary.create_file_writer(summary_dir_valid)
    for cont in range(0, len(train_loss_dd_tracker)):
        with summary_writer_train.as_default():
            tf.summary.scalar(name="epoch:physics_informed_loss", data=train_loss_pi_tracker[cont], step=cont)
            tf.summary.scalar(name="epoch:data_driven_loss", data=train_loss_dd_tracker[cont], step=cont)
            tf.summary.scalar(name="epoch:total_loss",
                              data=train_loss_dd_tracker[cont]+train_loss_pi_tracker[cont], step=cont)
        with summary_writer_valid.as_default():
            tf.summary.scalar(name="epoch:physics_informed_loss", data=valid_loss_pi_tracker[cont], step=cont)
            tf.summary.scalar(name="epoch:data_driven_loss", data=valid_loss_dd_tracker[cont], step=cont)
            tf.summary.scalar(name="epoch:total_loss",
                              data=valid_loss_dd_tracker[cont]+valid_loss_pi_tracker[cont], step=cont)
    summary_writer_train.flush()
    summary_writer_valid.flush()


def save_and_update_loss_txt(
        logs_checkpoint: Path, train_loss_dd_unsaved: List[float],
        train_loss_pi_unsaved: List[float],
        valid_loss_dd_unsaved: List[float],
        valid_loss_pi_unsaved: List[float]) -> None:
    """open the loss logs and append the unseen losses

    Args:
        logs_checkpoint (Path): path to the logs checkpoint directory
        train_loss_dd_unsaved (List[float]): list of unseen data-driven training losses
        train_loss_pi_unsaved (List[float]): list of unseen physics-informed training losses
        valid_loss_dd_unsaved (List[float]): list of unseen data-driven validation losses
        valid_loss_pi_unsaved (List[float]): list of unseen physics-informed validation losses
    """

    logs_checkpoint.mkdir(parents=True, exist_ok=True)
    open_txt_add_arr(logs_checkpoint/f"training_loss_dd.txt", train_loss_dd_unsaved)
    open_txt_add_arr(logs_checkpoint/f"training_loss_pi.txt", train_loss_pi_unsaved)
    open_txt_add_arr(logs_checkpoint/f"valid_loss_dd.txt", valid_loss_dd_unsaved)
    open_txt_add_arr(logs_checkpoint/f"valid_loss_pi.txt", valid_loss_pi_unsaved)


def open_txt_add_arr(filename: str, array: List[float]) -> None:
    """Opens a text file and appends a list of numbers to it.

    Args:
        filename (str): name of the text file
        array: array to append to the file.

    Returns:
        None
    """
    f = open(filename, 'a')
    np.savetxt(f, array)
    f.close()
