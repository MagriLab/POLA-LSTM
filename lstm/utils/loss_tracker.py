import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from typing import List


class LossTracker:
    def __init__(self, logs_checkpoint):
        self.train_loss_dd_tracker = np.array([])
        self.train_loss_pi_tracker = np.array([])
        self.train_loss_reg_tracker = np.array([])
        self.valid_loss_dd_tracker = np.array([])
        self.valid_loss_pi_tracker = np.array([])
        self.valid_loss_reg_tracker = np.array([])
        self.true_loss_val_tracker = np.array([])

        self.logs_checkpoint = logs_checkpoint
        self.logs_checkpoint.mkdir(parents=True, exist_ok=True)


    def append_loss_to_tracker(self, loss_type, loss_dd, loss_reg, loss_pi, n_batches):
        if loss_type == 'train':
            self.train_loss_dd_tracker = np.append(self.train_loss_dd_tracker, loss_dd/n_batches)
            self.train_loss_pi_tracker = np.append(self.train_loss_pi_tracker, loss_pi/n_batches)
            self.train_loss_reg_tracker = np.append(self.train_loss_reg_tracker, loss_reg/n_batches)
        if loss_type == 'valid':
            self.valid_loss_dd_tracker = np.append(self.valid_loss_dd_tracker, loss_dd/n_batches)
            self.valid_loss_pi_tracker = np.append(self.valid_loss_pi_tracker, loss_pi/n_batches)
            self.valid_loss_reg_tracker = np.append(self.valid_loss_reg_tracker, loss_reg/n_batches)
        self.save_and_update_loss_txt(loss_type)

    def loss_arr_to_tensorboard(self, logs_checkpoint: Path) -> None:
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
        for cont in range(0, len(self.train_loss_dd_tracker)):
            with summary_writer_train.as_default():
                tf.summary.scalar(name="epoch:physics_informed_loss", data=self.train_loss_pi_tracker[cont], step=cont)
                tf.summary.scalar(name="epoch:data_driven_loss", data=self.train_loss_dd_tracker[cont], step=cont)
                tf.summary.scalar(name="epoch:reg_loss", data=self.train_loss_reg_tracker[cont], step=cont)
            with summary_writer_valid.as_default():
                tf.summary.scalar(name="epoch:physics_informed_loss", data=self.valid_loss_pi_tracker[cont], step=cont)
                tf.summary.scalar(name="epoch:data_driven_loss", data=self.valid_loss_dd_tracker[cont], step=cont)
                tf.summary.scalar(name="epoch:reg_loss", data=self.valid_loss_reg_tracker[cont], step=cont)
        summary_writer_train.flush()
        summary_writer_valid.flush()

    def save_and_update_loss_txt(self, loss_type) -> None:
        """open the loss logs and append the unseen losses

        Args:
            logs_checkpoint (Path): path to the logs checkpoint directory
            train_loss_dd_unsaved (List[float]): list of unseen data-driven training losses
            train_loss_pi_unsaved (List[float]): list of unseen physics-informed training losses
            valid_loss_dd_unsaved (List[float]): list of unseen data-driven validation losses
            valid_loss_pi_unsaved (List[float]): list of unseen physics-informed validation losses
        """
        if loss_type == 'train':
            self.open_txt_add_arr(self.logs_checkpoint/f"training_loss_dd.txt", self.train_loss_dd_tracker[-1:])
            self.open_txt_add_arr(self.logs_checkpoint/f"training_loss_pi.txt", self.train_loss_pi_tracker[-1:])
            self.open_txt_add_arr(self.logs_checkpoint/f"training_loss_reg.txt", self.train_loss_reg_tracker[-1:])
        if loss_type == 'valid':
            self.open_txt_add_arr(self.logs_checkpoint/f"valid_loss_dd.txt", self.valid_loss_dd_tracker[-1:])
            self.open_txt_add_arr(self.logs_checkpoint/f"valid_loss_pi.txt", self.valid_loss_pi_tracker[-1:])
            self.open_txt_add_arr(self.logs_checkpoint/f"valid_loss_reg.txt", self.valid_loss_reg_tracker[-1:])

    def open_txt_add_arr(self, filename: Path, array: List[float]) -> None:
        """Opens a text file and appends a list of numbers to it.

        Args:
            filename (Path): name of the text file
            array: array to append to the file.

        Returns:
            None
        """
        with open(filename, mode='at', newline='\n') as f:
            np.savetxt(f, array, delimiter=' ', newline='\n')
