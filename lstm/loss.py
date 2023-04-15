import tensorflow as tf
from .differential_equations.lorenz96 import RK4_step_l96, euler_step_l96
from .differential_equations.ks import ks_time_step_batch

class Loss():
    def __init__(self, args, idx_lst, system, dd_loss_label) -> None:
        self.args = args
        self.idx_lst = idx_lst
        self.system = system
        self.dd_loss_label = dd_loss_label
        print(f"Numerial Equation: {self.system}")
    
    @tf.function
    def data_driven_loss(self, prediction, y_batch_train):
        mse = tf.keras.losses.MeanSquaredError()
        if self.dd_loss_label == 'full':
            loss_dd = mse(y_batch_train, prediction)
        else:
            loss_dd = mse(tf.gather(y_batch_train, indices=self.idx_lst, axis=2),
                        tf.gather(prediction, indices=self.idx_lst, axis=2))
        return loss_dd

    def l2_loss(self, prediction):
        return tf.nn.l2_loss(prediction)

    def pi_loss(self, prediction):
        mse = tf.keras.losses.MeanSquaredError()
        if self.system == "KS":
            solver_time_step = ks_time_step_batch(prediction*self.args.standard_norm,
                                                  d=self.args.d, M=self.args.M, h=self.args.h)
            loss_pi = mse(prediction[:, 1:, :]*self.args.standard_norm, solver_time_step[:, :-1, :])
        elif self.system=='l96':
            loss_pi = mse(prediction[:, 1:, :]*self.args.standard_norm, RK4_step_l96(prediction*self.args.standard_norm, delta_t=self.args.delta_t)[:, :-1, :])
        else:
            print(f"{self.system} not defined yet")
            loss_pi=0.0
        return loss_pi

    @tf.function
    def loss_oloop(y_true, y_pred, washout=0):
        mse = tf.keras.losses.MeanSquaredError()
        loss = mse(y_true[:, washout:, :], y_pred[:, washout:, :])
        return loss

@tf.function
def loss_oloop(y_true, y_pred, washout=0):
    mse = tf.keras.losses.MeanSquaredError()
    loss = mse(y_true[:, washout:, :], y_pred[:, washout:, :])
    return loss

