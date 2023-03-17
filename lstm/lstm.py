import numpy as np
import tensorflow as tf
from .loss import Loss


# class LSTMTrainer:

#     def __init__(self, runner: LSTMRunner) -> None:
#         self.runner = runner

#     def train(self, n_epochs: int):
#         ...
#         # self.runner.model.train_step_pi
#         # self.runner(x, train=False)


class LSTMRunner:
    def __init__(self, args, system_name=None, idx_lst=None):
        self.n_cells = args.n_cells
        self.sys_dim = args.sys_dim
        self.activation = args.activation
        self.optimizer = args.optimizer  # edit this
        self.system = system_name
        self.standard_norm = args.standard_norm
        self.reg_weight = args.reg_weighing
        self.args = args
        self.pi_weight = args.pi_weighing
        self.dd_loss_label = args.dd_loss_label

        if idx_lst == None:
            self.idx_lst = np.arange(0, self.sys_dim)
        else:
            self.idx_lst = idx_lst

        self.loss = Loss(self.args, self.idx_lst, self.system, self.dd_loss_label)
        self.build_model()

    # def __call__(self, *args: Any, **kwds: Any) -> Any:
    #     pass

    def build_model(self, kernel_seed=123, recurrent_seed=123):
        self.model = tf.keras.Sequential()
        kernel_init = tf.keras.initializers.GlorotUniform(seed=kernel_seed)
        recurrent_init = tf.keras.initializers.Orthogonal(seed=recurrent_seed)
        self.model.add(
            tf.keras.layers.LSTM(
                self.n_cells, activation='tanh',
                name="LSTM_1", return_sequences=True, kernel_initializer=kernel_init,
                recurrent_initializer=recurrent_init))
        self.model.add(tf.keras.layers.Dense(self.sys_dim, name="Dense_1"))
        self.optimizer = tf.keras.optimizers.Adam()
        self.model.compile(optimizer=self.optimizer, metrics=["mse"], loss=self.loss.data_driven_loss)


    def load_model(self, model_path, epochs):
        self.model.load_weights(model_path / "model" / str(epochs) / "weights").expect_partial()

    @tf.function
    def train_step_pi(self, x_batch_train, y_batch_train, weight_reg=None, weight_pi=None):
        self.change_weights(weight_reg, weight_pi)
        with tf.GradientTape() as tape:
            prediction = self.model(x_batch_train, training=True)
            loss_dd = self.loss.data_driven_loss(prediction, y_batch_train)
            loss_reg = self.loss.l2_loss(prediction)
            loss_pi = self.loss.pi_loss(prediction)
            loss_sum = loss_dd + self.reg_weight*loss_reg + self.pi_weight*loss_pi
        grads = tape.gradient(loss_sum, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss_dd, loss_reg, loss_pi

    @tf.function
    def valid_step_pi(self, x_batch_valid, y_batch_valid):
        prediction = self.model(x_batch_valid, training=True)
        loss_dd = self.loss.data_driven_loss(prediction, y_batch_valid)
        loss_reg = self.loss.l2_loss(prediction)
        loss_pi = self.loss.pi_loss(prediction)
        return loss_dd, loss_reg, loss_pi

    def change_weights(self, weight_reg, weight_pi):
        if weight_reg != None:
            self.reg_weight = weight_reg
        if weight_pi != None:
            self.pi_weight = weight_pi
