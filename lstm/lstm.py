import numpy as np
import tensorflow as tf
from .loss import Loss
from .closed_loop_tools_mtm import create_test_window


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

    def __call__(self, input):
        return self.model(input)

    def build_model(self, kernel_seed=123, recurrent_seed=123):
        self.model = tf.keras.Sequential()
        kernel_init = tf.keras.initializers.GlorotUniform(seed=kernel_seed)
        recurrent_init = tf.keras.initializers.Orthogonal(seed=recurrent_seed)
        self.model.add(
            tf.keras.layers.LSTM(
                self.n_cells,
                activation="tanh",
                name="LSTM_1",
                return_sequences=True,
                kernel_initializer=kernel_init,
                recurrent_initializer=recurrent_init,
            )
        )
        self.model.add(tf.keras.layers.Dense(self.sys_dim, name="Dense_1"))
        self.optimizer = tf.keras.optimizers.Adam()
        self.model.compile(
            optimizer=self.optimizer, metrics=["mse"], loss=self.loss.data_driven_loss
        )

    def load_model(self, model_path, epochs):
        self.model.load_weights(
            model_path / "model" / str(epochs) / "weights"
        ).expect_partial()

    @tf.function
    def train_step_pi(
        self, x_batch_train, y_batch_train, weight_reg=None, weight_pi=None
    ):
        self.change_weights(weight_reg, weight_pi)
        with tf.GradientTape() as tape:
            prediction = self.model(x_batch_train, training=True)
            loss_dd = self.loss.data_driven_loss(prediction, y_batch_train)
            loss_reg = self.loss.l2_loss(prediction)
            loss_pi = self.loss.pi_loss(prediction)
            loss_sum = loss_dd + self.reg_weight * loss_reg + self.pi_weight * loss_pi
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
    
    def prediction(self, df, N=1000):
        test_window = create_test_window(df, window_size=self.args.window_size)
        u_t = test_window[:, 0, :]
        
        h = tf.Variable(self.model.layers[0].get_initial_state(test_window)[0], trainable=False)
        c = tf.Variable(self.model.layers[0].get_initial_state(test_window)[1], trainable=False)
        pred = np.zeros(shape=(N, self.sys_dim))
        pred[0, :] = u_t

        # prepare h,c and c from first window
        for i in range(1, self.args.window_size + 1):
            u_t = test_window[:, i - 1, :]
            u_t_eval = tf.gather(u_t, self.idx_lst, axis=1)
            u_t, h, c = self.lstm_step(u_t_eval, h, c)
            pred[i, :] = u_t
        for i in range(self.args.window_size + 1, N):
            u_t_eval = tf.gather(u_t, self.idx_lst, axis=1)
            u_t, h, c = self.lstm_step(u_t_eval, h, c)
            pred[i, :] = u_t

        return pred
    
    def lstm_step(self, u_t, h, c):
        """Executes one LSTM step for the Lyapunov exponent computation

        Args:
            u_t (tf.EagerTensor): differential equation at time t
            h (tf.EagerTensor): LSTM hidden state at time t
            c (tf.EagerTensor): LSTM cell state at time t
            model (keras.Sequential): trained LSTM
            idx (int): index of current iteration
            dim (int, optional): dimension of the lorenz system. Defaults to 3.

        Returns:
            u_t (tf.EagerTensor): LSTM prediction at time t/t+1
            h (tf.EagerTensor): LSTM hidden state at time t+1
            c (tf.EagerTensor): LSTM cell state at time t+1
        """
        z = tf.keras.backend.dot(u_t, self.model.layers[0].cell.kernel)
        z += tf.keras.backend.dot(h, self.model.layers[0].cell.recurrent_kernel)
        z = tf.keras.backend.bias_add(z, self.model.layers[0].cell.bias)

        z0, z1, z2, z3 = tf.split(z, 4, axis=1)

        i = tf.sigmoid(z0)
        f = tf.sigmoid(z1)
        c_new = f * c + i * tf.tanh(z2)
        o = tf.sigmoid(z3)

        h_new = o * tf.tanh(c_new)

        u_t = tf.reshape(
            tf.matmul(h_new, self.model.layers[1].get_weights()[0])
            + self.model.layers[1].get_weights()[1],
            shape=(1, self.sys_dim),
        )
        return u_t, h_new, c_new

