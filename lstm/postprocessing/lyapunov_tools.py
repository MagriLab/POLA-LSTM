import tensorflow as tf
import numpy as np

def lstm_step_comb(u_t, h, c, model, idx, dim=3):
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
    if idx > window_size:  # for correct Jacobian, must multiply W in the beginning
        u_t = tf.reshape(tf.matmul(h, model.layers[1].get_weights()[
            0]) + model.layers[1].get_weights()[1], shape=(1, dim))
    z = tf.keras.backend.dot(u_t, model.layers[0].cell.kernel)
    z += tf.keras.backend.dot(h, model.layers[0].cell.recurrent_kernel)
    z = tf.keras.backend.bias_add(z, model.layers[0].cell.bias)

    z0, z1, z2, z3 = tf.split(z, 4, axis=1)

    i = tf.sigmoid(z0)
    f = tf.sigmoid(z1)
    c_new = f * c + i * tf.tanh(z2)
    o = tf.sigmoid(z3)

    h_new = o * tf.tanh(c_new)
    if idx <= window_size:
        u_t = tf.reshape(tf.matmul(h_new, model.layers[1].get_weights()[
            0]) + model.layers[1].get_weights()[1], shape=(1, dim))
    return u_t, h_new, c_new


def step_and_jac(u_t_in, h, c, model, idx, dim):
    """advances LSTM by one step and computes the Jacobian

    Args:
        u_t_in (tf.EagerTensor): differential equation at time t
        h (tf.EagerTensor): LSTM hidden state at time t
        c (tf.EagerTensor): LSTM cell state at time t
        model (keras.Sequential): trained LSTM
        idx (int): index of current iteration

    Returns:
        u_t_in (tf.EagerTensor): coupled Jacobian at time t
        u_t_out (tf.EagerTensor): LSTM prediction at time t+1
        h_new (tf.EagerTensor): LSTM hidden state at time t+1
        c_new (tf.EagerTensor): LSTM cell state at time t+1
    """
    cell_dim = model.layers[1].get_weights()[0].shape[0]
    with tf.GradientTape(persistent=True) as tape_h:
        tape_h.watch(h)
        with tf.GradientTape(persistent=True) as tape_c:
            tape_c.watch(c)
            u_t_out, h_new, c_new = lstm_step_comb(u_t_in, h, c, model, idx, dim=dim)
            Jac_c_new_c = tf.reshape(tape_c.jacobian(c_new, c), shape=(cell_dim, cell_dim))
            Jac_h_new_c = tf.reshape(tape_c.jacobian(h_new, c), shape=(cell_dim, cell_dim))
        Jac_h_new_h = tf.reshape(tape_h.jacobian(h_new, h), shape=(cell_dim, cell_dim))
        Jac_c_new_h = tf.reshape(tape_h.jacobian(c_new, h), shape=(cell_dim, cell_dim))

    Jac = tf.concat([tf.concat([Jac_c_new_c, Jac_c_new_h], axis=1),
                    tf.concat([Jac_h_new_c, Jac_h_new_h], axis=1)], axis=0)

    return Jac, u_t_out, h_new, c_new
