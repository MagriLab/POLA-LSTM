import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
sys.path.append('../../../')
from lstm.closed_loop_tools_mtm import (compute_lyapunov_time_arr,
                                        create_test_window,
                                        prediction_closed_loop)
from lstm.lstm_model import load_model
from lstm.preprocessing.data_processing import (df_train_valid_test_split,
                                                train_valid_test_split)
from lstm.utils.qr_decomp import qr_factorization
from lstm.utils.supress_tf_warning import tensorflow_shutup
from lstm.utils.create_paths import make_img_filepath
from lstm.utils.config import load_config_to_dict
warnings.simplefilter(action="ignore", category=FutureWarning)
tf.keras.backend.set_floatx('float64')
tensorflow_shutup()


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


def step_and_jac(u_t_in, h, c, model, idx):
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
            u_t_out, h_new, c_new = lstm_step_comb(u_t_in, h, c, model, i, dim=3)
            Jac_c_new_c = tf.reshape(tape_c.jacobian(c_new, c), shape=(cell_dim, cell_dim))
            Jac_h_new_c = tf.reshape(tape_c.jacobian(h_new, c), shape=(cell_dim, cell_dim))
        Jac_h_new_h = tf.reshape(tape_h.jacobian(h_new, h), shape=(cell_dim, cell_dim))
        Jac_c_new_h = tf.reshape(tape_h.jacobian(c_new, h), shape=(cell_dim, cell_dim))

    Jac = tf.concat([tf.concat([Jac_c_new_c, Jac_c_new_h], axis=1),
                    tf.concat([Jac_h_new_c, Jac_h_new_h], axis=1)], axis=0)

    return Jac, u_t_out, h_new, c_new


mydf = np.genfromtxt(
    '/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/lorenz_data/CSV/100000/rk4_100000_norm_trans.csv',
    delimiter=",").astype(
    np.float64)
df_train, df_valid, df_test = df_train_valid_test_split(mydf[1:, :], train_ratio=0.5, valid_ratio=0.25)
time_train, time_valid, time_test = train_valid_test_split(mydf[0, :], train_ratio=0.5, valid_ratio=0.25)

model_path = f'/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/models/rk4/100000/256/'
model_dict = load_config_to_dict(model_path)

dim = df_train.shape[0]
window_size = model_dict['LORENZ_DATA']['WINDOW_SIZE']
n_cell = model_dict['ML_CONSTRAINTS']['N_CELLS']
epochs = model_dict['ML_CONSTRAINTS']['N_EPOCHS']
dt = model_dict['LORENZ_DATA']['DELTA T']  # time step

make_img_filepath(model_path)
model = load_model(model_path, epochs, model_dict, dim=dim)

# Compare this prediction with the LE prediction
n_length = 2*window_size+1
lyapunov_time, prediction = prediction_closed_loop(
    model, time_test, df_train, n_length, window_size=window_size, c_lyapunov=0.9
)

# Set up parameters for LE computation
t_lyap = 0.9**(-1)
norm_time = 1
N_lyap = int(t_lyap/dt)
N = 2*N_lyap
Ntransient = max(int(N/10), window_size+2)
N_test = N - Ntransient
print(f'N:{N}, Ntran: {Ntransient}, Ntest: {N_test}')
Ttot = np.arange(int(N_test/norm_time)) * dt * norm_time
N_test_norm = int(N_test/norm_time)
print(f'N_test_norm: {N_test_norm}')

# Lyapunov Exponents timeseries
LE = np.zeros((N_test_norm, dim))
# q and r matrix recorded in time
qq_t = np.zeros((n_cell+n_cell, dim, N_test_norm))
rr_t = np.zeros((dim, dim, N_test_norm))
np.random.seed(1)
delta = scipy.linalg.orth(np.random.rand(n_cell+n_cell, dim))
q, r = qr_factorization(delta)
delta = q[:, :dim]

# initialize model and test window
test_window = create_test_window(df_train, window_size=window_size)
u_t = test_window[:, 0, :]
h = tf.Variable(model.layers[0].get_initial_state(test_window)[0], trainable=False)
c = tf.Variable(model.layers[0].get_initial_state(test_window)[1], trainable=False)
pred = np.zeros(shape=(N, 3))
pred[0, :] = u_t

start_time = time.time()

# prepare h,c and c from first window
for i in range(1, window_size+1):
    u_t = test_window[:, i-1, :]
    u_t, h, c = lstm_step_comb(u_t, h, c, model, i)
    pred[i, :] = u_t

# compute delta on transient
for i in range(window_size, Ntransient):
    jacobian, u_t, h, c = step_and_jac(u_t, h, c, model, i)
    pred[i, :] = u_t
    delta = np.matmul(jacobian, delta)

    if i % norm_time == 0:
        q, r = qr_factorization(delta)
        delta = q[:, :dim]

# compute lyapunov exponent based on qr decomposition

for i in range(Ntransient, N):
    indx = i-Ntransient
    jacobian, u_t, h, c = step_and_jac(u_t, h, c, model, i)
    pred[i, :] = u_t
    delta = np.matmul(jacobian, delta)
    if i % norm_time == 0:
        q, r = qr_factorization(delta)
        delta = q[:, :dim]

        rr_t[:, :, indx] = r
        qq_t[:, :, indx] = q
        LE[indx] = np.abs(np.diag(r[:dim, :dim]))

        if i % 100 == 0:
            print(f'Inside closed loop i = {i}')
            if indx != 0:
                lyapunov_exp = np.cumsum(np.log(LE[1:indx]), axis=0) / np.tile(Ttot[1:indx], (dim, 1)).T
                # print(f'Lyapunov exponents: {lyapunov_exp[-1] } ')

lyapunov_exp = np.cumsum(np.log(LE[1:]), axis=0) / np.tile(Ttot[1:], (dim, 1)).T
print(f'Total time: {time.time()-start_time}')
print(f'Final Lyapunov exponents: {lyapunov_exp[-1]}')

np.savetxt('{model_path}lyapunov_exp_{N_test}.txt', lyapunov_exp)
print('lyapunov_exp saved at {model_path}lyapunov_exp_{N_test}.txt')

# Create plot and directly save it
lyapunov_exp_loaded= lyapunov_exp
lyapunov_exp_num= np.array([8.92681657e-01,  1.00317044e-03, -1.45605834e+01])
# lyapunov_exp_num = np.array([ 1.03778442e+00,  3.70627282e-03, -1.49920354e+01])
fig= plt.figure(figsize=(15, 5))
ax= fig.add_subplot(111)
lyapunov_time= compute_lyapunov_time_arr(np.arange(0, 100000, 0.01), window_size=window_size, c_lyapunov=0.9)
plt.plot(lyapunov_time[: len(lyapunov_exp_loaded)], lyapunov_exp_loaded[:, 0],
         label = f'LSTM lyapunov_exp +, final value: {lyapunov_exp_loaded[-1, 0]:.3f}')
plt.plot(lyapunov_time[:len(lyapunov_exp_loaded)], lyapunov_exp_loaded[:, 1],
         label = f'LSTM lyapunov_exp +, final value: {lyapunov_exp_loaded[-1, 1]:.3f}')
plt.plot(lyapunov_time[:len(lyapunov_exp_loaded)], lyapunov_exp_loaded[:, 2],
         label = f'LSTM lyapunov_exp +, final value: {lyapunov_exp_loaded[-1, 2]:.3f}')
for i in range(len(lyapunov_exp_num)):
    plt.plot(lyapunov_time[:len(lyapunov_exp_loaded)], np.ones(
        shape=(1, len(lyapunov_exp_loaded))).T * lyapunov_exp_num[i], 'k--')
    ax.text(lyapunov_time[len(lyapunov_exp_loaded)]+5, lyapunov_exp_num[i], f'{lyapunov_exp_num[i]:.3f}', ha = "center")
plt.plot(
    lyapunov_time[: len(lyapunov_exp_loaded)],
    np.ones(shape=(1, len(lyapunov_exp_loaded))).T * lyapunov_exp_num[2],
    'k--', label="Euler lyapunov_exp")
plt.xlabel('LT')
plt.xlim(0, lyapunov_time[len(lyapunov_exp_loaded)]+10)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.75))
plt.title("Lyapunov Exponents of the Lorenz System")
plt.savefig(f'{model_path}{N_test}_test_lyapunox_exp.png', dpi=100, facecolor="w", bbox_inches="tight")
print(f'Plot saved at {model_path}{N_test}_test_lyapunox_exp.png')
plt.close()


plt.plot(np.arange(0, len(pred)), pred)
plt.plot(np.arange(window_size, window_size + len(prediction)), prediction, 'k:')
plt.plot(np.arange(0, len(pred)), df_train.T[window_size:window_size+len(pred), :], '--')
plt.show()
