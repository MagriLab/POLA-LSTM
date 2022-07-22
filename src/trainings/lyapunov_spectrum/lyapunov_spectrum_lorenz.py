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
from lstm.lstm_model import build_pi_model
from lstm.preprocessing.data_processing import (df_train_valid_test_split,
                                                train_valid_test_split)
from lstm.utils.qr_decomp import qr_factorization
from lstm.utils.supress_tf_warning import tensorflow_shutup


warnings.simplefilter(action="ignore", category=FutureWarning)
tf.keras.backend.set_floatx('float64')
tensorflow_shutup()


def lstm_step(window_input, h, c, model, i, dim=3):
    if i > 1:
        window_input = tf.reshape(tf.matmul(h, model.layers[1].get_weights()[
                                  0]) + model.layers[1].get_weights()[1], shape=(1, dim))
    z = tf.keras.backend.dot(window_input, model.layers[0].cell.kernel)
    z += tf.keras.backend.dot(h, model.layers[0].cell.recurrent_kernel)
    z = tf.keras.backend.bias_add(z, model.layers[0].cell.bias)

    z0, z1, z2, z3 = tf.split(z, 4, axis=1)

    i = tf.sigmoid(z0)
    f = tf.sigmoid(z1)
    c_new = f * c + i * tf.tanh(z2)
    o = tf.sigmoid(z3)

    h_new = o * tf.tanh(c_new)

    # out = tf.matmul(h_new, model.layers[1].get_weights()[0]) + model.layers[1].get_weights()[1]
    return window_input, h_new, c_new


def step_and_jac(window_input, h, c, model, i):
    cell_dim = model.layers[1].get_weights()[0].shape[0]

    with tf.GradientTape(persistent=True) as tape_h:
        tape_h.watch(h)
        with tf.GradientTape(persistent=True) as tape_c:
            tape_c.watch(c)
            out, h_new, c_new = lstm_step(window_input, h, c, model, i)
        Jac_c_new_c = tf.reshape(tape_c.jacobian(c_new, c), shape=(cell_dim, cell_dim))
        Jac_h_new_c = tf.reshape(tape_c.jacobian(h_new, c), shape=(cell_dim, cell_dim))
    Jac_h_new_h = tf.reshape(tape_h.jacobian(h_new, h), shape=(cell_dim, cell_dim))
    Jac_c_new_h = tf.reshape(tape_h.jacobian(c_new, h), shape=(cell_dim, cell_dim))
    Jac = tf.concat([tf.concat([Jac_c_new_c, Jac_c_new_h], axis=1),
                    tf.concat([Jac_h_new_c, Jac_h_new_h], axis=1)], axis=0)

    return Jac, out, h_new, c_new


mydf = np.genfromtxt(
    '/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/lorenz_data/CSV/10000/Lorenz_trans_001_norm_10000.csv',
    delimiter=",").astype(
    np.float64)
df_train, df_valid, df_test = df_train_valid_test_split(mydf[1:, :], train_ratio=0.5, valid_ratio=0.25)
time_train, time_valid, time_test = train_valid_test_split(mydf[0, :], train_ratio=0.5, valid_ratio=0.25)

# Windowing
dim = 3
window_size = 100
batch_size = 128
cell_dim = 10
shuffle_buffer_size = df_train.shape[0]
model_path = '/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/models/lorenz/euler/10000-many-diff_loss/'
img_filepath = model_path + "images/time_dev/"

if not os.path.exists(img_filepath):
    os.makedirs(img_filepath)
epochs = 10000

model = build_pi_model(cell_dim, dim=dim)
model.load_weights(model_path + "model/" + str(epochs) + "/weights").expect_partial()
n_length = window_size+50
lyapunov_time, prediction = prediction_closed_loop(
    model, time_test, df_test, n_length, window_size=window_size, c_lyapunov=0.9
)

test_window = create_test_window(df_test, window_size=window_size)
N_units = 10
dt = 0.01  # time step
t_lyap = 0.9**(-1)
norm_time = 1
N_lyap = int(t_lyap/dt)
N = 110*N_lyap

Ntransient = 10*N_lyap
N_test = N - Ntransient
print('N', N, 'Ntran', Ntransient, 'N_test', N_test)
Ttot = np.arange(int(N_test/norm_time)) * dt * norm_time

N_test_norm = int(N_test/norm_time)
print('N_test_norm', N_test_norm)
# Lyapunov Exponents timeseries
LE = np.zeros((N_test_norm, dim))
# Q matrix recorded in time
QQ_t = np.zeros((N_units+N_units, dim, N_test_norm))
# R matrix recorded in time
RR_t = np.zeros((dim, dim, N_test_norm))
window = test_window[:, 0, :]
h = tf.Variable(model.layers[0].get_initial_state(test_window)[0], trainable=False)
c = tf.Variable(model.layers[0].get_initial_state(test_window)[1], trainable=False)
pred = np.zeros(shape=(N, 3))
pred[0, :] = window
delta = scipy.linalg.orth(np.random.rand(N_units+N_units, dim))
Q, R = qr_factorization(delta)
delta = Q[:, :dim]

start_time = time.time()

for i in np.arange(0, Ntransient):
    if i < window_size:
        window = test_window[:, i, :]
    jacobian, window, h, c = step_and_jac(window, h, c, model, i)
    pred[i, :] = window
    delta = np.matmul(jacobian, delta)

    if i % norm_time == 0:
        Q, R = qr_factorization(delta)
        delta = Q[:, :dim]

indx = 0
for i in np.arange(Ntransient, N):
    jacobian, window, h, c = step_and_jac(window, h, c, model, i)
    pred[i, :] = window
    jacobian = tf.transpose(jacobian)
    delta = np.matmul(jacobian, delta)
    if i % norm_time == 0:
        Q, R = qr_factorization(delta)
        delta = Q[:, :dim]

        RR_t[:, :, indx] = R
        QQ_t[:, :, indx] = Q
        LE[indx] = np.abs(np.diag(R[:dim, :dim]))

        indx += 1

        if i % 100 == 0:
            print('Inside closed loop i=', i)

LEs = np.cumsum(np.log(LE[1:]), axis=0) / np.tile(Ttot[1:], (dim, 1)).T
print('Total time: ', time.time()-start_time)
print('Lyapunov exponents: ', LEs[-1])
np.savetxt(model_path+'lyapunov_exp_'+str(N_test)+'.txt', LEs)
print('LEs saved at', model_path+'lyapunov_exp_'+str(N_test)+'.txt')

# Create plot and directly save it
LEs_loaded = np.loadtxt(model_path+'lyapunov_exp_'+str(N_test)+'.txt')
LEs_euler = np.array([1.11109771,  -0.04123815, -14.9934126])
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111)
lyapunov_time = compute_lyapunov_time_arr(np.arange(0, 100000, 0.01), window_size=window_size, c_lyapunov=0.9)
plt.plot(lyapunov_time[:len(LEs_loaded)], LEs_loaded[:, 0],
         label='LSTM LEs +, final value: '+"%.2f" % LEs_loaded[-1, 0])
plt.plot(lyapunov_time[:len(LEs_loaded)], LEs_loaded[:, 1],
         label='LSTM LEs 0, final value: '+"%.2f" % LEs_loaded[-1, 1])
plt.plot(lyapunov_time[:len(LEs_loaded)], LEs_loaded[:, 2],
         label='LSTM LEs -, final value: '+"%.2f" % LEs_loaded[-1, 2])
for i in range(len(LEs_euler)):
    plt.plot(lyapunov_time[:len(LEs_loaded)], np.ones(shape=(1, len(LEs_loaded))).T * LEs_euler[i], 'k--')
    ax.text(lyapunov_time[len(LEs_loaded)]+5, LEs_euler[i], "%.2f" % LEs_euler[i], ha="center")
plt.plot(
    lyapunov_time[: len(LEs_loaded)],
    np.ones(shape=(1, len(LEs_loaded))).T * LEs_euler[2],
    'k--', label="Euler LEs")
plt.xlabel('LT')
plt.xlim(0, lyapunov_time[len(LEs_loaded)]+10)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.75))
plt.title("Lyapunov Exponents of the Lorenz System")
plt.savefig(model_path+'test_lyapunox_exp.png', dpi=100, facecolor="w", bbox_inches="tight")
print('Plot saved at', model_path+'test_lyapunox_exp.png')
plt.close()

plt.plot(pred)
plt.plot(df_test.T[:len(pred), :], ':')
plt.show()
