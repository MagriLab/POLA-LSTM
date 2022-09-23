import sys
import time
import warnings
import einops
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
sys.path.append('../../../')
from lstm.utils.config import load_config_to_dict
from lstm.utils.create_paths import make_img_filepath
from lstm.utils.supress_tf_warning import tensorflow_shutup
from lstm.utils.qr_decomp import qr_factorization
from lstm.preprocessing.data_processing import (df_train_valid_test_split,
                                                train_valid_test_split)
from lstm.lstm_model import load_model
from lstm.closed_loop_tools_mtm import (compute_lyapunov_time_arr,
                                        create_test_window,
                                        prediction_closed_loop)
warnings.simplefilter(action="ignore", category=FutureWarning)
tf.keras.backend.set_floatx('float64')
tensorflow_shutup()



# def lstm_step_comb(u_t, h, c, model, idx, dim=3):
#     """Executes one LSTM step for the Lyapunov exponent computation

#     Args:
#         u_t (tf.EagerTensor): differential equation at time t
#         h (tf.EagerTensor): LSTM hidden state at time t
#         c (tf.EagerTensor): LSTM cell state at time t
#         model (keras.Sequential): trained LSTM
#         idx (int): index of current iteration
#         dim (int, optional): dimension of the lorenz system. Defaults to 3.

#     Returns:
#         u_t (tf.EagerTensor): LSTM prediction at time t/t+1
#         h (tf.EagerTensor): LSTM hidden state at time t+1
#         c (tf.EagerTensor): LSTM cell state at time t+1
#     """
#     if idx > window_size:  # for correct Jacobian, must multiply W in the beginning
#         u_t = tf.reshape(tf.matmul(h, model.layers[1].get_weights()[
#             0]) + model.layers[1].get_weights()[1], shape=(1, dim))
#     z = tf.keras.backend.dot(u_t, model.layers[0].cell.kernel)
#     z += tf.keras.backend.dot(h, model.layers[0].cell.recurrent_kernel)
#     z = tf.keras.backend.bias_add(z, model.layers[0].cell.bias)

#     z0, z1, z2, z3 = tf.split(z, 4, axis=1)

#     i = tf.sigmoid(z0)
#     f = tf.sigmoid(z1)
#     c_new = f * c + i * tf.tanh(z2)
#     o = tf.sigmoid(z3)

#     h_new = o * tf.tanh(c_new)
#     if idx <= window_size:
#         u_t = tf.reshape(tf.matmul(h_new, model.layers[1].get_weights()[
#             0]) + model.layers[1].get_weights()[1], shape=(1, dim))
#     return u_t, h_new, c_new

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
        u_t_temp = u_t
        u_t = u_t[:, ::4]
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
        u_t_temp = u_t
    return u_t_temp, h_new, c_new


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


def step_and_jac_analytical(u_t, h, c, model, idx, dim):
    """advances LSTM by one step and computes the Jacobian

    Args:
        u_t (tf.EagerTensor): differential equation at time t
        h (tf.EagerTensor): LSTM hidden state at time t
        c (tf.EagerTensor): LSTM cell state at time t
        model (keras.Sequential): trained LSTM
        idx (int): index of current iteration

    Returns:
        u_t (tf.EagerTensor): coupled Jacobian at time t
        u_t_out (tf.EagerTensor): LSTM prediction at time t+1
        h_new (tf.EagerTensor): LSTM hidden state at time t+1
        c_new (tf.EagerTensor): LSTM cell state at time t+1
    """
    n_cell = model.layers[1].get_weights()[0].shape[0]
    cell_dim = n_cell
    if idx > window_size:  # for correct Jacobian, must multiply W in the beginning
        u_t = tf.reshape(tf.matmul(h, model.layers[1].get_weights()[
            0]) + model.layers[1].get_weights()[1], shape=(1, dim))
        u_t_temp = u_t
        u_t = u_t[:, ::4]
    z = tf.keras.backend.dot(u_t, model.layers[0].cell.kernel)
    z += tf.keras.backend.dot(h, model.layers[0].cell.recurrent_kernel)
    z = tf.keras.backend.bias_add(z, model.layers[0].cell.bias)

    z0, z1, z2, z3 = tf.split(z, 4, axis=1)

    i = tf.sigmoid(z0)
    f = tf.sigmoid(z1)
    c_tilde = tf.tanh(z2)
    i_c_tilde = i * c_tilde
    c_new = f * c + i_c_tilde
    o = tf.sigmoid(z3)

    h_new = o * tf.tanh(c_new)

    Jac_z_h = tf.transpose(tf.matmul(model.layers[1].get_weights()[0][:, ::4], model.layers[0].cell.kernel)+model.layers[0].cell.recurrent_kernel)
    Jac_i_z = einops.rearrange(tf.linalg.diag(i*(1-i)), '1 i j -> i j')
    Jac_i_h = tf.matmul(Jac_i_z, Jac_z_h[:cell_dim, :])
    Jac_f_h = tf.matmul(einops.rearrange(tf.linalg.diag(f*(1-f)), '1 i j -> i j'), Jac_z_h[cell_dim:2*cell_dim, :])
    Jac_o_h = tf.matmul(einops.rearrange(tf.linalg.diag(o*(1-o)), '1 i j -> i j'), Jac_z_h[3*cell_dim:4*cell_dim, :])
    Jac_c_t_h = tf.matmul(tf.reshape(tf.linalg.diag(1- c_tilde**2), shape=(cell_dim,cell_dim)), Jac_z_h[2*cell_dim:3*cell_dim, :])
    Jac_i_c_tilde = (Jac_c_t_h * tf.transpose(i)+ Jac_i_h*tf.transpose(c_tilde))
    Jac_c_new_c = tf.reshape(tf.linalg.diag(f), shape=(cell_dim,cell_dim)) 
    Jac_h_new_c = tf.reshape(tf.linalg.diag(o * (1- tf.tanh(c_new)**2)), shape=(cell_dim,cell_dim)) *Jac_c_new_c 
    Jac_c_new_h = Jac_i_c_tilde + Jac_f_h * tf.transpose(c) 
    Jac_h_new_h = (tf.matmul(einops.rearrange(tf.linalg.diag(1- tf.tanh(c_new)**2), '1 i j -> i j'), Jac_c_new_h )* tf.transpose(o)+ Jac_o_h*tf.transpose(tf.tanh(c_new)))
    Jac = tf.concat([tf.concat([Jac_c_new_c, Jac_c_new_h], axis=1),
            tf.concat([Jac_h_new_c, Jac_h_new_h], axis=1)], axis=0)
    return Jac, u_t_temp, h_new, c_new

print('Analytical derivative')


mydf = np.genfromtxt(
    '/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/diff_dyn_sys/KS_flow/CSV/KS_160_dx60_rk4_99000_stand_3.52_deltat_0.25_trans.csv',
    # '/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/diff_dyn_sys/KS_flow/CSV/KS_80_2n_dx60_rk4_99000_stand_3.47_deltat_0.25_trans.csv',
    delimiter=",").astype(
    np.float64)
df_train, df_valid, df_test = df_train_valid_test_split(mydf[1:, :], train_ratio=0.5, valid_ratio=0.25)
time_train, time_valid, time_test = train_valid_test_split(mydf[0, :], train_ratio=0.5, valid_ratio=0.25)

model_path = f'/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/models/ks/D160-4n/42500/25-200/'
model_dict = load_config_to_dict(model_path)

dim = df_train.shape[0]
window_size = model_dict['DATA']['WINDOW_SIZE']
n_cell = model_dict['ML_CONSTRAINTS']['N_CELLS']
epochs = 1000 #model_dict['ML_CONSTRAINTS']['N_EPOCHS']
dt = 0.25 #model_dict['DATA']['DELTA T']  # time step


make_img_filepath(model_path)
model = load_model(model_path, epochs, model_dict, dim=dim)
print('--- model successfully loaded---')
# Compare this prediction with the LE prediction
test_window = create_test_window(df_test[::4, :])
model.predict(test_window)
print('--- successfully initialized---')
# Set up parameters for LE computation
t_lyap = 0.09**(-1)
start_time = time.time()
# Set up parameters for LE computation
t_lyap = 0.09**(-1)
norm_time = 1
N_lyap = int(t_lyap/dt)
N = 100*N_lyap
dim = df_train.shape[0]
print(f'Dimension {dim}')
Ntransient = max(int(N/100), window_size+2)
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
pred = np.zeros(shape=(N, dim))
pred[0, :] = u_t

start_time = time.time()

# prepare h,c and c from first window
for i in range(1, window_size+1):
    u_t = test_window[:, i-1, :]
    u_t, h, c = lstm_step_comb(u_t[:, ::4], h, c, model, i, dim)
    pred[i, :] = u_t
    
i=window_size
jacobian, u_t, h, c = step_and_jac(u_t[:, ::4], h, c, model, i, dim)
pred[i, :] = u_t
delta = np.matmul(jacobian, delta)
q, r = qr_factorization(delta)
delta = q[:, :dim]

# compute delta on transient
for i in range(window_size+1, Ntransient):
    jacobian, u_t, h, c = step_and_jac_analytical(u_t[:, ::4], h, c, model, i, dim)
    pred[i, :] = u_t
    delta = np.matmul(jacobian, delta)

    if i % norm_time == 0:
        q, r = qr_factorization(delta)
        delta = q[:, :dim]

print('Finished on Transient')
# compute lyapunov exponent based on qr decomposition
start_time=time.time()
for i in range(Ntransient, N):
    jacobian, u_t, h, c = step_and_jac_analytical(u_t[:, ::4], h, c, model, i, dim)
    indx = i-Ntransient
    pred[i, :] = u_t
    delta = np.matmul(jacobian, delta)
    if i % norm_time == 0:
        q, r = qr_factorization(delta)
        delta = q[:, :dim]

        rr_t[:, :, indx] = r
        qq_t[:, :, indx] = q
        LE[indx] = np.abs(np.diag(r[:dim, :dim]))

        if i % 1000 == 0:
            print(f'Inside closed loop i = {i}')
            if indx != 0:
                lyapunov_exp = np.cumsum(np.log(LE[1:indx]), axis=0) / np.tile(Ttot[1:indx], (dim, 1)).T
                print(f'Lyapunov exponents: {lyapunov_exp[-1] } ')
                print(f'Time for 1000 steps: {time.time()-start_time}')
                start_time = time.time()

lyapunov_exp = np.cumsum(np.log(LE[1:]), axis=0) / np.tile(Ttot[1:], (dim, 1)).T
print(f'Total time: {time.time()-start_time}')
print(f'Final Lyapunov exponents: {lyapunov_exp[-1]}')
np.savetxt(f'{model_path}lyapunov_exp_{N_test}.txt', lyapunov_exp)
print(f'lyapunov_exp saved at {model_path}lyapunov_exp_{N_test}.txt')



np.savetxt(f'{model_path}lyapunov_exp_{N_test}.txt', lyapunov_exp)
print(f'lyapunov_exp saved at {model_path}lyapunov_exp_{N_test}.txt')
