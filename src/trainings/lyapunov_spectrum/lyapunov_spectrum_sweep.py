
import sys
import os
from pathlib import Path
import time
import warnings
import random
import einops
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
                                                train_valid_test_split, create_df_nd_random_md_mtm_idx)
from lstm.utils.qr_decomp import qr_factorization
from lstm.utils.supress_tf_warning import tensorflow_shutup
from lstm.utils.create_paths import make_folder_filepath
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
        u_t_temp = u_t
        u_t = tf.gather(u_t, idx_lst, axis=1)
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
        u_t = tf.gather(u_t, idx_lst, axis=1)
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

    Jac_z_h = tf.transpose(
        tf.matmul(tf.gather(model.layers[1].get_weights()[0],
                            idx_lst, axis=1),
                  model.layers[0].cell.kernel) + model.layers[0].cell.recurrent_kernel)
    Jac_i_z = einops.rearrange(tf.linalg.diag(i*(1-i)), '1 i j -> i j')
    Jac_i_h = tf.matmul(Jac_i_z, Jac_z_h[:cell_dim, :])
    Jac_f_h = tf.matmul(einops.rearrange(tf.linalg.diag(f*(1-f)), '1 i j -> i j'), Jac_z_h[cell_dim:2*cell_dim, :])
    Jac_o_h = tf.matmul(einops.rearrange(tf.linalg.diag(o*(1-o)), '1 i j -> i j'), Jac_z_h[3*cell_dim:4*cell_dim, :])
    Jac_c_t_h = tf.matmul(
        tf.reshape(tf.linalg.diag(1 - c_tilde ** 2),
                   shape=(cell_dim, cell_dim)),
        Jac_z_h[2 * cell_dim: 3 * cell_dim, :])
    Jac_i_c_tilde = (Jac_c_t_h * tf.transpose(i) + Jac_i_h*tf.transpose(c_tilde))
    Jac_c_new_c = tf.reshape(tf.linalg.diag(f), shape=(cell_dim, cell_dim))
    Jac_h_new_c = tf.reshape(tf.linalg.diag(o * (1 - tf.tanh(c_new)**2)), shape=(cell_dim, cell_dim)) * Jac_c_new_c
    Jac_c_new_h = Jac_i_c_tilde + Jac_f_h * tf.transpose(c)
    Jac_h_new_h = (tf.matmul(einops.rearrange(tf.linalg.diag(1 - tf.tanh(c_new)**2), '1 i j -> i j'),
                   Jac_c_new_h) * tf.transpose(o) + Jac_o_h*tf.transpose(tf.tanh(c_new)))
    Jac = tf.concat([tf.concat([Jac_c_new_c, Jac_c_new_h], axis=1),
                     tf.concat([Jac_h_new_c, Jac_h_new_h], axis=1)], axis=0)
    return Jac, u_t_temp, h_new, c_new


mydf = np.genfromtxt(
    '/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/trainings/Yael_CSV/L63/rk4_100000_norm_trans.csv',
    delimiter=",").astype(
    np.float64)



for name in ['skilled-sweep-11',
 'dashing-sweep-20',
 'winter-sweep-32',
 'avid-sweep-13',
 'eternal-sweep-58',
 'deep-sweep-36',
 'radiant-sweep-55',
 'super-sweep-38',
 'dutiful-sweep-15',
 'legendary-sweep-53',
 'blooming-sweep-51',
 'kind-sweep-60',
 'dulcet-sweep-17',
 'glorious-sweep-31',
 'leafy-sweep-59',
 'solar-sweep-56',
 'classic-sweep-39',
 'ruby-sweep-34',
 'leafy-sweep-14',
 'apricot-sweep-16',
 'toasty-sweep-35',
 'elated-sweep-54',
 'dulcet-sweep-12',
 'legendary-sweep-57',
 'wandering-sweep-52',
 'different-sweep-19',
 'wandering-sweep-37',
 'fiery-sweep-33',
 'amber-sweep-18',
 'devoted-sweep-40']:
    model_path = Path('/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/models/l63/sweep/D10-1') / name 
    model_dict = load_config_to_dict(model_path)

    dim = 3  # df_train.shape[0]
    window_size = model_dict['DATA']['WINDOW_SIZE']
    n_cell = model_dict['ML_CONSTRAINTS']['N_CELLS']
    epochs = max([int(i) for i in next(os.walk(model_path /'model'))[1]])
    print(f'Epochs {epochs}')
    dt = model_dict['DATA']['DELTA T']  # time step
    batch_size = model_dict['ML_CONSTRAINTS']['BATCH_SIZE']
    img_filepath = make_folder_filepath(model_path, 'images')
    img_filapath_folder = make_folder_filepath(Path('/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/models/l63/sweep/D10-1'), 'images')
    model = load_model(model_path, epochs, model_dict, dim=dim)
    upsampling = model_dict['DATA']['UPSAMPLING']
    train_ratio = model_dict['DATA']['TRAINING RATIO']
    valid_ratio = model_dict['DATA']['VALID RATIO']
    random.seed(0)
    idx_lst = random.sample(range(1, 4), 3)
    idx_lst.sort()
    print(idx_lst)
    df_train, df_valid, df_test = df_train_valid_test_split(
        mydf[idx_lst, :: upsampling],
        train_ratio=train_ratio, valid_ratio=valid_ratio)
    time_train, time_valid, time_test = train_valid_test_split(
        mydf[0, ::upsampling], train_ratio=train_ratio, valid_ratio=valid_ratio)
    # Compare this prediction with the LE prediction
    n_length = 2*window_size+1
    n_random_idx = 1
    t_lyap = 0.89**(-1)
    N_lyap = int(t_lyap / (dt*upsampling))
    print(df_train.shape)
    idx_lst, train_dataset = create_df_nd_random_md_mtm_idx(
        df_train.transpose(),
        window_size, 256, df_train.shape[0],
        n_random_idx=n_random_idx)
    print(type(train_dataset))
    for batch, label in train_dataset.take(1):
        print(f'Shape of batch: {batch.shape} \n Shape of Label {label.shape}')
    batch_pred = model(batch)
    print('Analytical derivative')

    print('--- model successfully loaded---')

    print('--- successfully initialized---')
    random.seed(0)
    print(idx_lst)
    # Set up parameters for LE computation
    start_time = time.time()
    norm_time = 1
    N_lyap = int(t_lyap/(upsampling*dt))
    N = 500*N_lyap
    Ntransient = max(int(N/100), window_size+2)
    N_test = N - Ntransient
    print(f'N:{N}, Ntran: {Ntransient}, Ntest: {N_test}')
    Ttot = np.arange(int(N_test/norm_time)) * (upsampling*dt) * norm_time
    N_test_norm = int(N_test/norm_time)
    print(f'N_test_norm: {N_test_norm}')
    le_dim = 3
    # Lyapunov Exponents timeseries
    LE = np.zeros((N_test_norm, le_dim))
    # q and r matrix recorded in time
    qq_t = np.zeros((n_cell+n_cell, le_dim, N_test_norm))
    rr_t = np.zeros((le_dim, le_dim, N_test_norm))
    np.random.seed(1)
    delta = scipy.linalg.orth(np.random.rand(n_cell+n_cell, le_dim))
    q, r = qr_factorization(delta)
    delta = q[:, :le_dim]

    # initialize model and test window
    test_window = create_test_window(df_test, window_size=window_size)
    u_t = test_window[:, 0, :]
    h = tf.Variable(model.layers[0].get_initial_state(test_window)[0], trainable=False)
    c = tf.Variable(model.layers[0].get_initial_state(test_window)[1], trainable=False)
    pred = np.zeros(shape=(N, dim))
    pred[0, :] = u_t

    start_time = time.time()

    # prepare h,c and c from first window
    for i in range(1, window_size+1):
        u_t = test_window[:, i-1, :]
        u_t_eval = tf.gather(u_t, idx_lst, axis=1)
        u_t, h, c = lstm_step_comb(u_t_eval, h, c, model, i, dim)
        pred[i, :] = u_t

    i = window_size
    u_t_eval = tf.gather(u_t, idx_lst, axis=1)
    jacobian, u_t, h, c = step_and_jac(u_t_eval, h, c, model, i, dim)
    pred[i, :] = u_t
    delta = np.matmul(jacobian, delta)
    q, r = qr_factorization(delta)
    delta = q[:, :le_dim]

    # compute delta on transient
    for i in range(window_size+1, Ntransient):
        u_t_eval = tf.gather(u_t, idx_lst, axis=1)
        jacobian, u_t, h, c = step_and_jac_analytical(u_t_eval, h, c, model, i, dim)
        pred[i, :] = u_t
        delta = np.matmul(jacobian, delta)

        if i % norm_time == 0:
            q, r = qr_factorization(delta)
            delta = q[:, :le_dim]

    print('Finished on Transient')
    # compute lyapunov exponent based on qr decomposition

    for i in range(Ntransient, N):
        u_t_eval = tf.gather(u_t, idx_lst, axis=1)
        jacobian, u_t, h, c = step_and_jac_analytical(u_t_eval, h, c, model, i, dim)
        indx = i-Ntransient
        pred[i, :] = u_t
        delta = np.matmul(jacobian, delta)
        if i % norm_time == 0:
            q, r = qr_factorization(delta)
            delta = q[:, :le_dim]

            rr_t[:, :, indx] = r
            qq_t[:, :, indx] = q
            LE[indx] = np.abs(np.diag(r[:le_dim, :le_dim]))

            if i % 1000 == 0:
                print(f'Inside closed loop i = {i}')
                if indx != 0:
                    lyapunov_exp = np.cumsum(np.log(LE[1:indx]), axis=0) / np.tile(Ttot[1:indx], (le_dim, 1)).T
                    print(f'Lyapunov exponents: {lyapunov_exp[-1] } ')

    lyapunov_exp = np.cumsum(np.log(LE[1:]), axis=0) / np.tile(Ttot[1:], (le_dim, 1)).T

    ref_lyap=np.loadtxt('/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/trainings/Yael_CSV/L63/l63_lyapunov_exponents.txt')
    print(f'Reference exponents: {ref_lyap[-1, :]}')
    np.savetxt(f'{img_filepath}{epochs}_lyapunov_exp_{N_test}.txt', lyapunov_exp)
    n_lyap=3
    fullspace = np.arange(1,n_lyap+1)
    fs=12
    ax = plt.figure().gca()

    # plt.title(r'KS, $26/160 \to 160$ dof')
    plt.rcParams.update({'font.size': fs})
    plt.grid(True,c='lightgray',linestyle='--', linewidth=0.5)
    plt.ylabel(r'$\lambda_k$',fontsize=fs)
    plt.xlabel(r'$k$',fontsize=fs)
        
    plt.plot(fullspace, ref_lyap[-1, :n_lyap],'k-s', markersize=8,label='target')
    plt.plot(fullspace, lyapunov_exp[-1, :n_lyap],'r-o', markersize=6,label='LSTM')
    # plt.plot(fullspace, np.append(np.append(lyapunov_exp_loaded[-1, :7], [0, 0]), lyapunov_exp_loaded[-1, 7:n_lyap-2]),'b-^', markersize=6,label='LSTM - 2 shifted like Vlachas')

    plt.legend()
    plt.savefig(img_filepath/f'{epochs}_{N_test}_scatterplot_lyapunox_exp.png', dpi=100, facecolor="w", bbox_inches="tight")
    plt.savefig(img_filapath_folder/f'{name}_scatterplot_lyapunox_exp.png', dpi=100, facecolor="w", bbox_inches="tight")
    plt.close()
    print(f'{name} : Lyapunov exponents: {lyapunov_exp[-1] } ')
