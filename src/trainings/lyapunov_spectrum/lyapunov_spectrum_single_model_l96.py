
import sys
import os
from pathlib import Path
import time
import warnings
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy
import h5py
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.set_visible_devices(gpus[2], 'GPU')
        tf.config.set_logical_device_configuration(gpus[2], [tf.config.LogicalDeviceConfiguration(memory_limit=3072)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Virtual devices must be set before GPUs have been initialize
      print(e)

sys.path.append('../../../')
from lstm.closed_loop_tools_mtm import create_test_window
from lstm.lstm import LSTMRunner
from lstm.utils.config import load_config_to_argparse, load_config_to_dict
from lstm.preprocessing.data_processing import (df_train_valid_test_split,
                                                train_valid_test_split, create_df_nd_random_md_idx)
from lstm.utils.qr_decomp import qr_factorization
from lstm.utils.supress_tf_warning import tensorflow_shutup
from lstm.utils.create_paths import make_folder_filepath
from lstm.postprocessing.lyapunov_tools import lstm_step_comb, step_and_jac, step_and_jac_analytical
from lstm.loss import loss_oloop
from lstm.postprocessing import clv_func_clean, clv_angle_plot
warnings.simplefilter(action="ignore", category=FutureWarning)
tf.keras.backend.set_floatx('float64')
tensorflow_shutup()


ref_lyap=np.loadtxt('../Yael_CSV/L96/dim_10_lyapunov_exponents.txt')[-1, :]

mydf = np.genfromtxt(
    '../Yael_CSV/L96/dim_10_rk4_200000_0.01_stand13.33_trans.csv',delimiter=",").astype(np.float64)

model_path = Path('../../models/l96/MLDADS/D10_pi-7/wise-sweep-1')
args = load_config_to_argparse(model_path)
model_dict = load_config_to_dict(model_path)
dim = 10 # df_train.shape[0]
args.sys_dim = dim
args.standard_norm = 13.33
n_random_idx = 7
epochs = max([int(i) for i in next(os.walk(model_path /'model'))[1]])

img_filepath = make_folder_filepath(model_path, 'images')        
df_train, df_valid, df_test = df_train_valid_test_split(
    mydf[1:, :: args.upsampling],
    train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)
time_train, time_valid, time_test = train_valid_test_split(
    mydf[0, ::args.upsampling], train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)
# Compare this prediction with the LE prediction
t_lyap = 1.5**(-1)
N_lyap = int(t_lyap / (args.delta_t*args.upsampling))
idx_lst, train_dataset = create_df_nd_random_md_idx(
    df_train.transpose(),
    args.window_size, args.batch_size, df_train.shape[0],
    n_random_idx=n_random_idx)

for batch, label in train_dataset.take(1):
    print(f'Shape of batch: {batch.shape} \n Shape of Label {label.shape}')
# runner = LSTMRunner(args, 'l96', idx_lst)
# runner.load_model(model_path, epochs)
# model = runner.model

model = load_model(model_path, epochs, model_dict, dim=dim)

batch_pred = model(batch)
print('--- model successfully loaded---')

print('--- successfully initialized---')
random.seed(0)
print(idx_lst)
# Set up parameters for LE computation
start_time = time.time()
norm_time = 1
N_lyap = int(t_lyap/(args.upsampling*args.delta_t))
N = 500*N_lyap
Ntransient = max(int(N/100), args.window_size+2)
N_test = N - Ntransient
print(f'N:{N}, Ntran: {Ntransient}, Ntest: {N_test}')
Ttot = np.arange(int(N_test/norm_time)) * (args.upsampling*args.delta_t) * norm_time
N_test_norm = int(N_test/norm_time)
print(f'N_test_norm: {N_test_norm}')
le_dim = dim
# Lyapunov Exponents timeseries
LE = np.zeros((N_test_norm, le_dim))
# q and r matrix recorded in time
qq_t = np.zeros((args.n_cells+args.n_cells, le_dim, N_test_norm))
rr_t = np.zeros((le_dim, le_dim, N_test_norm))
# Lyapunov Exponents timeseries
LE   = np.zeros((N_test_norm, le_dim))
# Instantaneous Lyapunov Exponents timeseries
IBLE   = np.zeros((N_test_norm,le_dim))
# Q matrix recorded in time
QQ_t = np.zeros((args.n_cells+args.n_cells, le_dim, N_test_norm))
# R matrix recorded in time
RR_t =  np.zeros((le_dim, le_dim, N_test_norm))



np.random.seed(1)
delta = scipy.linalg.orth(np.random.rand(args.n_cells+args.n_cells, le_dim))
q, r = qr_factorization(delta)
delta = q[:, :le_dim]

# initialize model and test window
test_window = create_test_window(df_test, window_size=args.window_size)
u_t = test_window[:, 0, :]
h = tf.Variable(model.layers[0].get_initial_state(test_window)[0], trainable=False)
c = tf.Variable(model.layers[0].get_initial_state(test_window)[1], trainable=False)
pred = np.zeros(shape=(N, dim))
pred[0, :] = u_t

start_time = time.time()

# prepare h,c and c from first window
for i in range(1, args.window_size+1):
    u_t = test_window[:, i-1, :]
    u_t_eval = tf.gather(u_t, idx_lst, axis=1)
    u_t, h, c = lstm_step_comb(u_t_eval, h, c, model, args, i, dim=dim)
    pred[i, :] = u_t

i = args.window_size
u_t_eval = tf.gather(u_t, idx_lst, axis=1)
jacobian, u_t, h, c = step_and_jac(u_t_eval, h, c, model, args, i, dim=dim)
pred[i, :] = u_t
delta = np.matmul(jacobian, delta)
q, r = qr_factorization(delta) #q dimension 200, 20 r (20,20)
delta = q[:, :le_dim]

# compute delta on transient
for i in range(args.window_size+1, Ntransient):
    u_t_eval = tf.gather(u_t, idx_lst, axis=1)
    jacobian, u_t, h, c = step_and_jac_analytical(u_t_eval, h, c, model, args, i, dim=dim)
    pred[i, :] = u_t
    delta = np.matmul(jacobian, delta)

    if i % norm_time == 0:
        q, r = qr_factorization(delta)
        delta = q[:, :le_dim]

print('Finished on Transient')
# compute lyapunov exponent based on qr decomposition
indx=0
for i in range(Ntransient, N):
    u_t_eval = tf.gather(u_t, idx_lst, axis=1)
    jacobian, u_t, h, c = step_and_jac_analytical(u_t_eval, h, c, model, args, i, dim=dim)

    pred[i, :] = u_t
    delta = np.matmul(jacobian, delta)
    if i % norm_time == 0:
        q, r = qr_factorization(delta)
        delta = q[:, :le_dim]            
        qq_t[:,:,indx] = q
        rr_t[:,:,indx] = r

        LE[indx]       = np.abs(np.diag(r))
        for j in range(le_dim):
            IBLE[indx, j] = np.dot(q[:,j].T, np.dot(jacobian,q[:,j]))
        indx += 1


        if i % 10000 == 0:
            print(f'Inside closed loop i = {i}')
            if indx != 0:
                lyapunov_exp = np.cumsum(np.log(LE[1:indx]), axis=0) / np.tile(Ttot[1:indx], (le_dim, 1)).T
                print(f'Lyapunov exponents: {lyapunov_exp[-1] } ')

print(f'Time Duration: {time.time()-start_time}')
thetas_clv, il, D = clv_func_clean.CLV_calculation(qq_t, rr_t, args.sys_dim, 2*args.n_cells, args.delta_t, [3, 1], fname=model_path/f'{N}_clvs.h5', system='lorenz96')

lyapunov_exp = np.cumsum(np.log(LE[1:]), axis=0) / np.tile(Ttot[1:], (le_dim, 1)).T

print(f'Reference exponents: {ref_lyap[:]}')
np.savetxt(model_path/f'{epochs}_lyapunov_exp_{N_test}.txt', lyapunov_exp)
n_lyap=le_dim
fullspace = np.arange(1,n_lyap+1)
fs=10
ax = plt.figure().gca()

# plt.title(r'KS, $26/160 \to 160$ dof')
plt.rcParams.update({'font.size': fs})
plt.grid(True,c='lightgray',linestyle='--', linewidth=0.5)
plt.ylabel(r'$\lambda_k$',fontsize=fs)
plt.xlabel(r'$k$',fontsize=fs)
    
plt.plot(fullspace, ref_lyap[ :n_lyap],'k-s', markersize=8,label='target')
plt.plot(fullspace, lyapunov_exp[-1, :n_lyap],'r-o', markersize=6,label='LSTM')
# plt.plot(fullspace, np.append(np.append(lyapunov_exp_loaded[-1, :7], [0, 0]), lyapunov_exp_loaded[-1, 7:n_lyap-2]),'b-^', markersize=6,label='LSTM - 2 shifted like Vlachas')

plt.legend()
plt.savefig(img_filepath/f'{args.pi_weighing}_{N_test}_scatterplot_lyapunox_exp.png', dpi=100, facecolor="w", bbox_inches="tight")
plt.close()

# plot theta distribution
f2 = h5py.File(Path('../Yael_CSV/L96/ESN_target_CLV_dt_0.01_dim_10.h5'),'r+')

FTCLE_lstm = thetas_clv.T
FTCLE_targ = np.array(f2.get('thetas_clv')).T
N_max = min(FTCLE_lstm.shape[1], FTCLE_targ.shape[1])
FTCLE_lstm = FTCLE_lstm[:, :N_max]
FTCLE_targ = FTCLE_targ[:, :N_max]
clv_angle_plot.plot_clv_pdf(FTCLE_lstm, FTCLE_targ, img_filepath, system='lorenz96')