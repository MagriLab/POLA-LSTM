import sys
import os
from pathlib import Path
import time
import warnings
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
import h5py

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.set_visible_devices(gpus[2], "GPU")
        tf.config.set_logical_device_configuration(
            gpus[2], [tf.config.LogicalDeviceConfiguration(memory_limit=3072)]
        )
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialize
        print(e)

sys.path.append("../../../")
from lstm.closed_loop_tools_mtm import create_test_window
from lstm.lstm import LSTMRunner
from lstm.utils.config import load_config_to_argparse
from lstm.preprocessing.data_class import Dataclass
from lstm.utils.qr_decomp import qr_factorization
from lstm.utils.supress_tf_warning import tensorflow_shutup
from lstm.utils.create_paths import make_folder_filepath
from lstm.utils.config import load_config_to_argparse, load_config_to_dict
from lstm.postprocessing.lyapunov_tools import (
    lstm_step_comb,
    step_and_jac,
    step_and_jac_analytical,
)
from lstm.postprocessing.clv_func_clean import normalize
from lstm.postprocessing import clv_func_clean, clv_angle_plot

warnings.simplefilter(action="ignore", category=FutureWarning)
tf.keras.backend.set_floatx("float64")
tensorflow_shutup()


# data.ref_lyap=np.loadtxt('../Yael_CSV/L96/dim_10_lyapunov_exponents.txt')[-1, :]

# ref_clv_hf = h5py.File(Path('../Yael_CSV/L96/ESN_target_CLV_dt_0.01_dim_10.h5'),'r+')
# FTCLE_targ = np.array(ref_clv_hf.get('thetas_clv')).T
# ref_clv_hf.close()
def compute_CLV(QQ, RR):
    """
    Calculates the Covariant Lyapunov Vectors (CLVs) using the Ginelli et al, PRL 2007 method.

    Args:
    - QQ (numpy.ndarray): matrix containing the timeseries of Gram-Schmidt vectors (shape: (n_cells_x2,NLy,tly))
    - RR (numpy.ndarray): matrix containing the timeseries of upper-triangualar R  (shape: (NLy,NLy,tly))
    - NLy (int): number of Lyapunov exponents
    - n_cells_x2 (int): dimension of the hidden states
    - dt (float): integration time step
    - subspace_LEs_indeces (numpy.ndarray): indices of the Lyapunov exponents signs for positive and neutral. (shape: (2,))

    Returns:
    - nothing
    """
    n_cells_x2 = QQ.shape[0]
    NLy = QQ.shape[1]
    dt=0.25
    tly = np.shape(QQ)[-1]
    su = int(tly / 10)
    sd = int(tly / 10)
    s  = su          # index of spinup time
    e  = tly+1 - sd  # index of spindown time
    tau = int(dt/dt)     #time for finite-time lyapunov exponents

    #Calculation of CLVs
    C = np.zeros((NLy,NLy,tly))  # coordinates of CLVs in local GS vector basis
    D = np.zeros((NLy,tly))  # diagonal matrix
    V = np.zeros((n_cells_x2,NLy,tly))  # coordinates of CLVs in physical space (each column is a vector)

    # FTCLE
    il  = np.zeros((NLy,tly+1)) #Finite-time lyapunov exponents along CLVs

    # initialise components to I
    C[:,:,-1] = np.eye(NLy)
    D[:,-1]   = np.ones(NLy)
    V[:,:,-1] = np.dot(np.real(QQ[:,:,-1]), C[:,:,-1])

    for i in reversed(range( tly-1 ) ):
        C[:,:,i], D[:,i]        = normalize(scipy.linalg.solve_triangular(np.real(RR[:,:,i]), C[:,:,i+1]))
        V[:,:,i]                = np.dot(np.real(QQ[:,:,i]), C[:,:,i])

    # FTCLE computations
    for j in np.arange(D.shape[1]): #time loop
        il[:,j] = -(1./dt)*np.log(D[:,j])
        

    #normalize CLVs before measuring their angles.
    timetot = np.shape(V)[-1]

    for i in range(NLy):
        for t in range(timetot-1):
            V[:,i,t] = V[:,i,t] / np.linalg.norm(V[:,i,t])
    return V


le_sweep_models = [ "lively-sweep-12", "blooming-sweep-10", "celestial-sweep-8", "icy-sweep-9", "legendary-sweep-4", "tough-sweep-5", "dazzling-sweep-3"] #"generous-sweep-8"] #, "misunderstood-sweep-12", "divine-sweep-2", "woven-sweep-4",

norm = 13.33
sweep_path = Path('../L96_d20_rk4')
for folder_name in list(filter(lambda x: x != 'images', next(os.walk(sweep_path))[1])):
    sweep_models = list(filter(lambda x: x != 'images', next(os.walk(sweep_path/folder_name))[1]))
    img_filepath_folder = make_folder_filepath(sweep_path / folder_name,  'images')
    for model_name in sweep_models:
        if model_name in le_sweep_models:
            print(model_name)
            model_path = sweep_path / folder_name / model_name
            args = load_config_to_argparse(model_path)
            dim = 20 # df_train.shape[]
            args.sys_dim = dim
            args.data_path = Path('../Yael_CSV/L96/dim_20_rk4_200000_0.01_stand13.33_trans.csv')
            args.train_ratio = args.train_ratio*42500/200000
            args.valid_ratio = args.train_ratio*42500/200000
            args.lyap_path = Path('../Yael_CSV/L96/dim_20_lyapunov_exponents.txt')
            data = Dataclass(args)
            epochs = max([int(i) for i in next(os.walk(model_path /'model'))[1]])

            img_filepath = make_folder_filepath(model_path, 'images')        

            # Compare this prediction with the LE prediction
            t_lyap = 1.55**(-1)
            N_lyap = int(t_lyap / (args.delta_t * args.upsampling))
            runner = LSTMRunner(args, "KS_dd", data.idx_lst)
            runner.load_model(model_path, epochs)
            model = runner.model
            print("--- successfully initialized---")
            for batch, label in data.train_dataset.take(1):
                print(batch.shape)
            batch_pred = model(batch)

            #
            random.seed(0)
            # Set up parameters for LE computation
            start_time = time.time()
            norm_time = 1
            N_lyap = int(t_lyap / (args.upsampling * args.delta_t))
            N = 1 00* N_lyap
            print(f"{N, args.upsampling*args.delta_t}")
            Ntransient = max(int(N / 100), args.window_size + 2)
            N_test = N - Ntransient
            print(f"N:{N}, Ntran: {Ntransient}, Ntest: {N_test}")
            Ttot = (
                np.arange(int(N_test / norm_time))
                * (args.upsampling * args.delta_t)
                * norm_time
            )
            N_test_norm = int(N_test / norm_time)
            print(f"N_test_norm: {N_test_norm}")
            le_dim = 20
            # Lyapunov Exponents timeseries
            LE = np.zeros((N_test_norm, le_dim))
            # q and r matrix recorded in time
            qq_t = np.zeros((args.n_cells + args.n_cells, le_dim, N_test_norm))
            rr_t = np.zeros((le_dim, le_dim, N_test_norm))
            # Lyapunov Exponents timeseries
            LE = np.zeros((N_test_norm, le_dim))
            # Instantaneous Lyapunov Exponents timeseries
            IBLE = np.zeros((N_test_norm, le_dim))
            # Q matrix recorded in time
            QQ_t = np.zeros((args.n_cells + args.n_cells, le_dim, N_test_norm))
            # R matrix recorded in time
            RR_t = np.zeros((le_dim, le_dim, N_test_norm))

            np.random.seed(1)
            delta = scipy.linalg.orth(np.random.rand(args.n_cells + args.n_cells, le_dim))
            q, r = qr_factorization(delta)
            delta = q[:, :le_dim]

            # initialize model and test window
            test_window = create_test_window(data.df_test, window_size=args.window_size)
            u_t = test_window[:, 0, :]
            h = tf.Variable(
                model.layers[0].get_initial_state(test_window)[0], trainable=False
            )
            c = tf.Variable(
                model.layers[0].get_initial_state(test_window)[1], trainable=False
            )
            pred = np.zeros(shape=(N, dim))
            pred[0, :] = u_t
            # prepare h,c and c from first window
            for i in range(1, args.window_size + 1):
                u_t = test_window[:, i - 1, :]
                u_t_eval = tf.gather(u_t, data.idx_lst, axis=1)
                u_t, h, c = lstm_step_comb(
                    u_t_eval, h, c, model, args, i, data.idx_lst, dim
                )
                pred[i, :] = u_t

            i = args.window_size
            u_t_eval = tf.gather(u_t, data.idx_lst, axis=1)
            jacobian, u_t, h, c = step_and_jac(
                u_t_eval, h, c, model, args, i, data.idx_lst, dim
            )
            pred[i, :] = u_t
            delta = np.matmul(jacobian, delta)
            q, r = qr_factorization(delta)
            delta = q[:, :le_dim]

            # compute delta on transient
            for i in range(args.window_size + 1, Ntransient):
                u_t_eval = tf.gather(u_t, data.idx_lst, axis=1)
                jacobian, u_t, h, c = step_and_jac_analytical(
                    u_t_eval, h, c, model, args, i, data.idx_lst, dim
                )
                pred[i, :] = u_t
                delta = np.matmul(jacobian, delta)
                if i % norm_time == 0:
                    q, r = qr_factorization(delta)
                    delta = q[:, :le_dim]
            indx = 0
            print("Finished on Transient")
            # compute lyapunov exponent based on qr decomposition
            start_time = time.time()
            for i in range(Ntransient, N):
                u_t_eval = tf.gather(u_t, data.idx_lst, axis=1)
                jacobian, u_t, h, c = step_and_jac_analytical(
                    u_t_eval, h, c, model, args, i, data.idx_lst, dim
                )
                pred[i, :] = u_t

                delta = np.matmul(jacobian, delta)
                if i % norm_time == 0:
                    q, r = qr_factorization(delta)
                    delta = q[:, :le_dim]
                    qq_t[:, :, indx] = q
                    rr_t[:, :, indx] = r

                    LE[indx] = np.abs(np.diag(r))
                    for j in range(le_dim):
                        IBLE[indx, j] = np.dot(q[:, j].T, np.dot(jacobian, q[:, j]))
                    indx += 1

                    if i % 1000 == 0:
                        print(f"Inside closed loop i = {i}")
                        if indx > 1:
                            lyapunov_exp = (
                                np.cumsum(np.log(LE[1:indx]), axis=0)
                                / np.tile(Ttot[1:indx], (le_dim, 1)).T
                            )
                            print(f"Lyapunov exponents: {lyapunov_exp.shape, lyapunov_exp[-1] } ")

            # thetas_clv, il, D = clv_func_clean.CLV_calculation(qq_t, rr_t, args.sys_dim, 2*args.n_cells, args.delta_t, [6, 1], fname=model_path/f'{N}_clvs.h5', system='lorenz96')

            lyapunov_exp = (
                np.cumsum(np.log(LE[1:]), axis=0) / np.tile(Ttot[1:], (le_dim, 1)).T
            )

            print(f"Reference exponents: {data.ref_lyap[-1, :]}")
            np.savetxt(model_path / f"{epochs}_lyapunov_exp_{N_test}.txt", lyapunov_exp)
            n_lyap = 20
            fullspace = np.arange(1, n_lyap + 1)
            fs = 10
            ax = plt.figure().gca()

            # plt.title(r'KS, $26/160 \to 160$ dof')
            plt.rcParams.update({"font.size": fs})
            plt.grid(True, c="lightgray", linestyle="--", linewidth=0.5)
            plt.ylabel(r"$\lambda_k$", fontsize=fs)
            plt.xlabel(r"$k$", fontsize=fs)

            plt.plot(fullspace, data.ref_lyap[-1, :n_lyap], "k-s", markersize=8, label="target")
            plt.plot(
                fullspace, lyapunov_exp[-1, :n_lyap], "r-o", markersize=6, label="LSTM"
            )
            # plt.plot(fullspace, np.append(np.append(lyapunov_exp_loaded[-1, :7], [0, 0]), lyapunov_exp_loaded[-1, 7:n_lyap-2]),'b-^', markersize=6,label='LSTM - 2 shifted like Vlachas')

            plt.legend()
            plt.savefig(
                img_filepath / f"{args.pi_weighing}_{N_test}_scatterplot_lyapunox_exp.png",
                dpi=100,
                facecolor="w",
                bbox_inches="tight",
            )
            plt.savefig(
                img_filepath_folder
                / f"{args.pi_weighing}_{model_name}_scatterplot_lyapunox_exp.png",
                dpi=100,
                facecolor="w",
                bbox_inches="tight",
            )
            plt.close()
            print(f"{model_name} : Lyapunov exponents: {lyapunov_exp[-1] } ")
            V = compute_CLV(qq_t, rr_t)
            # plot theta distribution
            filename = model_path / f"v_matrix_{N_test}.h5"
            with h5py.File(filename, "w") as hf:
                hf.create_dataset("v", data=V)

            # f1 = h5py.File(model_path/f'{N}_clvs.h5','r+')
            # FTCLE_lstm = np.array(f1.get('thetas_clv')).T
            # f1.close()

            # N_max = min(FTCLE_lstm.shape[1], FTCLE_targ.shape[1])
            # print(f"{N_max, FTCLE_lstm.shape, FTCLE_targ.shape}")
            # clv_angle_plot.plot_clv_pdf(FTCLE_lstm[:, :N_max], FTCLE_targ[:, :N_max], img_filepath, system='lorenz96')
