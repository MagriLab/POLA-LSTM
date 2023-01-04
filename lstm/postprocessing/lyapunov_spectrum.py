
import sys
from pathlib import Path
import time
import warnings
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Tuple
import argparse
import numpy as np
import scipy
import tensorflow as tf
sys.path.append('../../')
from lstm.utils.supress_tf_warning import tensorflow_shutup
from lstm.utils.qr_decomp import qr_factorization
from lstm.postprocessing.lyapunov_tools import step_and_jac, step_and_jac_analytical, lstm_step_comb
warnings.simplefilter(action="ignore", category=FutureWarning)
tf.keras.backend.set_floatx('float64')
tensorflow_shutup()


def compute_lyapunov_exp(test_window: np.ndarray, model, model_dict: Union[dict, argparse.Namespace],
                         N: int, dim: int, le_dim: Optional[int] = None, idx_lst: Optional[List[int]] = None,
                         save_path: Optional[Path] = None) -> np.ndarray:
    """Compute the Lyapunov exponents for a given model.

    Args:
        test_window (np.ndarray): starting point of computation
        model (Model): The model to use for the computation.
        model_dict (Dict[str, Any]): Dictionary of model parameters.
        N (int): Number of samples to use for the computation.
        dim (int): Dimension of the data.
        le_dim (int): Dimension of the Lyapunov exponent.
        idx_lst (List[int], optional): List of indices to use for the computation. If not specified, all indices will be used.
        save_path (Path, optional): If specified, the Lyapunov exponents will be saved to this path.

    Returns:
        np.ndarray: A list of Lyapunov exponents, one for each index in `idx_lst` (or all indices if `idx_lst` is not specified).
    """
    if le_dim == None:
        le_dim = dim
    if idx_lst == None:
        idx_lst = np.arange(0, dim)
    if type(model_dict) == dict:
        print('Identified Dictionary')
        n_cell = model_dict['ML_CONSTRAINTS']['N_CELLS']
        dt = model_dict['DATA']['DELTA T']  # time step
        upsampling = model_dict['DATA']['UPSAMPLING']
        window_size = model_dict['DATA']['WINDOW_SIZE']
    elif type(model_dict) == argparse.Namespace:
        model_dict = vars(model_dict)
        print('Identified argparse')
        n_cell = model_dict['n_cells']
        dt = model_dict['delta_t']  # time step
        upsampling = model_dict['upsampling']
        window_size = model_dict['window_size']
    norm_time = 1
    Ntransient = max(int(N/100), window_size+2)
    N_test = N - Ntransient
    Ttot = np.arange(int(N_test/norm_time)) * (upsampling*dt) * norm_time
    N_test_norm = int(N_test/norm_time)

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
    u_t = test_window[:, 0, :]
    h = tf.Variable(model.layers[0].get_initial_state(test_window)[0], trainable=False)
    c = tf.Variable(model.layers[0].get_initial_state(test_window)[1], trainable=False)
    pred = np.zeros(shape=(N, dim))
    pred[0, :] = u_t

    # prepare h,c and c from first window
    for i in range(1, window_size+1):
        u_t = test_window[:, i-1, :]
        u_t_eval = tf.gather(u_t, idx_lst, axis=1)
        u_t, h, c = lstm_step_comb(u_t_eval, h, c, model, window_size, i, idx_lst, dim)
        pred[i, :] = u_t

    i = window_size
    u_t_eval = tf.gather(u_t, idx_lst, axis=1)
    jacobian, u_t, h, c = step_and_jac(u_t_eval, h, c, model, window_size, i, idx_lst, dim)
    pred[i, :] = u_t
    delta = np.matmul(jacobian, delta)
    q, r = qr_factorization(delta)
    delta = q[:, :le_dim]

    # compute delta on transient
    for i in range(window_size+1, Ntransient):
        u_t_eval = tf.gather(u_t, idx_lst, axis=1)
        jacobian, u_t, h, c = step_and_jac_analytical(u_t_eval, h, c, model, window_size, i, idx_lst, dim)
        pred[i, :] = u_t
        delta = np.matmul(jacobian, delta)

        if i % norm_time == 0:
            q, r = qr_factorization(delta)
            delta = q[:, :le_dim]

    print('Finished on Transient')
    # compute lyapunov exponent based on qr decomposition
    start_time = time.time()
    for i in range(Ntransient, N):
        u_t_eval = tf.gather(u_t, idx_lst, axis=1)
        jacobian, u_t, h, c = step_and_jac_analytical(u_t_eval, h, c, model, window_size, i, idx_lst, dim)
        indx = i-Ntransient
        pred[i, :] = u_t
        delta = np.matmul(jacobian, delta)
        if i % norm_time == 0:
            q, r = qr_factorization(delta)
            delta = q[:, :le_dim]

            rr_t[:, :, indx] = r
            qq_t[:, :, indx] = q
            LE[indx] = np.abs(np.diag(r[:le_dim, :le_dim]))

            if i % 10000 == 0:
                print(f'Inside closed loop i = {i}, Time: {time.time()-start_time}')
                start_time = time.time()
                if indx != 0:
                    lyapunov_exp = np.cumsum(np.log(LE[1:indx]), axis=0) / np.tile(Ttot[1:indx], (le_dim, 1)).T
                    print(f'Lyapunov exponents: {lyapunov_exp[-1] } ')

    lyapunov_exp = np.cumsum(np.log(LE[1:]), axis=0) / np.tile(Ttot[1:], (le_dim, 1)).T
    print(f'Final Lyapunov exponents: {lyapunov_exp[-1] } ')
    if save_path != None:
        np.savetxt((save_path/f'lyapunov_exp_{N_test}.txt'), lyapunov_exp)
    return lyapunov_exp


def lyapunov_scatterplot(
        ref_lyap: np.ndarray, lyapunov_exp: np.ndarray, n_lyap: Optional[int] = None, img_filepath: Optional[Path] = None,
        name: Optional[str] = None, second_img_filepath: Optional[Path] = None) -> None:
    """Create a scatterplot of the Lyapunov exponents for a given model.

    Args:
        ref_lyap_path (Path): Path to the file containing the reference Lyapunov exponents.
        lyapunov_exp (np.ndarray): The computed Lyapunov exponents to plot.
        n_lyap (Optional[int]): Number of Lyapunov exponents to plot. If not specified, the minimum of the number of exponents in `ref_lyap_path` and `lyapunov_exp` will be used.
        img_filepath (Optional[Path]): If specified, the plot will be saved to this file.
        name (Optional[str]):Name of the model to use in the plot title and file name (if saving to a file).
        second_img_filepath (Optional[Path]): If specified, the plot will be saved to this file with the given `name`.
    """
    print('Check0')

    # If n_lyap is not specified, use the minimum of the number of exponents in the reference and the given exponents
    if n_lyap == None:
        n_lyap = min(ref_lyap.shape[1], lyapunov_exp.shape[1])

    # Set up the plot
    fullspace = np.arange(1, n_lyap+1)
    print('Check1')
    fs = 12
    plt.rcParams.update({'font.size': fs})
    plt.grid(True, c='lightgray', linestyle='--', linewidth=0.5)
    plt.ylabel(r'$\lambda_k$', fontsize=fs)
    plt.xlabel(r'$k$', fontsize=fs)
    print('Check2')
    # Plot the reference and given exponents
    plt.plot(fullspace, ref_lyap[-1, :n_lyap], 'k-s', markersize=8, label='target')
    plt.plot(fullspace, lyapunov_exp[-1, :n_lyap], 'r-o', markersize=6, label='LSTM')
    print('Check4')
    # Add a legend and save the plot if a file path is specified
    plt.legend()
    if img_filepath != None:
        plt.savefig(img_filepath/f'{lyapunov_exp.shape[0]}_scatterplot_lyapunox_exp.png',
                    dpi=100, facecolor="w", bbox_inches="tight")
    if name != None and second_img_filepath != None:
        plt.savefig(second_img_filepath/f'{name}_scatterplot_lyapunox_exp.png',
                    dpi=100, facecolor="w", bbox_inches="tight")
    plt.close()
    print(f'{name} : Lyapunov exponents: {lyapunov_exp[-1] } ')


def return_lyap_err(ref_lyap: np.ndarray, lyapunov_exp: np.ndarray) -> Tuple[float, float]:
    """Compute the error between reference and given Lyapunov exponents.


    Args:
        ref_lyap_path (Path): Path to the file containing the reference Lyapunov exponents.
        lyapunov_exp (np.ndarray): Array of Lyapunov exponents to compare with the reference.

    Returns:
        Tuple[float, float]: A tuple containing the maximum percent error and L2 error between the reference and given exponents.
    """
    # Use the minimum of the number of exponents in the reference and the given exponents
    n_lyap = min(ref_lyap.shape[1], lyapunov_exp.shape[1])
    print(f'load path, {n_lyap}')
    # Compute the maximum percent error
    max_lyap_percent_error = np.abs(ref_lyap[-1, 0] - lyapunov_exp[-1, 0])/(ref_lyap[-1, 0])

    # Compute the L2 error
    l_2_error = np.linalg.norm(ref_lyap[-1, :n_lyap] - lyapunov_exp[-1, :n_lyap])

    return max_lyap_percent_error, l_2_error
