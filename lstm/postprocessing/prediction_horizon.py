import numpy as np


def compute_lyapunov_time_arr(time_vector, c_lyapunov=0.90566,  window_size=100):
    t_lyapunov = 1 / c_lyapunov
    lyapunov_time = (time_vector[window_size:] -
                     time_vector[window_size]) / t_lyapunov
    return lyapunov_time

def pred_nominator(sol_array):
    """returns an array where the i-th entry corresponds to the l2 norm of np_array[:i] """
    norm_for_idx = [np.linalg.norm(sol_array[i, :]) for i in range(0, sol_array.shape[0])]
    return np.array(norm_for_idx)

def pred_denominator(sol_array):
    """returns an array where the i-th entry corresponds to the l2 norm of np_array[:i] """
    norm_for_idx = [np.sqrt(np.mean(np.sum(sol_array[i, :]**2))) for i in range(0, sol_array.shape[0])]
    return np.array(norm_for_idx)


def compute_pred_horizon_idx(pred, num_sol, threshold=0.2, window_size=100):
    """the idx of the prediction horizon based on a threshold """
    nominator = pred_nominator(num_sol[window_size: window_size + len(pred)] - pred)
    denominator = pred_denominator(num_sol[window_size: window_size + len(pred)])

    pred_idx = len(pred)
    for i in range(1, len(pred)):
        if nominator[i] > threshold*denominator[i]:
            return i-1

    return pred_idx

def predict_horizon_layapunov_time(pred, num_sol, time_vector, window_size=100, threshold=0.2):
    pred_idx = compute_pred_horizon_idx(pred, num_sol, window_size=100, threshold=threshold)
    pred_lt = compute_lyapunov_time_arr(time_vector, window_size=window_size)[pred_idx]
    return pred_lt