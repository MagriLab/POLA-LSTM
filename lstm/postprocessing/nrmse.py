import numpy as np

# implementation based on Backpropagation algorithms and RC in RNNs for 
# the forecasting of complex spatiotemporal dynamics by Vlachas (2020)


def nrmse(pred, df_test, window_size=100, n_length=None):
    """ normalized root mean square error """
    if n_length == None:
        n_length = len(pred)
    std = np.std(df_test[:, window_size:window_size+n_length])
    diff = pred[:n_length, :] - df_test[:, window_size:window_size+n_length].T
    return np.sqrt(np.mean(diff**2/std))


def vpt(pred, df_test, threshold, window_size=100):
    """valid prediction time"""
    for i in range(1, len(pred)):
        nrmse_i = nrmse(pred, df_test, window_size=window_size, n_length=i)
        if nrmse_i > threshold:
            return i-1
    return len(pred)
