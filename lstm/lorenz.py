import numpy as np
import tensorflow as tf

def fixpoints(total_points=5000, beta=2.667, rho=28, sigma=10, unnorm=False):
    if unnorm == True:
        x_max = 1
        y_max = 1
        z_max = 1
    else:
        x_max = 19.6195
        y_max = 27.3171
        z_max = 48.0537

    x_fix = np.sqrt(beta*(rho-1))
    y_fix = np.sqrt(beta*(rho-1))
    z_fix = rho-1
    return x_fix/x_max, y_fix/y_max, z_fix/z_max

x_max = 19.6195
y_max = 27.3171
z_max = 48.0537


def norm_lorenz(pred, sigma=10, beta=2.667, rho=28, norm=True):
    if norm:
        x_max = 19.6195
        y_max = 27.3171
        z_max = 48.0537
    else:
        x_max = 1
        y_max = 1
        z_max = 1

    x = pred[:, 0]
    y = pred[:, 1]
    z = pred[:, 2]

    x_t = -sigma * (x/y_max - y/x_max)
    y_t = rho * x/y_max/z_max - y/x_max/z_max - x * z/y_max
    z_t = -beta * z/x_max/y_max + x * y/z_max
    return x_t, y_t, z_t

def norm_lorenz_batch(pred, sigma=10, beta=2.667, rho=28, norm=True):
    if norm:
        x_max = 19.6195
        y_max = 27.3171
        z_max = 48.0537
    else:
        x_max = 1
        y_max = 1
        z_max = 1

    x = pred[:, :, 0]
    y = pred[:, :, 1]
    z = pred[:, :, 2]

    x_t = -sigma * (x/y_max - y/x_max)
    y_t = rho * x/y_max/z_max - y/x_max/z_max - x * z/y_max
    z_t = -beta * z/x_max/y_max + x * y/z_max
    return x_t, y_t, z_t
