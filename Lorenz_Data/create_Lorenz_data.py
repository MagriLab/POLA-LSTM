import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting
import time  # pause plot
import csv
import math
from torch import zero_


def lorenz(t, X, sigma=10, beta=2.667, rho=28):
    """The Lorenz equations."""
    u, v, w = X
    up = -sigma * (u - v)
    vp = rho * u - v - u * w
    wp = -beta * w + u * v
    return up, vp, wp


def standardize_dataset(data):
    return data / (np.max(np.abs(data)) - np.min(np.abs(data)))


def normalize_dataset(data):
    print("Maximum: ", np.max(np.abs(data)))
    return data / np.max(np.abs(data))


def remove_transient_phase(t_trans, t, x, y, z):
    delta_t = t[1] - t[0]
    idx_end_trans = math.ceil(t_trans / delta_t)
    return t[idx_end_trans:], x[idx_end_trans:], y[idx_end_trans:], z[idx_end_trans:]


def save_data(filename, t, x, y, z):
    with open(filename, "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(t)
        writer.writerow(normalize_dataset(x))
        writer.writerow(normalize_dataset(y))
        writer.writerow(normalize_dataset(z))


if __name__ == "__main__":
    # Lorenz paramters and initial conditions.
    sigma, beta, rho = 10, 2.667, 28
    print("Sigma, beta, rho of Lorenz equation: ", sigma, beta, rho)
    u0, v0, w0 = 0, 1, 1
    print("Initial conditons: ", u0, v0, w0)
    # Maximum time point and total number of time points.
    tmax, n = 100, 120000  # delta t = 0.001
    print("Max T: ", tmax, " Delta t: ", tmax / n)
    print("number of time points: ", n)
    # Integrate the Lorenz equations.
    sol = solve_ivp(
        lorenz, (0, tmax), (u0, v0, w0), args=(sigma, beta, rho), dense_output=True
    )  # This is RK45

    # Interpolate solution onto the time grid, t.
    t = np.linspace(0, tmax, n)
    x, y, z = sol.sol(t)

    print("Maximum: ", np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z)))
    print("Successfully solved the Lorenz equation using RK45")
    t, x, y, z = remove_transient_phase(20, t, x, y, z)
    print("Timesteps after Transient Cut off:", len(t))
    save_data("CSV/Lorenz_trans_0001_norm_100000.csv", t, x, y, z)
    # x_der, y_der, z_der = lorenz(t, (x, y, z))
    # , y_der.shape, z_der.shape)
    # save_data("CSV/Lorenz_norm_der_0005.csv", t, x_der, y_der, z_der)
