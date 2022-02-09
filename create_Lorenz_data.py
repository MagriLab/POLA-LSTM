import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting
import time  # pause plot
import csv


def lorenz(t, X, sigma=10, beta=2.667, rho=28):
    """The Lorenz equations."""
    u, v, w = X
    up = -sigma * (u - v)
    vp = rho * u - v - u * w
    wp = -beta * w + u * v
    return up, vp, wp


def standardize_dataset(data):
    return data / (np.max(np.abs(data)) - np.min(np.abs(data)))


def save_data(filename, t, x, y, z):
    with open(filename, "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(t)
        writer.writerow(standardize_dataset(x))
        writer.writerow(standardize_dataset(y))
        writer.writerow(standardize_dataset(z))


if __name__ == "__main__":
    # Lorenz paramters and initial conditions.
    sigma, beta, rho = 10, 2.667, 28
    print("Sigma, beta, rho of Lorenz equation: ", sigma, beta, rho)
    u0, v0, w0 = 0, 1, 1
    print("Initial conditons: ", u0, v0, w0)
    # Maximum time point and total number of time points.
    tmax, n = 100, 10000
    print("Max T: ", tmax)
    print("number of time points: ", n)
    # Integrate the Lorenz equations.
    sol = solve_ivp(
        lorenz, (0, tmax), (u0, v0, w0), args=(sigma, beta, rho), dense_output=True
    )  # This is RK45

    # Interpolate solution onto the time grid, t.
    t = np.linspace(0, tmax, n)
    x, y, z = sol.sol(t)
    print("Successfully solved the Lorenz equation using RK45")
    save_data("CSV/Lorenz_stand.csv", t, x, y, z)
