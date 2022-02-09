import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting
import time  # pause plot


def solution_from_np(filename):
    lorenz_sol = np.genfromtxt(filename, delimiter=",")
    time = lorenz_sol[0, :]
    x = lorenz_sol[1, :]
    y = lorenz_sol[2, :]
    z = lorenz_sol[3, :]
    return time, x, y, z


if __name__ == "main":
    t, x, y, z = solution_from_np("CSV/lorenz_init1.csv")
    t, x_2, y_2, z_2 = solution_from_np("CSV/lorenz_init2.csv")
    n = len(x)

    # Plot the Lorenz attractor using a Mat
    # plotlib 3D projection.
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    # ax.set_facecolor('k')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Make the line multi-coloured by plotting it in segments of length s which
    # change in colour across the whole time series.
    s = 100
    cmap = plt.cm.winter  # winter
    cmap_2 = plt.cm.autumn  # autumn
    for i in range(0, n - s, s):
        ax.plot(
            x[i : i + s + 1],
            y[i : i + s + 1],
            z[i : i + s + 1],
            color=cmap(i / n),
            alpha=0.4,
        )
        ax.plot(
            x_2[i : i + s + 1],
            y_2[i : i + s + 1],
            z_2[i : i + s + 1],
            color=cmap_2(i / n),
            alpha=0.4,
        )
        plt.title("Solution of Lorenz system")
        plt.show()
        plt.pause(0.1)  # plot both curves incrementally
