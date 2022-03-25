import numpy as np
import tensorflow as tf

# def max_from_norm(total_points = 100000):
#     global x_max, y_max, z_max
#     if total_points == 100000:
#         x_max = 19.62036351364186
#         y_max = 27.31708182056948
#         z_max = 48.05263683702385
#     elif total_points == 10000:
# x_max = 19.61996107472211
# y_max = 27.317071267968995
# z_max = 48.05315303164703
x_max = 19.619508366918392
y_max = 27.317051197038307
z_max = 48.05371246231375

def lorenz(pred, sigma=10, beta=2.667, rho=28):
    """ use chaotic Lorenz system to generate right hand side

    Args:
        pred (3D Tensor): network prediction
        sigma (int):  Defaults to 10.
        beta (float): Defaults to 2.667.
        rho (int): Defaults to 28.

    Returns:
        right hand side
    """
    x = pred[:, 0]
    y = pred[:, 1]
    z = pred[:, 2]
    x_t = -sigma * (x - y)
    y_t = rho * x - y - x * z
    z_t = -beta * z + x * y
    return x_t, y_t, z_t


def norm_lorenz(pred, sigma=10, beta=2.667, rho=28):

    x = pred[:, 0]
    y = pred[:, 1]
    z = pred[:, 2]

    x_t = -sigma * (x/y_max - y/x_max)
    y_t = rho * x/y_max/z_max - y/x_max/z_max - x * z/y_max
    z_t = -beta * z/x_max/y_max + x * y/z_max
    return x_t, y_t, z_t
