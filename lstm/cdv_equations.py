import numpy as np
import tensorflow as tf

u_star_1 = 0.95
u_star_4 = -0.76095
C = 0.1
beta = 1.25
gamma = 0.2
b = 0.5

epsilon = 16*np.sqrt(2) / (5*np.pi)


def alpha_m(m):
    numerator = 8*np.sqrt(2) * m**2 * (b**2 + m**2 - 1)
    denominator = np.pi*(4*m**2 - 1)*(b**2 + m**2)
    return numerator/denominator


def beta_m(m):
    numerator = beta * b**2
    denominator = b**2 + m**2
    return numerator/denominator


def delta_m(m):
    constant = 64*np.sqrt(2) / (15*np.pi)
    numerator = b**2 - m**2 + 1
    denominator = b**2 + m**2
    return constant*numerator/denominator


def gamma_star_m(m):
    numerator = 4*np.sqrt(2)*m*b
    denominator = np.pi*(4*m**2 - 1)
    return gamma * numerator/denominator


def gamma_m(m):
    numerator = 4*np.sqrt(2)*m**3*b
    denominator = np.pi*(4*m**2 - 1)*(b**2+m**2)
    return gamma * numerator/denominator


def cdv_system(u):
    u_1 = u[:, 0]
    u_2 = u[:, 1]
    u_3 = u[:, 2]
    u_4 = u[:, 3]
    u_5 = u[:, 4]
    u_6 = u[:, 5]

    u_dot_1 = gamma_star_m(1)*u_3 - C*(u_1 - u_star_1)
    u_dot_2 = -(alpha_m(1)*u_1 - beta_m(1))*u_3 - C*u_2 - delta_m(1)*u_4*u_6
    u_dot_3 = (alpha_m(1)*u_1 - beta_m(1))*u_2 - gamma_m(1)*u_1 - C*u_3 + delta_m(1)*u_4*u_5
    u_dot_4 = gamma_star_m(2)*u_6 - C*(u_4-u_star_4) + epsilon*(u_2*u_6 - u_3*u_5)
    u_dot_5 = - (alpha_m(2)*u_1 - beta_m(2))*u_6 - C*u_5 - delta_m(2)*u_4*u_3
    u_dot_6 = (alpha_m(2)*u_1 - beta_m(2))*u_5 - gamma_m(2)*u_4 - C*u_6 + delta_m(2)*u_4*u_2

    return np.array([u_dot_1, u_dot_2, u_dot_3, u_dot_4, u_dot_5, u_dot_6])


def cdv_system(u):
    u_1 = u[:, 0]
    u_2 = u[:, 1]
    u_3 = u[:, 2]
    u_4 = u[:, 3]
    u_5 = u[:, 4]
    u_6 = u[:, 5]

    u_dot_1 = gamma_star_m(1)*u_3 - C*(u_1 - u_star_1)
    u_dot_2 = -(alpha_m(1)*u_1 - beta_m(1))*u_3 - C*u_2 - delta_m(1)*u_4*u_6
    u_dot_3 = (alpha_m(1)*u_1 - beta_m(1))*u_2 - gamma_m(1)*u_1 - C*u_3 + delta_m(1)*u_4*u_5
    u_dot_4 = gamma_star_m(2)*u_6 - C*(u_4-u_star_4) + epsilon*(u_2*u_6 - u_3*u_5)
    u_dot_5 = - (alpha_m(2)*u_1 - beta_m(2))*u_6 - C*u_5 - delta_m(2)*u_4*u_3
    u_dot_6 = (alpha_m(2)*u_1 - beta_m(2))*u_5 - gamma_m(2)*u_4 - C*u_6 + delta_m(2)*u_4*u_2

    return u_dot_1, u_dot_2, u_dot_3, u_dot_4, u_dot_5, u_dot_6


def cdv_system_tensor(u):
    u_1 = u[:, :, 0]
    u_2 = u[:, :, 1]
    u_3 = u[:, :, 2]
    u_4 = u[:, :, 3]
    u_5 = u[:, :, 4]
    u_6 = u[:, :, 5]

    u_dot_1 = gamma_star_m(1)*u_3 - C*(u_1 - u_star_1)
    u_dot_2 = -(alpha_m(1)*u_1 - beta_m(1))*u_3 - C*u_2 - delta_m(1)*u_4*u_6
    u_dot_3 = (alpha_m(1)*u_1 - beta_m(1))*u_2 - gamma_m(1)*u_1 - C*u_3 + delta_m(1)*u_4*u_5
    u_dot_4 = gamma_star_m(2)*u_6 - C*(u_4-u_star_4) + epsilon*(u_2*u_6 - u_3*u_5)
    u_dot_5 = - (alpha_m(2)*u_1 - beta_m(2))*u_6 - C*u_5 - delta_m(2)*u_4*u_3
    u_dot_6 = (alpha_m(2)*u_1 - beta_m(2))*u_5 - gamma_m(2)*u_4 - C*u_6 + delta_m(2)*u_4*u_2

    return tf.stack([u_dot_1, u_dot_2, u_dot_3, u_dot_4, u_dot_5, u_dot_6], 2)
