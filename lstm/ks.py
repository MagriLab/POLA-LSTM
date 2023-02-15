import numpy as np
from numpy.fft import fft, ifft
import tensorflow as tf
np.random.seed(0)


def ks_time_step(u, d=60, M=16, h=0.25):
    # for a single initial condition
    v = fft(u)
    N = u.shape[0]
    k = np.transpose(np.conj(np.concatenate((np.arange(0, N/2), np.array([0]), np.arange(-N/2+1, 0))))) * (2*np.pi/d)
    L = k**2 - k**4
    E = np.exp(h*L)
    E_2 = np.exp(h*L/2)
    r = np.exp(1j*np.pi*(np.arange(1, M+1)-0.5) / M)
    LR = h*np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], N, axis=0)
    Q = h*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=1))
    f1 = h*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1))
    f2 = h*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1))
    f3 = h*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1))
    g = -0.5j*k

    Nv = g * fft(u**2)
    a = E_2*v + Q*Nv
    Na = g * fft(np.real(ifft(a))**2)
    b = E_2*v + Q*Na
    Nb = g * fft(np.real(ifft(b))**2)
    c = E_2*a + Q*(2*Nb-Nv)
    Nc = g * fft(np.real(ifft(c))**2)
    diff_v = Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
    v = E*v + diff_v

    return np.real(ifft(v))


def ks_time_step_array(u, d=60, M=16, h=0.25):
    N = u.shape[1]  # (window, dim)
    v = fft(u).T
    k = np.transpose(np.conj(np.concatenate((np.arange(0, N/2), np.array([0]), np.arange(-N/2+1, 0))))) * (2*np.pi/d)
    L = k**2 - k**4
    E = np.exp(h*L)
    E_2 = np.exp(h*L/2)
    r = np.exp(1j*np.pi*(np.arange(1, M+1)-0.5) / M)
    LR = h*np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], N, axis=0)

    Q = h*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=1))
    f1 = h*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1))
    f2 = h*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1))
    f3 = h*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1))
    g = -0.5j*k

    Nv = np.einsum('ji,i->i j', fft(u**2), g)  # (window, dim) (dim,) (dim, window)
    a = np.einsum('i j,i->i j', v, E_2) + np.einsum('i j,i->i j', Nv, Q)  # (dim, window) (dim,) (dim, window)
    Na = np.einsum('j i,i->i j', fft(np.real(ifft(a.T))**2), g)  # (window, dim) (dim,) (dim, window)
    b = np.einsum('i j,i->i j', v, E_2) + np.einsum('i j,i->i j', Na, Q)  # (dim, window) (dim,) (dim, window)
    Nb = np.einsum('j i,i->i j', fft(np.real(ifft(b.T))**2), g)  # (window, dim) (dim,) (dim, window)
    c = np.einsum('i j,i->i j', a, E_2) + np.einsum('i j,i->i j', 2*Nb-Nv, Q)  # (dim, window) (dim,) (dim, window)
    Nc = np.einsum('j i,i->i j', fft(np.real(ifft(c.T))**2), g)  # (window, dim) (dim,) (dim, window)
    diff_v = np.einsum('i j,i->i j', Nv, f1) + np.einsum('i j,i->i j', 2*(Na+Nb),
                                                         f2) + np.einsum('i j,i->i j', Nc, f3)  # (dim, window)
    v = np.einsum('i j,i->i j', v, E) + diff_v  # (dim, window)
    return np.real(ifft(v.T))  # (window, dim)


def ks_time_step_batch(u, d=60, M=16, h=0.25):
    # u before (time_steps, dim) now (batch, window, dim)
    N = u.shape[2]
    k = np.transpose(np.conj(np.concatenate((np.arange(0, N/2), np.array([0]), np.arange(-N/2+1, 0))))) * (2*np.pi/d)
    L = k**2 - k**4
    E = np.exp(h*L)
    E_2 = np.exp(h*L/2)
    M =16
    r = np.exp(1j*np.pi*(np.arange(1, M+1)-0.5) / M)
    LR = h * np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], N, axis=0)
    Q = h*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=1))
    f1 = h*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1))
    f2 = h*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1))
    f3 = h*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1))
    g = -0.5j*k

    u = tf.complex(u, tf.zeros(u.shape, dtype=u.dtype))
    v = tf.transpose(tf.signal.fft3d(u), (0, 2, 1))  # (batch,dim, window)
    Nv = tf.einsum('b j i,i->b i j', tf.signal.fft3d(tf.square(u)), g)
    a = tf.einsum('b i j,i->b i j', v, E_2) + tf.einsum('b i j,i->b i j',
                                                        Nv, Q)  # (dim, window) (dim,) (batch, dim, window)
    Na = tf.einsum('b j i,i-> b i j', tf.signal.fft3d(tf.signal.ifft3d(tf.transpose(a, (0, 2, 1)))**2),
                   g)  # (window, dim) (dim,) (batch, dim, window)
    b = tf.einsum('b i j,i->b i j', v, E_2) + tf.einsum('b i j,i->b i j',
                                                        Na, Q)  # (dim, window) (dim,) (batch, dim, window)
    Nb = tf.einsum('b j i,i->b i j', tf.signal.fft3d(tf.signal.ifft3d(tf.transpose(b, (0, 2, 1)))**2),
                   g)  # (window, dim) (dim,), (batch, dim, window)
    c = tf.einsum('b i j,i->b i j', a, E_2) + tf.einsum('b i j,i->b i j',
                                                        2*Nb-Nv, Q)  # (dim, window) (dim,)(batch, dim, window)
    Nc = tf.einsum('b j i,i->b i j', tf.signal.fft3d(tf.signal.ifft3d(tf.transpose(c, (0, 2, 1)))**2),
                   g)  # (window, dim) (dim,)(batch, dim, window)
    diff_v = tf.einsum('b i j,i->b i j', Nv, f1) + tf.einsum('b i j,i->b i j', 2*(Na+Nb),
                                                             f2) + tf.einsum('b i j,i->b i j', Nc, f3)  # (batch, dim, window)
    v = tf.einsum('b i j,i-> b i j', v, E) + diff_v  # (batch, dim, window)

    return tf.math.real(tf.signal.ifft3d(tf.transpose(v, (0, 2, 1))))
