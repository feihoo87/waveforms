import numpy as np
import scipy.constants as const


def Svv(f, T, Z=lambda f: 50 * np.ones_like(f)):
    """
    power spectral density of the series noise voltage

    Johnson–Nyquist noise
    https://en.wikipedia.org/wiki/Johnson%E2%80%93Nyquist_noise
    
    f : list of frequency
    T : temperature
    Z : frequency-dependent complex electrical impedance
        (default 50 Ohm)
    """
    from scipy.constants import h, k

    if callable(Z):
        R = np.real(Z(f))
    else:
        R = np.real(Z)
    x = h * f / (k * T)
    x, R, f, T = np.broadcast_arrays(x, R, f, T)
    ret = np.zeros_like(x)
    mask1 = x < 37
    mask2 = x >= 37
    ret[mask1] = 4 * k * T[mask1] * R[mask1] * x[mask1] / (np.exp(x[mask1]) -
                                                           1)
    ret[mask2] = 4 * h * f[mask2] * R[mask2] * np.exp(-x[mask2])
    return ret


def Sii(f, T, Z=lambda f: 50 * np.ones_like(f)):
    """
    power spectral density of the series noise voltage

    Johnson–Nyquist noise
    https://en.wikipedia.org/wiki/Johnson%E2%80%93Nyquist_noise
    
    f : list of frequency
    T : temperature
    Z : frequency-dependent complex electrical impedance
        (default 50 Ohm)
    """
    from scipy.constants import h, k

    if callable(Z):
        Y = np.real(1 / Z(f))
    else:
        Y = np.real(1 / Z)
    x = h * f / (k * T)
    x, Y, f, T = np.broadcast_arrays(x, Y, f, T)
    ret = np.zeros_like(x)
    mask1 = x < 37
    mask2 = x >= 37
    ret[mask1] = 4 * k * T[mask1] * Y[mask1] * x[mask1] / (np.exp(x[mask1]) -
                                                           1)
    ret[mask2] = 4 * h * f[mask2] * Y[mask2] * np.exp(-x[mask2])
    return ret


def atts(f, atts=[], input=None):
    """
    power spectral density at MXC
    
    f : list of frequency
    atts: list of tuples (temperature, attenuator)    
    """
    if input is not None:
        spec = input
    else:
        spec = 0.5 * Svv(f, 300)
    for T, att in atts:
        A = 10**(-att / 10)
        spec = spec / A + 0.5 * Svv(f, T) * (A - 1) / A
    return spec


def atts_and_heat(f, atts=[], input=None):
    """
    power spectral density at MXC
    
    f : list of frequency
    atts: list of tuples (temperature, attenuator)    
    """
    heat = np.zeros_like(f)
    if input is not None:
        spec = input
    else:
        spec = 0.5 * Svv(f, 300)
    for T, att in atts:
        A = 10**(-att / 10)
        heat += 300 / T * (A - 1) / A * spec
        spec = spec / A + 0.5 * Svv(f, T) * (A - 1) / A
    return spec, heat


def thermal_excitation(T, f01, *levels):
    """
    p1 / p0 = exp(-beta*E01)
    p0 + p1 = 1
    p1 = tanh(-beta * E01 / 2) / 2 + 1 / 2
    """
    if len(levels) == 0:
        return 0.5 * np.tanh(-0.5 * const.h * f01 / const.k / T) + 0.5
    else:
        levels = np.hstack([[f01], levels])
        pp = np.exp(-const.h * levels / const.k / T)
        p0 = 1 / (np.sum(pp) + 1)
        return pp * p0


def Z_in(w, ZL, l, Z0=50, v=1e8):
    """Impedance of the transmission line
    """
    a = 1j * np.tan(w * l / v)
    #return Z0*(np.tanh(1j*w*l/v)+np.arctanh(ZL/Z0))
    return Z0 * (ZL + Z0 * a) / (Z0 + ZL * a)


def S21(w, l, ZL, Z0=50, v=1e8):
    """S21 of the transmission line
    """
    z = Z_in(w, ZL, l, Z0, v) / Z0
    phi = w * l / v
    #return (1+Z0/z)*np.exp(-1j*w*l/v)/2+(1-Z0/z)*np.exp(1j*w*l/v)/2
    return np.cos(phi) - 1j / z * np.sin(phi)


def lorentzianAmp(x, x0, gamma):
    """lorentzian peak"""
    return 1 / (1 + ((x - x0) / gamma)**2)


def lorentzian(x, x0, gamma):
    """complex lorentzian peak
    
    lorentzianAmp(x, x0, gamma) = lorentzian(x, x0, gamma) * conj(lorentzian(x, x0, gamma))
    """
    return 1 / (1 + 1j * (x - x0) / gamma)


def gaussian(x, x0, sigma):
    """gaussian peak"""
    return np.exp(-0.5 * ((x - x0) / sigma)**2)


def lorentzianGaussian(x, x0, gamma, sigma):
    """complex lorentzian peak
    """
    if gamma == 0:
        return gaussian(x, x0, sigma)
    elif sigma == 0:
        return lorentzian(x, x0, gamma)
    else:
        return np.convolve(lorentzian(x, x0, gamma),
                           gaussian(x, x0, sigma),
                           mode='same')


def peaks(x, peaks, background=0):
    """
    peaks: list of (center, width, amp, shape)
           shape should be either 'gaussian' or 'lorentzian'
    background: a float, complex or ndarray with the same shape of `x`
    """
    ret = np.zeros_like(x)
    for center, width, amp, shape in peaks:
        if shape == 'gaussian':
            ret += amp * gaussian(x, center, width)
        elif shape == 'lorentzian':
            ret += amp * lorentzian(x, center, width)
        else:
            ret += amp * lorentzian(x, center, width)

    return ret + background


def complexPeaks(x, peaks, background=0):
    """
    peaks: list of (center, width, amp)
    background: a float, complex or ndarray with the same shape of `x`
    """
    ret = np.zeros_like(x, dtype=np.complex)
    for x0, gamma, A, *_ in peaks:
        ret += A * lorentzian(x, x0, gamma)
    return ret + background


def decay(t, tau):
    """
    exponential decay
    """
    a = -(1 / np.asarray(tau))**(np.arange(len(tau)) + 1)
    a = np.hstack([a[::-1], [0]])
    return np.exp(np.poly1d(a)(t))


def oscillation(t, spec=((1, 1), ), amplitude=1, offset=0):
    """
    oscillation
    """
    ret = np.zeros_like(t, dtype=np.complex)
    for A, f in spec:
        ret += A * np.exp(2j * np.pi * f * t)
    return amplitude * np.real(ret) + offset
