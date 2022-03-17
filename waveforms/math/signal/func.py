import numpy as np
import scipy.constants as const


def Svv(f, T, Z=lambda f: 50 * np.ones_like(f)):
    """
    power spectral density of the series noise voltage
    
    f : list of frequency
    T : temperature
    Z : frequency-dependent complex electrical impedance
        (default 50 Ohm)
    """
    from scipy.constants import h, k
    eta = h * f / (k * T) / (np.exp(h * f / (k * T)) - 1)
    return 4 * k * T * np.real(Z(f)) * eta


def atts(f, atts=[], input=None):
    """
    power spectral density at MXC
    
    f : list of frequency
    atts: list of tuples (temperature, attenuator)    
    """
    if input is not None:
        spec = input
    else:
        spec = Svv(f, 300)
    for T, att in atts:
        A = 10**(-att / 10)
        spec = spec / A + Svv(f, T) * (A - 1) / A
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
        spec = Svv(f, 300)
    for T, att in atts:
        A = 10**(-att / 10)
        heat += 300 / T * (A - 1) / A * spec
        spec = spec / A + Svv(f, T) * (A - 1) / A
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
