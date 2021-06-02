from itertools import cycle
from typing import Optional, Sequence

import numpy as np
from scipy.fftpack import fft, ifft, ifftshift


def getFTMatrix(f_list: Sequence[float],
                numOfPoints: int,
                phase_list: Optional[Sequence[float]] = None,
                weight: Optional[np.ndarray] = None,
                sampleRate: float = 1e9) -> np.ndarray:
    """
    get a matrix for Fourier transform

    Args:
        f_list (Sequence[float]): list of frequencies
        numOfPoints (int): size of signal frame
        phase_list (Optional[Sequence[float]], optional): list of phase. Defaults to None.
        weight (Optional[np.ndarray], optional): weight or list of weight. Defaults to None.
        sampleRate (float, optional): sample rate of signal. Defaults to 1e9.

    Returns:
        numpy.ndarray: exp matrix
    
    >>> shots, numOfPoints, sampleRate = 100, 1000, 1e9
    >>> f1, f2 = -12.7e6, 32.8e6
    >>> signal = np.random.randn(shots, numOfPoints)
    >>> e = getFTMatrix([f1, f2], numOfPoints, sampleRate=sampleRate)
    >>> ret = signal @ e
    >>> ret.shape
    (100, 2)
    >>> t = np.arange(numOfPoints) / sampleRate
    >>> signal = 0.8 * np.sin(2 * np.pi * f1 * t) + 0.2 * np.cos(2 * np.pi * f2 * t)
    >>> signal @ e
    array([-0.00766509-0.79518987j,  0.19531432+0.00207068j])
    >>> spec = 2 * np.fft.fft(signal) / numOfPoints
    >>> freq = np.fft.fftfreq(numOfPoints)
    >>> e = getFTMatrix(freq, numOfPoints, sampleRate=1)
    >>> np.allclose(spec, signal @ e)
    True
    """
    e = []
    t = np.linspace(0, numOfPoints / sampleRate, numOfPoints, endpoint=False)
    if weight is None or len(weight) == 0:
        weight = np.full(numOfPoints, 2 / numOfPoints)
    if phase_list is None or len(phase_list) == 0:
        phase_list = np.zeros_like(f_list)
    if weight.ndim == 1:
        weightList = cycle(weight)
    else:
        weightList = weight
    for f, phase, weight in zip(f_list, phase_list, weightList):
        e.append(weight * np.exp(-1j * (2 * np.pi * f * t + phase)))
    return np.asarray(e).T


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


def kernel(sig_in, sig_out, sample_rate, bw=None, skip=0):
    #b, a = signal.butter(3, bw / (0.5*sample_rate), 'low')
    #sig_out = signal.filtfilt(b, a, sig_out)
    corr = fft(sig_in) / fft(sig_out)
    ker = np.real(ifftshift(ifft(corr)))
    if bw is not None and bw < 0.5 * sample_rate:
        #b, a = signal.butter(3, bw / (0.5*sample_rate), 'low')
        #ker = signal.filtfilt(b, a, ker)
        k = np.exp(-0.5 * np.linspace(-3.0, 3.0, int(2 * sample_rate / bw))**2)
        ker = np.convolve(ker, k / k.sum(), mode='same')
    return ker[int(skip):len(ker) - int(skip)]


def predistort(sig, ker):
    return np.convolve(sig, ker, mode='same')


if __name__ == "__main__":
    import doctest
    doctest.testmod()
