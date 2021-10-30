from itertools import repeat
from typing import Optional, Sequence

import numpy as np
from scipy.fftpack import fft, fftfreq, ifft, ifftshift
from scipy.signal import fftconvolve, lfilter


def getFTMatrix(fList: Sequence[float],
                numOfPoints: int,
                phaseList: Optional[Sequence[float]] = None,
                weight: Optional[np.ndarray] = None,
                sampleRate: float = 1e9) -> np.ndarray:
    """
    get a matrix for Fourier transform

    Args:
        fList (Sequence[float]): list of frequencies
        numOfPoints (int): size of signal frame
        phaseList (Optional[Sequence[float]], optional): list of phase. Defaults to None.
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
    if phaseList is None or len(phaseList) == 0:
        phaseList = np.zeros_like(fList)
    if weight.ndim == 1:
        weightList = repeat(weight)
    else:
        weightList = weight
    for f, phase, weight in zip(fList, phaseList, weightList):
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


def extractKernel(sig_in, sig_out, sample_rate, bw=None, skip=0):
    corr = fft(sig_in) / fft(sig_out)
    ker = np.real(ifftshift(ifft(corr)))
    if bw is not None and bw < 0.5 * sample_rate:
        k = np.exp(-0.5 * np.linspace(-3.0, 3.0, int(2 * sample_rate / bw))**2)
        ker = np.convolve(ker, k / k.sum(), mode='same')
    return ker[int(skip):len(ker) - int(skip)]


def zDistortKernel(dt: float, params: Sequence[tuple]) -> np.ndarray:
    t = 3 * np.asarray(params)[:, 0].max()
    omega = 2 * np.pi * fftfreq(int(t / dt) + 1, dt)

    H = 1
    for tau, A in params:
        H += (1j * A * omega * tau) / (1j * omega * tau + 1)

    ker = ifftshift(ifft(1 / H)).real
    return ker


def exp_decay_filter(amp, tau, sample_rate):
    alpha = 1 - np.exp(-1 / (sample_rate * tau * (1 + amp)))

    if amp >= 0:
        k = amp / (1 + amp - alpha)
        a = [(1 - k + k * alpha), -(1 - k) * (1 - alpha)]
    else:
        k = -amp / (1 + amp) / (1 - alpha)
        a = [(1 + k - k * alpha), -(1 + k) * (1 - alpha)]

    b = [1, -(1 - alpha)]

    return b, a


def reflection_filter(f, A, tau):
    return (1 - A) / (1 - A * np.exp(-2j * np.pi * f * tau))


def reflection(sig, A, tau, sample_rate):
    freq = np.fft.fftfreq(len(sig), 1 / sample_rate)
    return np.fft.ifft(np.fft.fft(sig) * reflection_filter(freq, A, tau)).real


def correct_reflection(sig, A, tau, sample_rate, fft=True):
    if fft:
        freq = np.fft.fftfreq(len(sig), 1 / sample_rate)
        return np.fft.ifft(np.fft.fft(sig) /
                           reflection_filter(freq, A, tau)).real
    else:
        return 1 / (1 - A) * sig - A / (1 - A) * shift(sig, tau,
                                                       1 / sample_rate)


def predistort(sig: np.ndarray,
               filters: list = None,
               ker: np.ndarray = None) -> np.ndarray:
    if filters is None:
        filters = []
    for b, a in filters:
        sig = lfilter(b, a, sig)

    if ker is None:
        return sig

    size = len(sig)
    sig = np.hstack((np.zeros_like(sig), sig, np.zeros_like(sig)))
    start = size + len(ker) // 2
    stop = start + size
    return fftconvolve(sig, ker, mode='full')[start:stop]


def shift(signal: np.ndarray, delay: float, dt: float) -> np.ndarray:
    """
    delay a signal

    Args:
        signal (np.ndarray): input signal
        delay (float): delayed time
        dt (float): time step of signal samples

    Returns:
        np.ndarray: delayed signal
    """
    points = int(delay // dt)
    delta = delay / dt - points

    if delta > 0:
        ker = np.array([0, 1 - delta, delta])
        signal = np.convolve(signal, ker, mode='same')

    if points == 0:
        return signal

    ret = np.zeros_like(signal)
    if points < 0:
        ret[:points] = signal[-points:]
    else:
        ret[points:] = signal[:-points]
    return ret


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


if __name__ == "__main__":
    import doctest
    doctest.testmod()
