from itertools import repeat
from types import MappingProxyType
from typing import Optional, Sequence, cast

import numpy as np
import scipy.sparse as sp


def freeze(x):
    """Freeze a mutable object.
    """
    if isinstance(x, (int, float, complex, str, bytes, type(None))):
        pass
    elif isinstance(x, (list, tuple)):
        return tuple([freeze(y) for y in x])
    elif isinstance(x, dict):
        return MappingProxyType({k: freeze(v) for k, v in x.items()})
    elif isinstance(x, set):
        return frozenset([freeze(y) for y in x])
    elif isinstance(x, (np.ndarray, np.matrix)):
        x.flags.writeable = False
    elif isinstance(x, sp.spmatrix):
        cast(np.ndarray, getattr(x, 'data')).flags.writeable = False
        if getattr(x, 'format') in {'csr', 'csc', 'bsr'}:
            cast(np.ndarray, getattr(x, 'indices')).flags.writeable = False
            cast(np.ndarray, getattr(x, 'indptr')).flags.writeable = False
        elif getattr(x, 'format') == 'coo':
            cast(np.ndarray, getattr(x, 'row')).flags.writeable = False
            cast(np.ndarray, getattr(x, 'col')).flags.writeable = False
    elif isinstance(x, bytearray):
        x = bytes(x)
    return x


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
        phase_list = np.zeros_like(fList)
    else:
        phase_list = phaseList
    if weight.ndim == 1:
        weight_list = repeat(weight)
    else:
        weight_list = weight
    for f, phase, weight in zip(fList, phase_list, weight_list):
        e.append(weight * np.exp(-1j * (2 * np.pi * f * t + phase)))
    return np.asarray(e).T


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
