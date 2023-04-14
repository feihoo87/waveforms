import functools

import numpy as np
from scipy.signal import find_peaks

from ._fit import fit


def func_rabi(t, A, Tr, freq, phi, B):
    return A * np.exp(-t / Tr) * np.cos(2 * np.pi * freq * t + phi) + B


def guess_rabi(t, y, static_params, freq):
    if 'B' in static_params:
        B = static_params['B']
    else:
        B = np.median(y)

    if freq is None:
        freq = np.fft.fftfreq(t.shape[0], t[1] - t[0])[1:(t.shape[0] // 3)]
        amp = np.abs(np.fft.fft(y))[1:(t.shape[0] // 3)]
        index, peak_height = find_peaks(amp / np.max(amp),
                                        height=0.01,
                                        distance=(t.shape[0] // 3))
        index = index[np.argsort(peak_height)[-1]]
        freq = freq[index]
        spec = np.max(amp) * 2 / len(t)
    else:
        spec = 0.0

    A = np.median(np.abs(y - B)) * np.sqrt(2)

    if spec < 0.2 * A:
        freq = 0.25 / (t[-1] - t[0])

    freq = max(1.0, freq)
    T = 1 / freq
    N = len(t[t <= T])

    A0 = np.median(np.abs(y[:N] - B)) * np.sqrt(2)
    A1 = np.median(np.abs(y[-N:] - B)) * np.sqrt(2)
    att = -np.log(min(1, A1 / A0))
    if att < 1e-16:
        Tr = 1.0
    elif 2 * T > (t[-1] - t[0]):
        Tr = (t[-1] - t[0]) / att
    else:
        Tr = (t[-1] - t[0] - T) / att

    return {'A': A0, 'B': B, 'Tr': Tr, 'freq': freq, 'phi': np.pi}


def fit_rabi(t, y, freq=None, static_params=None):
    """
    Fit Rabi oscillation data to a cosine function.

    Args:
        t (np.ndarray): time array
        y (np.ndarray): data array
        Tr (float): Rabi oscillation decay time
        phi (float): phase
        freq (float): Rabi frequency
        static_params (dict): static parameters for fitting

    Returns:
        popt (np.ndarray): optimized parameters
        pcov (np.ndarray): covariance matrix
        fitted (np.ndarray): fitted data
    """

    return fit(func_rabi,
               t,
               y,
               guess=functools.partial(guess_rabi, freq=freq),
               static_params=static_params)
