import functools
import inspect

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from waveforms.math.fit.qubit_dynamics import *


def fit(func,
        xdata,
        ydata,
        p0=None,
        sigma=None,
        bounds=(-np.inf, np.inf),
        guess=None,
        static_params=None):
    """
    Fit data to a function.

    Args:
        func (function): fitting function
        xdata (np.ndarray): x data
        ydata (np.ndarray): y data
        p0 (np.ndarray | dict): initial parameters
        sigma (np.ndarray): standard deviation of ydata
        bounds (tuple): lower and upper bounds of parameters
        guess (function): function to guess initial parameters
        static_params (dict): static parameters for fitting

    Returns:
        arg_names (list): parameter names
        popt (np.ndarray): optimized parameters
        pcov (np.ndarray): covariance matrix
        fitted (np.ndarray): fitted data
    """
    sig = inspect.signature(func)

    if static_params is None:
        static_params = {}

    arg_names = []
    for i, (arg_name, param) in enumerate(sig.parameters.items()):
        if i < 1:
            continue
        if arg_name not in static_params:
            arg_names.append(arg_name)

    def func_wrapper(x, *args):
        params = dict(zip(arg_names, args)) | static_params
        return func(x, **params)

    if guess is None and p0 is None:
        raise ValueError('Initial parameters are not specified!')

    if guess is not None:
        p0 = guess(xdata, ydata, static_params)

    if isinstance(p0, dict):
        p0 = [p0[arg_name] for arg_name in arg_names]

    try:
        popt, pcov = curve_fit(func_wrapper,
                               xdata,
                               ydata,
                               p0=p0,
                               sigma=sigma,
                               absolute_sigma=True,
                               method='trf',
                               bounds=bounds)
        fitted = func_wrapper(xdata, *popt)
        return arg_names, popt, pcov, fitted
    except:
        raise ValueError('Fitting failed!')


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
