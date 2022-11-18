import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def fit_rabi(x, y, Tr=100e-6, phi=-np.pi, freq=None):

    def func(t, A, Tr, freq, phi, B):
        return A * np.exp(-t / Tr) * np.cos(2 * np.pi * freq * t + phi) + B

    A = (np.max(y) - np.min(y)) / 2
    B = (np.max(y) + np.min(y)) / 2
    if freq is not None:
        p0 = [A, Tr, freq, phi, B]
    else:
        freq = np.fft.fftfreq(x.shape[0], x[1] - x[0])[1:(x.shape[0] // 2)]
        amp = np.abs(np.fft.fft(y))[1:(x.shape[0] // 2)]
        index, peak_height = find_peaks(amp / np.max(amp),
                                        height=0.01,
                                        distance=(x.shape[0] // 2))
        index = index[np.argsort(peak_height)[-1]]
        freq = freq[index]
        p0 = [A, Tr, freq, phi, B]
    try:
        popt, pcov = curve_fit(func,
                               x,
                               y,
                               p0=p0,
                               method='trf',
                               bounds=((max(A - 0.3,
                                            0), 10e-9, freq * 0.4, -np.pi,
                                        B - 0.3), (A + 0.3, np.inf, freq * 1.6,
                                                   np.pi, B + 0.3)))
        fitted = func(x, *popt)
        return popt, pcov, fitted
    except:
        raise ValueError('TimeRabi fitting failed!')


def fit_ramsey(x, y, T1=1.0, Tphi=1.0, Delta=0, method='default'):
    pass
