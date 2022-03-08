from itertools import repeat
from typing import Optional, Sequence

import numpy as np
from scipy.fftpack import fft, fftfreq, ifft, ifftshift
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve, lfilter


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
    """
    exp decay filter

                    A
    H(w) = --------------------
            1 - 1j / (w * tau)

    Args:
        amp (float): amplitude of the filter
        tau (float): decay time
        sample_rate (float): sampling rate
    """

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


def distort(points, params, sample_rate):
    filters = []
    for amp, tau in np.asarray(params).reshape(-1, 2):
        b, a = exp_decay_filter(amp, tau, sample_rate)
        filters.append((a, b))
    return predistort(points, filters)


def phase_curve(t, params, df_dphi, pulse_width, start, wav, sample_rate):
    lim = max(np.max(np.abs(t)), 20e-6)
    num = round(2 * lim * sample_rate)
    tlist = np.arange(num) / sample_rate - lim
    points = wav(tlist)

    pulse_points = round(pulse_width * sample_rate)
    start_points = round((start + pulse_width) * sample_rate) - 1

    ker = np.hstack(
        [np.ones(pulse_points) / sample_rate,
         np.zeros(start_points)])

    points = np.convolve(2 * np.pi * df_dphi *
                         distort(points, params, sample_rate),
                         ker,
                         mode='same')
    return np.interp(t, tlist, points)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from waveforms import square

    data = np.load('Z_distortion.npz')

    x = data['time'] * 1e-6
    y = data['phase']
    df_dphi = 4343.313e6

    sample_rate = 2e9
    wav = 0.1 * (square(2e-6) << 1e-6)

    def f(t, *params):
        return phase_curve(t, params, df_dphi, 10e-9, 25e-9)

    params = [-0.03, 0.1e-6, 0.02, 0.3e-6]
    popt, pcov = curve_fit(f, x, y, p0=params)

    plt.plot(x / 1e-6, y, 'o')
    plt.semilogx(
        x / 1e-6,
        phase_curve(x,
                    params,
                    df_dphi,
                    10e-9,
                    0,
                    wav=wav,
                    sample_rate=sample_rate))
    plt.plot(x / 1e-6, f(x, *popt))

    plt.xlabel('delay [us]')
    plt.ylabel('phase')
    plt.show()
