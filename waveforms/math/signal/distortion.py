from itertools import repeat
from typing import Sequence

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


def high_pass_filter(tau, sample_rate):
    """
    high pass filter
    """
    k = 2.0 * tau * sample_rate
    a = [1.0, (1 - k) / (1 + k)]
    b = [k / (1 + k), -k / (1 + k)]
    return b, a


def exp_decay_filter_old(amp, tau, sample_rate):
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

    alpha = 1 - np.exp(-1 / (abs(sample_rate * tau) * (1 + amp)))

    if amp >= 0:
        k = amp / (1 + amp - alpha)
        a = [(1 - k + k * alpha), -(1 - k) * (1 - alpha)]
    else:
        k = -amp / (1 + amp) / (1 - alpha)
        a = [(1 + k - k * alpha), -(1 + k) * (1 - alpha)]

    b = [1 / a[0], -(1 - alpha) / a[0]]
    a = [1, a[1] / a[0]]

    return b, a


def exp_decay_filter(amp: float | Sequence[float],
                     tau: float | Sequence[float],
                     sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """
    exp decay filter

    Infinite impulse response as multiexponential decay. When input signal
    is the Heaviside theta function u(t), the output signal is:
    out(t) = u(t) * (1 - A_1 * exp(-t / tau_1) - A_2 * exp(-t / tau_2) ...)
    where A_i and tau_i are the amplitude and decay time of the i-th
    exponential decay.

    The transfer function of the filter is:

    H(w) = 1 - H_1(w) - H_2(w) - ... - H_n(w)

    where
                       A_i
    H_i(w) = --------------------------
              1 - 1 / (1j * w * tau_i)

    Args:
        amp (float): amplitude of the filter
        tau (float): decay time
        sample_rate (float): sampling rate

    Returns:
        tuple: (b, a) array like, numerator (b) and denominator (a)
        polynomials of the IIR filter. See scipy.signal.lfilter for more.
    """

    if isinstance(amp, (int, float, complex)):
        amp = [amp]
        tau = [tau]
    numerator, denominator = np.poly1d([0.0]), np.poly1d([1.0])
    for i, (A, t) in enumerate(zip(amp, tau)):
        denominator = denominator * np.poly1d([1, -1 / t])
        n = np.poly1d([-A, 0.0])
        for j, t_ in enumerate(tau):
            if j != i:
                n = n * np.poly1d([1, -1 / t_])
        numerator = numerator + n
    numerator = numerator + denominator
    xi = numerator.roots
    p = denominator.roots

    b, a = np.poly1d([1.0]), np.poly1d([1.0])
    for x in xi:
        b = b * np.poly1d([1, -np.exp(-x / sample_rate)])
    for p_ in p:
        a = a * np.poly1d([1, -np.exp(-p_ / sample_rate)])

    kd = numerator(0) * a(1) / denominator(0) / b(1)
    b, a = b.coeffs.real * kd, a.coeffs.real
    return b / a[0], a / a[0]


def reflection_filter(f, A, tau):
    """
    reflection filter

    Infinite impulse response as reflection. When input signal
    is in(t), the output signal is:
    out(t) = in(t) + A * in(t - tau) + A^2 * in(t - 2 * tau) + ...

    The transfer function of the filter is:
                      1 - A
    H(w) = ----------------------------
            1 - A * exp(- i * w * tau)
    Args:
        f (float): frequency
        A (float): amplitude of the reflection
        tau (float): delay time
    """
    return (1 - A) / (1 - A * np.exp(-2j * np.pi * f * tau))


def reflection(sig, A, tau, sample_rate):
    freq = np.fft.fftfreq(len(sig), 1 / sample_rate)
    return np.fft.ifft(np.fft.fft(sig) * reflection_filter(freq, A, tau)).real


def correct_reflection(sig, A, tau, sample_rate=None):
    from ...waveform import Waveform

    if isinstance(sig, Waveform):
        return 1 / (1 - A) * sig - A / (1 - A) * (sig >> tau)
    if sample_rate is not None:
        freq = np.fft.fftfreq(len(sig), 1 / sample_rate)
        return np.fft.ifft(np.fft.fft(sig) /
                           reflection_filter(freq, A, tau)).real
    else:
        raise ValueError('sample_rate is not given')


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
        b, a = exp_decay_filter(amp, abs(tau), sample_rate)
        filters.append((b, a))
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
