import warnings
from itertools import zip_longest
from typing import Sequence, cast

import numpy as np
from numpy.typing import NDArray
from scipy.fftpack import fft, fftfreq, ifft, ifftshift
from scipy.optimize import curve_fit, linear_sum_assignment
from scipy.signal import fftconvolve, lfilter, lfiltic, tf2zpk, zpk2sos, zpk2tf


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


def drag_fir(deltas, Ts, *, method="notch"):
    """deltas: rad/s；Ts: s；返回按 h[0], h[1], ... 排列的复数抽头。"""
    d = np.atleast_1d(np.asarray(deltas, dtype=float))

    if d.ndim != 1 or not np.isfinite(Ts) or Ts <= 0:
        raise ValueError("deltas 必须是一维，Ts 必须为有限正数")
    if np.any(~np.isfinite(d)) or np.any(d == 0):
        raise ValueError("Delta 必须有限且非零")

    theta = d * Ts
    if method == "central":
        a = 0.5 / theta
    elif method == "notch":
        if np.any(np.abs(theta) >= np.pi):
            raise ValueError("目标陷波必须严格位于 Nyquist 范围内")
        a = 0.5 / np.sin(theta)
    else:
        raise ValueError("method 必须为 'central' 或 'notch'")

    h = np.array([1.0 + 0j])
    for ai in a:
        h = np.convolve(h, [-1j * ai, 1.0, 1j * ai])
    return h


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

    ker = cast(NDArray[np.complex128], ifftshift(ifft(1 / H))).real
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


def _exp_decay_polynomials(
    amp: Sequence[float],
    tau: Sequence[float],
) -> tuple[np.poly1d, np.poly1d]:
    """Return the continuous-time numerator and denominator polynomials."""
    numerator, denominator = np.poly1d([0.0]), np.poly1d([1.0])
    for t in tau:
        denominator = denominator * np.poly1d([1, -1 / t])
    for i, A in enumerate(amp):
        n = np.poly1d([-A, 0.0])
        for j, t in enumerate(tau):
            if j != i:
                n = n * np.poly1d([1, -1 / t])
        numerator = numerator + n
    return numerator + denominator, denominator


def exp_decay_filter_to_cascade(
    amp: Sequence[float],
    tau: Sequence[float],
) -> list[list[float]]:
    """Convert a multiexponential response to real first-order stages.

    ``exp_decay_filter(amp, tau, sample_rate)`` treats the exponential
    components as a parallel sum. This function factors that transfer
    function into scalar ``exp_decay_filter(a, t, sample_rate)`` stages whose
    cascade is equivalent for every sample rate.

    The factorization is not unique because any zero can be paired with any
    pole. This implementation selects the pairing that minimizes the sum of
    the absolute stage amplitudes and keeps the time constants in input order.

    Args:
        amp: Amplitudes of the fitted exponential tails.
        tau: Positive decay times corresponding to ``amp``.

    Returns:
        A list ``[[a_1, t_1], [a_2, t_2], ...]``. Passing each pair to
        ``exp_decay_filter`` and cascading the resulting filters is equivalent
        to passing the original arrays to ``exp_decay_filter``.

    Raises:
        ValueError: If inputs are invalid, or if the transfer function has
            complex zeros and therefore cannot be factored into real
            first-order stages.

    Notes:
        With ``r_i = 1 / tau_i``, the original transfer function is

        ``H(s) = 1 - sum(A_i * s / (s + r_i))``.

        If ``-q_j`` is one of its zeros, pairing it with pole ``-r_i`` gives
        a scalar stage amplitude ``a_i = 1 - r_i / q_j``. A missing finite
        zero (a zero at infinity) gives ``a_i = 1``.
    """
    amp_array = np.asarray(amp, dtype=float)
    tau_array = np.asarray(tau, dtype=float)
    if amp_array.ndim != 1 or tau_array.ndim != 1:
        raise ValueError("amp and tau must be one-dimensional sequences")
    if len(amp_array) != len(tau_array):
        raise ValueError("amp and tau must have the same length")
    if not np.all(np.isfinite(amp_array)):
        raise ValueError("amp must contain only finite values")
    if not np.all(np.isfinite(tau_array)) or np.any(tau_array <= 0):
        raise ValueError("tau must contain only finite positive values")
    if len(amp_array) == 0:
        return []

    # Root finding is better conditioned after removing the common time scale.
    log_rates = -np.log(tau_array)
    scaled_rates = np.exp(log_rates - np.mean(log_rates))
    scaled_tau = 1 / scaled_rates
    numerator, _ = _exp_decay_polynomials(amp_array, scaled_tau)
    zeros = np.asarray(numerator.roots, dtype=complex)

    root_scale = np.maximum(1.0, np.abs(zeros.real))
    if np.any(np.abs(zeros.imag) > 1e-9 * root_scale):
        raise ValueError(
            "the response has complex zeros and cannot be represented by "
            "a cascade of real first-order exp_decay_filter stages"
        )
    real_zeros = zeros.real

    # A real zero cannot be exactly zero because H(0) == 1. Guard against a
    # numerically singular root before forming r_i / q_j.
    if np.any(np.abs(real_zeros) <= np.finfo(float).eps):
        raise ValueError("the response has a numerically singular zero")

    stage_amp = np.ones(len(tau_array), dtype=float)
    if len(real_zeros):
        costs = np.abs(1 - scaled_rates[:, None] / real_zeros[None, :])
        pole_indices, zero_indices = linear_sum_assignment(costs)
        stage_amp[pole_indices] = (
            1 - scaled_rates[pole_indices] / real_zeros[zero_indices]
        )

    return [[float(a), float(t)] for a, t in zip(stage_amp, tau_array)]


def exp_decay_filter_from_cascade(
    cascade: Sequence[Sequence[float]],
) -> tuple[list[float], list[float]]:
    """Convert real first-order stages back to multiexponential parameters.

    This is the inverse of :func:`exp_decay_filter_to_cascade` when all
    non-identity stages have distinct time constants.

    Args:
        cascade: First-order stages ``[[a_1, t_1], [a_2, t_2], ...]``.
            Each pair represents ``exp_decay_filter(a_i, t_i, sample_rate)``.

    Returns:
        A tuple ``([A_1, A_2, ...], [tau_1, tau_2, ...])`` suitable for
        ``exp_decay_filter(amp, tau, sample_rate)``. Identity stages with
        ``a_i == 0`` are retained as zero-amplitude terms.

    Raises:
        ValueError: If the input is invalid, or if two non-identity stages
            have the same time constant. Such a cascade has a repeated pole
            and generally contains terms such as ``t * exp(-t / tau)``, so it
            cannot be represented as a sum of simple exponential tails.

    Notes:
        Write ``r_i = 1 / t_i``. The residue of the cascade at ``s = -r_i``
        gives the parallel amplitude directly:

        ``A_i = a_i * product((r_j - (1 - a_j) * r_i) / (r_j - r_i))``

        where the product is over all other non-identity stages.
    """
    cascade_array = np.asarray(cascade, dtype=float)
    if cascade_array.size == 0:
        return [], []
    if cascade_array.ndim != 2 or cascade_array.shape[1] != 2:
        raise ValueError("cascade must have shape (n, 2)")
    if not np.all(np.isfinite(cascade_array)):
        raise ValueError("cascade must contain only finite values")

    stage_amp = cascade_array[:, 0]
    tau_array = cascade_array[:, 1]
    if np.any(tau_array <= 0):
        raise ValueError("cascade time constants must be positive")

    active = np.flatnonzero(stage_amp != 0)
    if len(np.unique(tau_array[active])) != len(active):
        raise ValueError(
            "non-identity cascade stages must have distinct time constants"
        )

    # Only rate ratios occur in the residue formula. Removing the common time
    # scale improves numerical conditioning without changing the result.
    log_rates = -np.log(tau_array)
    rates = np.exp(log_rates - np.mean(log_rates))
    amp_array = np.zeros(len(cascade_array), dtype=float)
    for i in active:
        others = active[active != i]
        factors = (
            rates[others] - (1 - stage_amp[others]) * rates[i]
        ) / (rates[others] - rates[i])
        amp_array[i] = stage_amp[i] * np.prod(factors)

    return amp_array.tolist(), tau_array.tolist()


def exp_decay_filter(
    amp: float | Sequence[float],
    tau: float | Sequence[float],
    sample_rate: float,
    inv: bool = False,
    output='ba'
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[
        np.float64]] | tuple[NDArray[np.float64], NDArray[np.float64], float]:
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
        inv (bool): if True, the filter is inverted
        output (str): output type, 'ba' for numerator (b) and denominator (a)
            polynomials, 'sos' for second-order sections, 'zpk' for zeros (z),
            poles (p) and gain (k). See scipy.signal.lfilter for more.

    Returns:
        if output is 'ba', return (b, a) array like, numerator (b) and denominator (a)
        polynomials of the IIR filter. See scipy.signal.lfilter for more.
        if output is 'sos', return array of second-order filter coefficients with shape
        (n_sections, 6). See scipy.signal.sosfilt for more.
        if output is 'zpk', return (z, p, k) array like, zeros (z), poles (p) and gain (k).
        See scipy.signal.zpk2tf for more.
    
    Raises:
        ValueError: if output is not 'ba', 'sos', or 'zpk'
    
    Notes:
        The filter is stable if all poles are inside the unit circle.
    """

    if isinstance(amp, (int, float, complex)):
        amp = [amp]
        tau = [cast(float, tau)]
    amp = cast(Sequence[float], amp)
    tau = cast(Sequence[float], tau)
    numerator, denominator = _exp_decay_polynomials(amp, tau)

    z = cast(NDArray[np.float64], np.exp(-numerator.roots / sample_rate))
    # p = cast(NDArray[np.float64], np.exp(-denominator.roots / sample_rate))
    p = np.exp(-1 / (np.asarray(tau) * sample_rate))

    if inv:
        z, p = p, z
    # remove poles outside the unit circle to make the filter stable
    p = p[np.abs(p) < 1]
    k = cast(float, (np.prod(1 - p) / np.prod(1 - z)).real)

    if output == 'sos':
        return cast(NDArray[np.float64], zpk2sos(z, p, k))
    elif output == 'ba':
        return cast(tuple[NDArray[np.float64], NDArray[np.float64]],
                    zpk2tf(z, p, k))
    elif output == 'zpk':
        return z, p, k
    else:
        raise ValueError(f"Invalid output type: {output}")


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
    from waveforms.waveform import ComplexWaveform, Waveform

    if isinstance(sig, (Waveform, ComplexWaveform)):
        return 1 / (1 - A) * sig - A / (1 - A) * (sig >> tau)
    if sample_rate is not None:
        freq = np.fft.fftfreq(len(sig), 1 / sample_rate)
        return np.fft.ifft(np.fft.fft(sig) /
                           reflection_filter(freq, A, tau)).real
    else:
        raise ValueError('sample_rate is not given')


def combine_filters(
    filters: list[tuple[np.ndarray,
                        np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """
    combine filters

    Args:
        filters (list): list of (b, a) array like, numerator (b) and denominator
        (a) polynomials of the IIR filter. See scipy.signal.lfilter for more.

    Returns:
        tuple: (b, a) array like, numerator (b) and denominator (a)
        polynomials of the combined filter. See scipy.signal.lfilter for more.
    """
    b, a = np.poly1d([1.0]), np.poly1d([1.0])
    for b_, a_ in filters:
        b = b * np.poly1d(b_)
        a = a * np.poly1d(a_)
    return b.coeffs, a.coeffs


def factor_filter(b, a):
    """
    factor filter

    Args:
        b (array_like): numerator polynomial of the IIR filter.
        a (array_like): denominator polynomial of the IIR filter.

    Returns:
        list: list of (b, a) array like, numerator (b) and denominator
    """
    b, a = np.poly1d(b), np.poly1d(a)
    p = a.roots
    q = b.roots
    b_amp = (b[0] / a[0])**(1 / max(len(q), len(p)))
    filters = []
    for a_, b_ in zip_longest(p, q, fillvalue=0):
        filters.append(([b_amp, -b_amp * b_], [1, -a_]))
    return filters


def stable_filter(exp_decay_filters: list, sample_rate: float):
    """
    check if the filter is stable

    Args:
        exp_decay_filters (list): list of (amp, tau) pairs
    """
    filters = []
    for amp, tau in exp_decay_filters:
        a, b = cast(tuple[NDArray[np.float64], NDArray[np.float64]],
                    exp_decay_filter(amp, tau, sample_rate))
        filters.append((b, a))

    b, a = combine_filters(filters)
    z, p, k = tf2zpk(b, a)
    if np.all(np.abs(p) < 1):
        return True
    else:
        return False


def predistort(
        sig: np.ndarray,
        filters: list | None = None,
        ker: np.ndarray | None = None,
        initial: float = 0.0,
        initial_x: np.ndarray | None = None,
        initial_y: np.ndarray | None = None,
        zi: np.ndarray | None = None,
        return_zf: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    if filters is not None:
        b, a = combine_filters(filters)
        z, p, k = tf2zpk(b, a)
        if np.all(np.abs(p) < 1):
            pass
        else:
            warnings.warn('Warning: filter is unstable')

        if zi is None:
            if initial_x is None:
                initial_x = np.full((len(b) - 1, ), initial)
            else:
                initial_x = np.asarray(initial_x)[:len(b) - 1]
            if initial_y is None:
                initial_y = np.full((len(a) - 1, ), initial)
            else:
                initial_y = np.asarray(initial_y)[:len(a) - 1]
            zi = lfiltic(
                b,
                a,
                initial_y,
                initial_x,
            )
        sig, zf = lfilter(b, a, sig, zi=zi)

    if ker is None:
        if return_zf:
            return sig, zf
        else:
            return sig

    size = len(sig)
    sig = np.hstack((np.zeros_like(sig), sig, np.zeros_like(sig)))
    start = size + len(ker) // 2
    stop = start + size
    points = fftconvolve(sig, ker, mode='full')[start:stop]
    if return_zf:
        return points, zf
    else:
        return points


def distort(points, params, sample_rate, initial=0.0):
    filters = []
    for amp, tau in np.asarray(params).reshape(-1, 2):
        b, a = cast(tuple[NDArray[np.float64], NDArray[np.float64]],
                    exp_decay_filter(amp, abs(tau), sample_rate))
        filters.append((b, a))
    return predistort(points, filters, initial=initial)


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
    import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

    from waveforms import square

    data = np.load('Z_distortion.npz')

    x = data['time'] * 1e-6
    y = data['phase']
    df_dphi = 4343.313e6

    sample_rate = 2e9
    wav = 0.1 * (square(2e-6) << 1e-6)

    def f(t, *params):
        return phase_curve(t, params, df_dphi, 10e-9, 25e-9, wav, sample_rate)

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
