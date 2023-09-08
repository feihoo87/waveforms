import numpy as np
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit
from scipy.signal import peak_widths

from ..signal.func import peaks


def peaks_fun(x, *args):
    n = (len(args) - 1) // 4
    p = []
    for i in range(n):
        center, sigma, gamma, amp = [*args[4 * i:4 * i + 4]]
        p.append((center, sigma, gamma, amp))
    bg = args[-1]
    return peaks(x, p, bg)


def _fit_single_peak(x, y):
    i = np.argmax(y)
    widths, *_ = peak_widths(
        y,
        [i],
        rel_height=0.5,
    )
    width = max(widths[0], 3)
    width = min(i, len(x) - i - 1, width)

    gamma = width / len(x) * (x[-1] - x[0])

    f0 = x[i]
    offset = np.median(y)
    amp = y[i] - offset

    start = max(0, i - 4 * int(width))
    stop = min(len(x), i + 4 * int(width))

    popt, pcov = curve_fit(
        peaks_fun,
        x[start:stop],
        y[start:stop], [f0, 0, gamma, amp, offset],
        bounds=([x[0], 0, (x[1] - x[0]) / 2, 0,
                 np.min(y) - np.max(y)], [
                     x[-1], x[-1] - x[0], x[-1] - x[0],
                     np.max(y) - np.min(y),
                     np.max(y)
                 ]),
        method='trf')
    return popt


def fit_peaks(x, y, n=1):
    """
    Fit peaks in y(x) with n peaks.

    Args:
        x: np.array
        y: np.array
        n: number of peaks to fit

    Returns:
        p: list of (center, width, amp, shape)
        bg: background
        See also: waveforms.math.signal.func.peaks
    """
    norm_x = Normalize(vmax=np.max(x), vmin=np.min(x))
    norm_y = Normalize(vmax=np.max(y), vmin=np.min(y))

    ydata = norm_y(y)
    xdata = norm_x(x)

    p = []
    for i in range(n):
        popt = _fit_single_peak(xdata, ydata)
        ydata -= peaks_fun(xdata, *popt)
        f0, sigma, gamma, amp, offset = popt
        p.extend([f0, sigma, gamma, amp])

    ydata = norm_y(y)
    p.append(np.median(ydata))
    popt, pcov = curve_fit(peaks_fun,
                           xdata,
                           ydata,
                           p0=p,
                           method='trf',
                           maxfev=10000)
    p = []
    for i in range(n):
        center, sigma, gamma, amp = [*popt[4 * i:4 * i + 4]]
        p.append(
            (norm_x.inverse(center), sigma, gamma * (np.max(x) - np.min(x)),
             amp * (np.max(y) - np.min(y))))
    bg = norm_y.inverse(popt[-1])
    return p, bg
