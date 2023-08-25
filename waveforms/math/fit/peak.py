import numpy as np
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit
from scipy.signal import peak_widths

from ..signal.func import peaks


def peaks_fun(x, *args):
    n = (len(args) - 1) // 3
    p = []
    for i in range(n):
        center, width, amp, shape = [*args[3 * i:3 * i + 3], 'lorentzianAmp']
        p.append((center, width, amp, shape))
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

    popt, pcov = curve_fit(peaks_fun,
                           x[start:stop],
                           y[start:stop], [f0, gamma, amp, offset],
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
        f0, gamma, amp, offset = popt
        p.extend([f0, gamma, amp])

    ydata = norm_y(y)
    p.append(np.median(ydata))
    popt, pcov = curve_fit(peaks_fun, xdata, ydata, p0=p, method='trf')
    p = []
    for i in range(n):
        center, width, amp, shape = [*popt[3 * i:3 * i + 3], 'lorentzianAmp']
        p.append((norm_x.inverse(center), width * (np.max(x) - np.min(x)),
                  amp * (np.max(y) - np.min(y)), shape))
    bg = norm_y.inverse(popt[-1])
    return p, bg
