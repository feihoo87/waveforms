import numpy as np
from scipy.optimize import leastsq

from waveforms.math.fit.simple import fit_circle
from waveforms.math.signal.func import complexPeaks


def get_unit_prefix(value):
    '''
    获取 value 合适的单位前缀，以及相应的倍数

    >>>
        y => 1e-24       Y => 1e24
        z => 1e-21       Z => 1e21
        a => 1e-18       E => 1e18
        f => 1e-15       P => 1e15
        p => 1e-12       T => 1e12
        n => 1e-9        G => 1e9
        u => 1e-6        M => 1e6
        m => 1e-3        k => 1e3

    Returns:
        (prefix, multiple)
    '''
    prefixs = [
        'y', 'z', 'a', 'f', 'p', 'n', 'u', 'm', '', 'k', 'M', 'G', 'T', 'P',
        'E', 'Z', 'Y'
    ]
    if value == 0:
        return '', 1
    x = np.floor(np.log10(abs(value)) / 3)
    x = 0 if x < -8 else x
    x = 0 if x > 8 else x
    return prefixs[int(x) + 8], 1000**(x)


def valueString(value, unit=""):
    """
    将 value 转换为更易读的形式

    >>> valueString(1.243e-7, 's')
    ... "124.3 ns"
    >>> valueString(1.243e10, 'Hz')
    ... "12.43 GHz"
    """
    prefix, k = get_unit_prefix(value)
    return f"{value/k:g} {prefix}{unit}"


def S21(f, f0, QL, Qc, phi):
    width = f0 / (2 * QL)
    amp = -QL / np.abs(Qc) * np.exp(1j * phi)
    return complexPeaks(f, [(f0, width, amp)], 1)


def invS21(freq, f0, Qc, Qi, phi):
    width = f0 / (2 * Qi)
    amp = Qi / np.abs(Qc) * np.exp(1j * phi)
    return complexPeaks(freq, [(f0, width, amp)], 1)


def getBackground(x, s):
    phi = np.unwrap(np.angle(s), 0.9 * np.pi)
    delay = (phi[-1] - phi[0]) / (2 * np.pi * (x[-1] - x[0]))
    A = np.abs(s).max()

    return delay, A


def guessParams(x, s, inverse=True):
    if inverse:
        data = 1 / s
    else:
        data = s

    xc, yc, R = fit_circle(np.real(data), np.imag(data))
    y = (data - xc - 1j * yc) / R
    theta = np.angle(0.5 * (y[0] + y[-1]))
    y /= np.exp(1j * theta)
    index = np.abs(np.outer(y, np.ones(3)) -
                   np.array([-1j, -1, 1j])).argmin(axis=0)
    f1, fc, f2 = x[index]
    FWHM = abs(f2 - f1)

    if inverse:
        #f0 = x[np.abs(data).argmax()]
        Qi = fc / FWHM
        Qe = Qi / (2 * R)
        QL = 1 / (1 / Qi + 1 / Qe)
        phi = np.arctan2(yc, xc - 1)
    else:
        #f0 = x[np.abs(data).argmin()]
        QL = fc / FWHM
        Qe = QL / (2 * R)
        Qi = 1 / (1 / QL - 1 / Qe)
        phi = np.arctan2(-yc, 1 - xc)
    return [fc, Qi, Qe, QL, phi]


def background(f, params):
    f0, A, delay, Aphi, a, b = params
    return A * ((a * (f - f0)**2 + b *
                 (f - f0)) + 1) * np.exp(1j * (delay * (f - f0) + Aphi))


def fitS21(x, s, params):

    def err(params, f, s21):
        f0, QL, Qe, phi, A, Aphi, delay = params
        y = s21 - S21(f, f0, abs(QL), abs(Qe), phi) * A * np.exp(
            1j * (delay * (f - x[0]) + Aphi))
        return np.abs(y)

    f0, Qi, Qe, QL, phi = params
    params = [f0, QL, Qe, phi, 1, 0, 0]

    res = leastsq(err, params, (x, s))
    f0, QL, Qe, phi, A, Aphi, delay = res[0]
    Qi = 1 / (1 / abs(QL) - 1 / abs(Qe))
    return f0, Qi, abs(Qe), abs(QL), phi, A, Aphi, delay


def fitInvS21(x, s, params, background=(1, 0, 0)):
    A, Aphi, delay = background

    def err(params, f, s21):
        f0, QL, Qe, phi, *_ = params
        y = 1 / s21 - invS21(f, f0, abs(Qe), abs(Qi),
                             phi) / (A * np.exp(1j * (delay *
                                                      (f - x[0]) + Aphi)))
        return np.abs(y)

    f0, Qi, Qe, QL, phi = params
    params = [f0, Qe, Qi, phi]

    res = leastsq(err, params, (x, s))
    f0, Qe, Qi, phi, *_ = res[0]
    QL = 1 / (1 / abs(Qi) + 1 / abs(Qe))
    return f0, abs(Qi), abs(Qe), QL, phi, A, Aphi, delay


def plotData(fig, x, y):
    ax = fig.add_subplot(211)
    ax.plot(x / 1e9, 20 * np.log10(np.abs(y)))
    ax.set_xlabel('Frequency / GHz')
    ax.set_ylabel('|S21| / dB')

    ax = ax.twinx()
    phase = np.unwrap(np.angle(y), 0.9 * np.pi)
    ax.plot(x / 1e9, phase / np.pi, color="C1")
    ax.set_xlabel('Frequency / GHz')
    ax.set_ylabel(r'Ang(S21) / $\pi$')

    ax = fig.add_subplot(212)
    ax.plot(x / 1e9, np.real(y), label='Re')
    ax.plot(x / 1e9, np.imag(y), label='Im')
    ax.set_xlabel('Frequency / GHz')
    ax.set_ylabel('S21')
    ax.legend()


def printReport(params, method='UCSB'):

    f0, Qi, Qe, QL, phi, *others = params
    if len(others) >= 3:
        A, Aphi, delay, *others = others
    else:
        A, Aphi, delay = 1, 0, 0
        others = ()
    if len(others) == 3:
        a, b = others
    else:
        a, b = 0, 0

    print(f'''{method} Method
    f0 = {valueString(f0, 'Hz')}
    Qi = {valueString(Qi)}    Qe = {valueString(Qe)}    QL = {valueString(QL)}
    phi = {phi*180/np.pi:g} deg
    A = {valueString(A)}    delay = {valueString(delay, 's')}
    
    Aphi = {Aphi*180/np.pi:g} deg   a = {a:g} Hz^-2    b = {b:g} Hz^-1''')
    print()


def plotFit(fig, f, s, params):
    from matplotlib.gridspec import GridSpec

    f0, Qi, Qe, QL, phi, *others = params
    if len(others) == 0:
        A, Aphi, delay = 1, 0, 0
    else:
        A, Aphi, delay = others

    y = s / (A * np.exp(1j * (delay * (f - f[0]) + Aphi)))
    invY = 1 / y
    
    invs21 = invS21(f, f0, Qe, Qi, phi)
    s21 = 1 / invs21

    gs = GridSpec(4, 4, figure=fig)

    ax = fig.add_subplot(gs[0:2, 0:2])
    ax.plot(np.real(y), np.imag(y), '.')
    ax.plot(np.real(s21), np.imag(s21))
    ax.axis('equal')
    ax.set_xlabel("Re(S21)")
    ax.set_ylabel("Im(S21)")

    ax = fig.add_subplot(gs[0:2, 2:4])
    ax.plot(np.real(invY), np.imag(invY), '.')
    ax.plot(np.real(invs21), np.imag(invs21))
    ax.axis('equal')
    ax.set_xlabel("Re(S21$^{-1}$)")
    ax.set_ylabel("Im(S21$^{-1}$)")

    ax = fig.add_subplot(gs[2:3, 0:4])
    ax.plot(f / 1e9, np.real(y), '.')
    ax.plot(f / 1e9, np.imag(y), '.')
    ax.plot(f / 1e9, np.real(s21), label='Re')
    ax.plot(f / 1e9, np.imag(s21), label='Im')
    ax.set_xlabel('Frequency / GHz')
    ax.set_ylabel('S21')
    ax.legend()

    ax = fig.add_subplot(gs[3:4, 0:4])
    ax.plot(f / 1e9, 20 * np.log10(np.abs(y)), '.', color="C0")
    ax.plot(f / 1e9, 20 * np.log10(np.abs(s21)), color="C2")
    ax.set_xlabel('Frequency / GHz')
    ax.set_ylabel('|S21| / dB')

    ax = ax.twinx()
    phase = np.unwrap(np.angle(y), 0.9 * np.pi)
    ax.plot(f / 1e9, phase / np.pi, '.', color="C1", alpha=0.5)
    phase = np.unwrap(np.angle(s21), 0.9 * np.pi)
    ax.plot(f / 1e9, phase / np.pi, color="C3", alpha=0.5)
    ax.set_xlabel('Frequency / GHz')
    ax.set_ylabel(r'Ang(S21) / $\pi$')
