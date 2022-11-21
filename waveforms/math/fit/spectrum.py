import numpy as np
from scipy.optimize import curve_fit


def transmon_spectrum(x, EJ, Ec, d, offset, period):
    from waveforms import Transmon

    x = (x - offset) / period
    q = Transmon(EJ=EJ, Ec=Ec, d=d)
    if isinstance(x, (int, float, complex)):
        return q.levels(flux=x)[1] - q.levels(flux=x)[0]
    else:
        y = []
        for b in x:
            y.append(q.levels(flux=b)[1] - q.levels(flux=b)[0])
        return np.asarray(y)


def transmon_spectrum_fast(x, EJ, Ec, d, offset, period):
    from scipy.special import mathieu_a, mathieu_b
    from waveforms.quantum.transmon import Transmon

    x = (x - offset) / period
    q = 0.5 * Transmon._flux_to_EJ(x, EJ, d) / Ec

    # if ng == 0:
    #     return Ec * (mathieu_b(2, -q) - mathieu_a(0, -q))
    # if ng == 0.5:
    #     return Ec * (mathieu_b(1, -q) - mathieu_a(0, -q))
    return Ec * (mathieu_b(1, -q) + mathieu_b(2, -q) -
                 2 * mathieu_a(0, -q)) / 2


def fit_transmon_spectrum(bias,
                          f01,
                          offset=0,
                          period=1,
                          f01_max=None,
                          f01_min=None,
                          alpha=None):
    from waveforms import Transmon

    x = (bias - offset) / period

    f01_max = np.max(f01) if f01_max is None else f01_max
    f01_min = np.min(f01) if f01_min is None else f01_min
    alpha = -0.24 if alpha is None else alpha

    q = Transmon(f01_max=f01_max, f01_min=f01_min, alpha=alpha)
    EJ, Ec, d = q.EJ, q.Ec, q.d

    return curve_fit(transmon_spectrum_fast,
                     bias,
                     f01,
                     p0=[EJ, Ec, d, offset, period])


def cavity_spectrum(x, offset, period, fc, f01, g=0.05, d=0.1, Ec=0.24):
    EJS = (f01 + Ec)**2 / 8 / Ec
    F = np.pi * (x - offset) / period

    EJ = EJS * np.sqrt(np.cos(F)**2 + d**2 * np.sin(F)**2)
    Delta = np.sqrt(8 * EJ * Ec) - Ec - fc
    a, b = Delta - np.sqrt(Delta**2 + g**2), Delta + np.sqrt(Delta**2 + g**2)
    ret = np.zeros_like(a)
    mask = np.abs(a) > np.abs(b)
    ret[mask] = b[mask]
    mask = np.abs(a) <= np.abs(b)
    ret[mask] = a[mask]
    return ret + fc
