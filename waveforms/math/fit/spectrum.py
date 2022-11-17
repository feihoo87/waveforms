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


def transmon_spectrum2(x, EJ, Ec, d, offset, period):
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

    return curve_fit(transmon_spectrum2,
                     bias,
                     f01,
                     p0=[EJ, Ec, d, offset, period])
