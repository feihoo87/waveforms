import numpy as np
from scipy.optimize import minimize
from scipy.signal import correlate


def fit_delay(waveform, data, sample_rate, fit=True):
    t = np.linspace(0, len(data) / sample_rate, len(data), endpoint=False)
    if isinstance(waveform, np.ndarray):
        fit = False
        corr = correlate(waveform, data, mode='same', method='fft')
    else:
        corr = correlate(waveform(t), data, mode='same', method='fft')
    delay = 0.5 * (t[0] + t[-1]) - t[np.argmax(corr)]

    if not fit:
        return delay

    def fun(delay, t, waveform, sig):
        if isinstance(delay, (int, float, complex)):
            ret = -correlate((waveform >> delay)(t), sig, mode='valid')[0]
            return ret
        else:
            return np.array([fun(d, t, waveform, sig) for d in delay])

    ret = minimize(fun, [delay], args=(t, waveform, data))
    return ret.x[0]
