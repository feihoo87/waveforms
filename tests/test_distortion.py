import numpy as np
from scipy.signal import sosfilt

from waveforms.math.signal.distortion import *


def test_exp_decay_filter():
    N = 5
    sample_rate = 1000

    amp, tau = np.random.randn(N) / 5, np.exp(np.abs(np.random.randn(N)))

    b, a = exp_decay_filter(amp, tau, sample_rate, output='ba')
    z, p, k = exp_decay_filter(amp, tau, sample_rate, output='zpk')
    sos = exp_decay_filter(amp, tau, sample_rate, output='sos')

    x = np.linspace(-10, 20, 30000, endpoint=False)
    y = np.heaviside(x, 1)
    yy = y
    for A, t in zip(amp, tau):
        yy = yy - A * y * np.exp(-x / t)

    np.allclose(sosfilt(sos, y), yy)
