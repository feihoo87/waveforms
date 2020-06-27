import numpy as np
import scipy.special as special

from waveforms import *


def test_waveform():
    t = np.linspace(-10, 10, 1001)

    wav = cos(1)
    assert np.allclose(wav(t), np.cos(t), atol=1e-04)

    wav = sin(1)
    assert np.allclose(wav(t), np.sin(t), atol=1e-04)

    width = 2
    wav = gaussian(width)
    std_sq2 = width / (4 * np.sqrt(np.log(2)))
    assert np.allclose(wav(t), np.exp(-(t / std_sq2)**2), atol=5e-03)


def test_shift():
    width = 2
    wav = gaussian(width) >> 3
    std_sq2 = width / (4 * np.sqrt(np.log(2)))
    assert np.allclose(wav(t), np.exp(-((t-3) / std_sq2)**2), atol=5e-03)
