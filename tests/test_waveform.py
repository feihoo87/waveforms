import numpy as np

from waveforms import *


def test_waveform():
    wav = cos(1)
    t = np.linspace(-10,10,1001)

    assert np.allclose(wav(t), np.cos(t))
