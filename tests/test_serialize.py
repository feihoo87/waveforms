import pickle

import numpy as np

from waveforms import *
from waveforms.waveform import WaveVStack


def test_waveform():
    wlist = []
    for x in range(100):
        wlist.append(drag(4.5e9, 20e-9, 0, -5e6, -200e6, 0, x * 30e-9))
    wav = WaveVStack(wlist)
    wav.start = 0
    wav.stop = 100e-6
    wav.sample_rate = 6e9

    buf = pickle.dumps(wav)
    assert isinstance(buf, bytes)
    wav2 = pickle.loads(buf)
    assert isinstance(wav2, WaveVStack)

    assert np.allclose(wav.sample(), wav2.sample())
    assert wav.simplify() == wav2.simplify()

    wav = wav.simplify()
    assert isinstance(wav, Waveform) and not isinstance(wav, WaveVStack)
    buf = pickle.dumps(wav)
    assert isinstance(buf, bytes)
    wav2 = pickle.loads(buf)
    assert isinstance(wav2, Waveform)

    assert np.allclose(wav.sample(), wav2.sample())
    assert wav == wav2
