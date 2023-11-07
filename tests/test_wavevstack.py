import numpy as np
import scipy.special as special
from scipy.signal import butter, lfilter, lfiltic, tf2sos

from waveforms import *
from waveforms.waveform import WaveVStack


def test_wavevstack():
    t = np.linspace(-10, 10, 1001)

    wlist = [cos(1), sin(2), gaussian(3), poly([1, -1 / 2, 1 / 6, -1 / 12])]
    w1 = zero()
    for w in wlist:
        w1 += w
    w2 = WaveVStack(wlist)
    assert w2.simplify() == w1

    assert np.allclose(w1(t), w2(t), atol=1e-04)

    w2.start = -10
    w2.stop = 10.02
    w2.sample_rate = 50
    assert np.allclose(w2.sample(), w1(t), atol=1e-04)


def test_tolist():
    wlist = [cos(1), sin(2), gaussian(3), poly([1, -1 / 2, 1 / 6, -1 / 12])]
    w = WaveVStack(wlist)

    l = w.tolist()
    assert l == [
        None, None, 0, 0, None, None, 4, 1, np.inf, 1, 1.0, 1, 1, 3, 4, 1, 0.0,
        1, np.inf, 1, 1.0, 1, 1, 3, 4, 2, 0.7853981633974483, 3, -2.25, 0,
        2.25, 1, 1.0, 1, 1, 3, 2, 0.9008418065898374, 0, np.inf, 0, 1, np.inf,
        4, 1, 0, -0.5, 1, 1, 2, 1, 0, 0.16666666666666666, 1, 2, 2, 1, 0,
        -0.08333333333333333, 1, 3, 2, 1, 0
    ]

    w2 = WaveVStack.fromlist(l)
    assert isinstance(w2, WaveVStack)
    assert w2.wlist == w.wlist


def test_op():
    t = np.linspace(-10, 10, 1001)

    wlist = [cos(1), sin(2), gaussian(3), poly([1, -1 / 2, 1 / 6, -1 / 12])]
    w1 = zero()
    for w in wlist:
        w1 += w
    w2 = WaveVStack(wlist)

    wav1 = w1 + sin(2)
    wav2 = w2 + sin(2)
    assert isinstance(wav2, WaveVStack)
    assert np.allclose(wav1(t), wav2(t))
    wav1 = w1 - sin(2)
    wav2 = w2 - sin(2)
    assert isinstance(wav2, WaveVStack)
    assert np.allclose(wav1(t), wav2(t))
    wav1 = w1 * sin(2) + 3
    wav2 = w2 * sin(2) + 3
    assert np.allclose(wav1(t), wav2(t))
    wav1 = w1 / 2
    wav2 = w2 / 2
    assert np.allclose(wav1(t), wav2(t))


def test_shift():
    t = np.linspace(-10, 10, 1001)

    wlist = [cos(1), sin(2), gaussian(3), poly([1, -1 / 2, 1 / 6, -1 / 12])]
    w1 = zero()
    for w in wlist:
        w1 += w
    w2 = WaveVStack(wlist)

    wav1 = w1 >> 0.6
    wav2 = w2 >> 0.6
    assert isinstance(wav2, WaveVStack)
    assert np.allclose(wav1(t), wav2(t))

    wav1 = w1 << 1.4
    wav2 = w2 << 1.4
    assert isinstance(wav2, WaveVStack)
    assert np.allclose(wav1(t), wav2(t))


def test_simplify():
    w1 = zero()
    w2 = []
    assert w1 == WaveVStack(w2).simplify()

    for freq in np.linspace(6.1, 6.5, 11) * 1e9:
        pulse = square(1e-6) >> 95e-6
        w1 += pulse * cos(2 * pi * freq)
        w2.append(pulse * cos(2 * pi * freq))
        assert w1 == WaveVStack(w2).simplify()
    assert w1 == WaveVStack(w2).simplify()

    for freq in np.linspace(6.1, 6.5, 3) * 1e9:
        pulse = square(1e-6) >> (95e-6 + np.random.randn() * 1e-9)
        w1 += pulse * cos(2 * pi * freq)
        w2.append(pulse * cos(2 * pi * freq))
        assert w1 == WaveVStack(w2).simplify()
    w1 += cos(2 * pi * freq * 0.9)
    w2.append(cos(2 * pi * freq * 0.9))
    assert w1 == WaveVStack(w2).simplify()


def test_filters():
    sample_rate = 1000

    b, a = butter(3, 4.0, 'lowpass', fs=sample_rate)
    init_y = 0
    zi = lfiltic(b, a, [init_y])

    t = np.linspace(-1, 1, 2000, endpoint=False)

    wav = WaveVStack([step(0) << 0.5, -step(0)])
    wav.sample_rate = sample_rate
    wav.start = -1
    wav.stop = 1
    wav.filters = (tf2sos(b, a), init_y)

    points = lfilter(b,
                     a,
                     np.heaviside(t + 0.5, 1) - np.heaviside(t, 1),
                     zi=zi)[0]

    assert np.allclose(wav.sample(), points, atol=1e-6)

    l = wav.tolist()
    wav2 = WaveVStack.fromlist(l)
    assert np.allclose(wav2.sample(), points, atol=1e-6)
