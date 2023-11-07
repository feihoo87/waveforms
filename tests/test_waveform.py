import numpy as np
import scipy.special as special
from scipy.signal import butter, lfilter, lfiltic, tf2sos

from waveforms import *


def test_waveform():
    t = np.linspace(-10, 10, 1001)

    wav = cos(1)
    assert np.allclose(wav(t), np.cos(t), atol=1e-04)

    wav.start = -10
    wav.stop = 10.02
    wav.sample_rate = 50
    assert np.allclose(wav.sample(), np.cos(t), atol=1e-04)

    wav = sin(1)
    assert np.allclose(wav(t), np.sin(t), atol=1e-04)

    width = 2
    wav = gaussian(width)
    std_sq2 = width / (4 * np.sqrt(np.log(2)))
    assert np.allclose(wav(t), np.exp(-(t / std_sq2)**2), atol=5e-03)

    wav = poly([1, -1 / 2, 1 / 6, -1 / 12])
    assert np.allclose(wav(t), np.poly1d([-1 / 12, 1 / 6, -1 / 2, 1])(t))

    sample_rate = 4e9
    width = 20e-9
    time_line = np.linspace(0, width * 100, int(width * 100 * sample_rate))
    wave = square(width) >> (width * 2)
    points = wave(time_line)
    assert isinstance(points, np.ndarray)


def test_tolist():
    pulse = gaussian(10) >> 5
    pulse += gaussian(10) >> 50
    pulse = pulse * cos(200)

    l = pulse.tolist()
    assert l == [
        np.inf, -np.inf, None, None, None, None, 5, -2.5, 0, 12.5, 1, 1.0, 2,
        1, 3, 2, 3.0028060219661246, 5, 1, 3, 4, 200, 0.0, 42.5, 0, 57.5, 1,
        1.0, 2, 1, 3, 2, 3.0028060219661246, 50, 1, 3, 4, 200, 0.0, np.inf, 0
    ]

    assert Waveform.fromlist(l) == pulse


def test_totree():
    pulse = gaussian(10) >> 5
    pulse += gaussian(10) >> 50
    pulse = pulse * cos(200)

    t = pulse.totree()
    assert t == ((np.inf, -np.inf, None, None, None, None),
                 ((-2.5, ()), (12.5, ((1.0, ((1, (2, 3.0028060219661246, 5)),
                                             (1, (4, 200, 0.0)))), )),
                  (42.5, ()), (57.5, ((1.0, ((1, (2, 3.0028060219661246, 50)),
                                             (1, (4, 200, 0.0)))), )), (np.inf,
                                                                        ())))
    assert Waveform.fromtree(t) == pulse


def test_op():
    t = np.linspace(-10, 10, 1001)

    wav = cos(1) + sin(2)
    assert np.allclose(wav(t), np.cos(t) + np.sin(2 * t))
    wav = cos(1) - sin(2)
    assert np.allclose(wav(t), np.cos(t) - np.sin(2 * t))
    wav = cos(1) * sin(2)
    assert np.allclose(wav(t), np.cos(t) * np.sin(2 * t))
    wav = cos(1) / 2
    assert np.allclose(wav(t), np.cos(t) / 2)


def test_simplify():
    t = np.linspace(-10, 10, 1001)
    wav = cos(1) * sin(2) * cos(3, 4)
    wav2 = wav.simplify()

    assert np.allclose(wav(t), np.cos(t) * np.sin(2 * t) * np.cos(3 * t + 4))
    assert np.allclose(wav2(t), np.cos(t) * np.sin(2 * t) * np.cos(3 * t + 4))


def test_simplify2():
    t = np.linspace(-2, 2, 1001)
    wav = 1j * (cos(9) >> 1) + 1 * (cos(9) >> 2) - 1j * (cos(9) >> 3)

    assert np.allclose(wav(t), wav.simplify()(t))


def test_simplify3():
    t = np.linspace(-2, 2, 1001)

    wav = 2 * (exp(1.01 + 22j)**2 << 1) * exp(1.01 + 22j)
    wav2 = wav.simplify()
    points = 2 * np.exp((1.01 + 22j) * (t + 1))**2 * np.exp((1.01 + 22j) * t)

    assert np.allclose(wav(t), points)
    assert np.allclose(wav2(t), points)


def test_shift():
    t = np.linspace(-10, 10, 1001)
    width = 2
    wav = gaussian(width) >> 3
    std_sq2 = width / (4 * np.sqrt(np.log(2)))
    assert np.allclose(wav(t), np.exp(-((t - 3) / std_sq2)**2), atol=5e-03)


def test_chirp():
    t = np.linspace(0, 10, 1000, endpoint=False)

    def _chirp(t, f0, f1, T, phi0=0, type='linear'):
        if type == 'linear':
            return np.sin(phi0 + 2 * np.pi * ((f1 - f0) /
                                              (2 * T) * t**2 + f0 * t))
        elif type == 'exponential':
            return np.sin(phi0 + 2 * np.pi * f0 * T *
                          ((f1 / f0)**(t / T) - 1) / np.log((f1 / f0)))
        elif type == 'hyperbolic':
            return np.sin(phi0 - 2 * np.pi * f0 * f1 * T /
                          (f1 - f0) * np.log(1 - (f1 - f0) * t / (f1 * T)))
        else:
            raise ValueError(f'Unknow type {type}')

    wav1 = chirp(1, 2, 10, 4, 'linear')
    wav2 = chirp(1, 2, 10, 4, 'exponential')
    wav3 = chirp(1, 2, 10, 4, 'hyperbolic')

    assert np.allclose(wav1(t), _chirp(t, 1, 2, 10, 4, 'linear'))
    assert np.allclose(wav2(t), _chirp(t, 1, 2, 10, 4, 'exponential'))
    assert np.allclose(wav3(t), _chirp(t, 1, 2, 10, 4, 'hyperbolic'))


def test_parser():
    w1 = (gaussian(10) <<
          100) + square(20, edge=5, type='linear') * cos(2 * pi * 23.1)
    w2 = wave_eval(
        "(gaussian(10) << 100) + square(20, edge=5, type='linear') * cos(2*pi*23.1)"
    )
    w3 = wave_eval(
        "((gaussian(10) << 50) + ((square(20, 5, type='linear') * cos(2*pi*23.1)) >> 50)) << 50"
    )
    w4 = wave_eval(
        "(gaussian(10) << 100) + square(20, 5, 'linear') * cos(2*pi*23.1)")
    assert w1 == w2
    assert w1 == w3
    assert w1 == w4

    w1 = poly([1, -1 / 2, 1 / 6, -1 / 12])
    w2 = wave_eval("poly([1, -1/2, 1/6, -1/12])")
    w3 = wave_eval("poly((1, -1/2, 1/6, -1/12))")

    assert w1 == w2
    assert w1 == w3


def test_filters():
    sample_rate = 1000

    b, a = butter(3, 4.0, 'lowpass', fs=sample_rate)
    init_y = 0
    zi = lfiltic(b, a, [init_y])

    t = np.linspace(-1, 1, 2000, endpoint=False)

    wav = step(0)
    wav.sample_rate = sample_rate
    wav.start = -1
    wav.stop = 1
    wav.filters = (tf2sos(b, a), init_y)

    points = lfilter(b, a, np.heaviside(t, 1), zi=zi)[0]

    assert np.allclose(wav.sample(), points)

    l = wav.tolist()
    wav2 = Waveform.fromlist(l)
    assert np.allclose(wav2.sample(), points)

    d = wav.totree()
    wav3 = Waveform.fromtree(d)
    assert np.allclose(wav3.sample(), points)
