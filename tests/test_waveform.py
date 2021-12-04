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

    wav = poly([1, -1 / 2, 1 / 6, -1 / 12])
    assert np.allclose(wav(t), np.poly1d([-1 / 12, 1 / 6, -1 / 2, 1])(t))


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
