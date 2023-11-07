import numpy as np

from waveforms.multy_drag import drag_sin, drag_sinx


def test_drag_sin():
    t0 = 0e-9
    freq = 5e9
    width = 22.22e-9
    delta = np.random.random() * 9.5e6 - 19e6
    plateau = 0
    block_freq = tuple(
        np.concatenate(
            (np.random.random([np.random.randint(4) + 1]) * 100e6 + 20e6,
             -np.random.random([np.random.randint(4) + 1]) * 100e6 - 20e6)))
    I = drag_sin(freq, width, plateau, delta, block_freq, 0, t0)
    Q = drag_sin(freq, width, plateau, delta, block_freq, -np.pi / 2, t0)
    wav = I - 1j * Q
    ttt = np.linspace(t0 - (width + plateau) * 10, t0 + (width + plateau) * 11,
                      1000001)
    for bq in block_freq:
        freq_list = (freq + np.linspace(-0.02e6, 0.02e6, 21) + bq).reshape(
            [1, -1])
        ff = np.exp(-2j * np.pi * freq_list * (ttt.reshape([-1, 1])))
        #         plt.plot(np.abs(wav(ttt)@ff))
        assert np.argmin(np.abs(wav(ttt) @ ff)) == 10

    block_freq = np.random.random() * 100e6 + 20e6

    I = drag_sin(freq, width, plateau, delta, block_freq, 0, t0)
    Q = drag_sin(freq, width, plateau, delta, block_freq, -np.pi / 2, t0)
    wav = I - 1j * Q
    ttt = np.linspace(t0 - (width + plateau) * 10, t0 + (width + plateau) * 11,
                      1000001)
    bq = block_freq
    freq_list = (freq + np.linspace(-0.02e6, 0.02e6, 21) + bq).reshape([1, -1])
    ff = np.exp(-2j * np.pi * freq_list * (ttt.reshape([-1, 1])))
    #     plt.plot(np.abs(wav(ttt)@ff))
    assert np.argmin(np.abs(wav(ttt) @ ff)) == 10


def test_drag_sinx():
    t0 = 0e-9
    freq = 5e9
    width = 22.22e-9
    delta = np.random.random() * 9.5e6 - 19e6
    plateau = 0
    block_freq = tuple(
        np.concatenate(
            (np.random.random([np.random.randint(4) + 1]) * 100e6 + 20e6,
             -np.random.random([np.random.randint(4) + 1]) * 100e6 - 20e6)))
    tab = np.random.random() * 0.8 + 0.2
    I = drag_sinx(freq, width, plateau, delta, block_freq, 0, t0, tab)
    Q = drag_sinx(freq, width, plateau, delta, block_freq, -np.pi / 2, t0, tab)
    wav = I - 1j * Q
    ttt = np.linspace(t0 - (width + plateau) * 10, t0 + (width + plateau) * 11,
                      1000001)
    for bq in block_freq:
        freq_list = (freq + np.linspace(-0.02e6, 0.02e6, 21) + bq).reshape(
            [1, -1])
        ff = np.exp(-2j * np.pi * freq_list * (ttt.reshape([-1, 1])))
        #         plt.plot(np.abs(wav(ttt)@ff))
        assert np.argmin(np.abs(wav(ttt) @ ff)) == 10

    block_freq = np.random.random() * 100e6 + 20e6

    I = drag_sinx(freq, width, plateau, delta, block_freq, 0, t0, tab)
    Q = drag_sinx(freq, width, plateau, delta, block_freq, -np.pi / 2, t0, tab)
    wav = I - 1j * Q
    ttt = np.linspace(t0 - (width + plateau) * 10, t0 + (width + plateau) * 11,
                      1000001)
    bq = block_freq
    freq_list = (freq + np.linspace(-0.02e6, 0.02e6, 21) + bq).reshape([1, -1])
    ff = np.exp(-2j * np.pi * freq_list * (ttt.reshape([-1, 1])))
    #     plt.plot(np.abs(wav(ttt)@ff))
    assert np.argmin(np.abs(wav(ttt) @ ff)) == 10
