import math

import numpy as np

from .waveform import (COS, NDIGITS, Waveform, _basic_wave, _format_DRAG,
                       _zero, inf, pi, registerBaseFunc)


def B_series_mat(bs: np.ndarray):
    aa = np.zeros([len(bs) + 1, 2, 2])
    aa[0] = np.array([np.identity(2)])
    for b in bs:
        bb = np.array([[0, b], [-b, 0]])
        aa[1:] = aa[1:] + aa[:-1] @ bb
    return aa


def _derivatives_sin_m(m: int, n: int, a: float = 1):
    aa = np.zeros([n + 1, m + 1])
    aa[0, m] = 1
    for i in range(1, n + 1):
        if i % 2:
            aa[i][:-1] = aa[i - 1][1:] * np.arange(1, m + 1) * a
        else:
            aa[i][:-2] = aa[i - 2][2:] * np.arange(1, m) * np.arange(2, m + 1)
            aa[i] = aa[i] - aa[i - 2] * np.arange(m + 1)**2
            aa[i] = aa[i] * (a**2)
    return aa


def _drag_omega_sin_n(t: np.ndarray,
                      t0: float,
                      width: float,
                      delta: float,
                      block_freq: np.ndarray = None,
                      plateau: float = 0):
    bs, m = [], 2
    if isinstance(block_freq, float):
        block_freq = (block_freq, )
    if block_freq is not None:
        #         bs = 0.5/width/(np.array(block_freq)-delta)
        bs = 1 / np.pi / 2 / (np.array(block_freq) - delta)
        m = max((len(bs) + 2) >> 1 << 1, m)

    B_mat = B_series_mat(bs)
    ps = np.arange(m + 1)
    o = np.pi / width
    A_mat = _derivatives_sin_m(m, len(bs), o)
    bbb = (np.piecewise(t, [
        t <= t0 + width / 2, (t > t0 + width / 2) *
        (t < t0 + plateau + width / 2), t >= t0 + plateau + width / 2
    ], [
        lambda x, o=o, t0=t0: np.sin(o * (x - t0)), 0,
        lambda x, o=o, t0=t0, plateau=plateau: np.sin(o * (x - t0 - plateau))
    ]))**(ps.reshape([-1, 1]))
    bbb[1::2] = bbb[1::2] * np.piecewise(t, [
        t <= t0 + width / 2, (t > t0 + width / 2) *
        (t < t0 + plateau + width / 2), t >= t0 + plateau + width / 2
    ], [
        lambda x, o=o, t0=t0: np.cos(o * (x - t0)), 0,
        lambda x, o=o, t0=t0, plateau=plateau: np.cos(o * (x - t0 - plateau))
    ])
    bbb = A_mat @ bbb

    bbbb = np.ones([m + 1])
    bbbb[1::2] = 0
    bbbb = A_mat @ bbbb
    coe = np.einsum('ijk,ki->j', B_mat, np.array([bbbb, np.zeros_like(bbbb)]))
    coeff = np.sqrt(np.sum(np.abs(coe)**2))

    ccc = np.array([bbb, np.zeros_like(bbb)])
    ccc[0, 0][(t > t0 + width / 2) * (t < t0 + plateau + width / 2)] = 1
    return np.einsum('ijk,kim->jm', B_mat, ccc) / coeff


def _derivatives_x_m_poly_a(f: np.ndarray, x: float):
    fff = np.copy(f)
    fff[0] -= 1
    m = f.shape[0]
    C_mat = np.zeros([m, m])
    for n in range(0, m):
        for l in range(0, m):
            C_mat[n, l] += (x**(m + l - n)) * math.factorial(
                m + l) / math.factorial(m + l - n)
    from scipy.linalg import inv
    C_inv = inv(C_mat)
    return np.poly1d([*np.flip(C_inv @ fff), *np.zeros_like(f[:-1]), 1])


def _drag_omega_sin_x_n(t: np.ndarray,
                        t0: float,
                        width: float,
                        delta: float,
                        block_freq: np.ndarray = None,
                        plateau: float = 0,
                        tab: float = 0.618):
    bs, m = [], 2
    if isinstance(block_freq, float):
        block_freq = (block_freq, )
    if block_freq is not None:
        bs = 1 / np.pi / 2 / (np.array(block_freq) - delta)
        m = max((len(bs) + 2) >> 1 << 1, m)

    B_mat = B_series_mat(bs)
    ps = np.arange(m + 1)
    o = np.pi / width
    A_mat = _derivatives_sin_m(m, len(bs), o)
    bbb = (np.piecewise(t, [
        t <= t0 + width / 2, (t > t0 + width / 2) *
        (t < t0 + plateau + width / 2), t >= t0 + plateau + width / 2
    ], [
        lambda x, o=o, t0=t0: np.sin(o * (x - t0)), 0,
        lambda x, o=o, t0=t0, plateau=plateau: np.sin(o * (x - t0 - plateau))
    ]))**(ps.reshape([-1, 1]))
    bbb[1::2] = bbb[1::2] * np.piecewise(t, [
        t <= t0 + width / 2, (t > t0 + width / 2) *
        (t < t0 + plateau + width / 2), t >= t0 + plateau + width / 2
    ], [
        lambda x, o=o, t0=t0: np.cos(o * (x - t0)), 0,
        lambda x, o=o, t0=t0, plateau=plateau: np.cos(o * (x - t0 - plateau))
    ])
    bbb = A_mat @ bbb

    tab_x = np.sin(o * (1 - tab) * width / 2)**np.arange(m + 1)
    tab_x[1::2] = tab_x[1::2] * np.cos(o * (1 - tab) * width / 2)
    tab_x = A_mat @ tab_x
    coeff_as_left = _derivatives_x_m_poly_a(tab_x, -tab * width / 2)

    tab_x = np.sin(o * (1 + tab) * width / 2)**np.arange(m + 1)
    tab_x[1::2] = tab_x[1::2] * np.cos(o * (1 + tab) * width / 2)
    tab_x = A_mat @ tab_x
    coeff_as_right = _derivatives_x_m_poly_a(tab_x, tab * width / 2)

    bbbb = np.ones([m + 1])
    bbbb[1::2] = 0
    bbbb = A_mat @ bbbb
    coe = np.einsum('ijk,ki->j', B_mat, np.array([bbbb, np.zeros_like(bbbb)]))
    coeff = np.sqrt(np.sum(np.abs(coe)**2))

    ccc = np.array([bbb, np.zeros_like(bbb)])
    ccc[0, 0][(t > t0 + width / 2) * (t < t0 + plateau + width / 2)] = 1
    for n in range(0, len(bs) + 1):
        ccc[0, n][(t >= t0 + width / 2 - tab * width / 2) *
                  (t <= t0 + width / 2)] = (np.polyder(
                      coeff_as_left,
                      m=n))(t[(t >= t0 + width / 2 - tab * width / 2) *
                              (t <= t0 + width / 2)] - t0 - width / 2)
        ccc[0, n][
            (t >= t0 + plateau + width / 2) *
            (t <= t0 + plateau + width / 2 + tab * width / 2)] = (np.polyder(
                coeff_as_right,
                m=n))(t[(t >= t0 + plateau + width / 2) *
                        (t <= t0 + plateau + width / 2 + tab * width / 2)] -
                      t0 - plateau - width / 2)
    return np.einsum('ijk,kim->jm', B_mat, ccc)


def _drag_sin(t: np.ndarray,
              t0: float,
              freq: float,
              width: float,
              delta: float,
              block_freq: np.ndarray,
              phase: float,
              plateau: float = 0):

    Omega_x, Omega_y = _drag_omega_sin_n(t=t,
                                         t0=t0,
                                         width=width,
                                         delta=delta,
                                         block_freq=block_freq,
                                         plateau=plateau)
    wt = 2 * np.pi * (freq + delta) * t - (2 * np.pi * delta * t0 + phase)
    return Omega_x * np.cos(wt) + Omega_y * np.sin(wt)


DRAG_SIN = registerBaseFunc(_drag_sin, _format_DRAG)


def drag_sin(freq, width, plateau=0, delta=0, block_freq=None, phase=0, t0=0):
    phase += pi * delta * (width + plateau)
    if isinstance(block_freq, float):
        block_freq = (block_freq, )
    return Waveform(seq=(_zero,
                         _basic_wave(DRAG_SIN, t0, freq, width, delta,
                                     block_freq, phase, plateau), _zero),
                    bounds=(round(t0,
                                  NDIGITS), round(t0 + width + plateau,
                                                  NDIGITS), +inf))


def _drag_sinx(t: np.ndarray,
               t0: float,
               freq: float,
               width: float,
               delta: float,
               block_freq: np.ndarray,
               phase: float,
               plateau: float = 0,
               tab: float = 0.618):

    Omega_x, Omega_y = _drag_omega_sin_x_n(t=t,
                                           t0=t0,
                                           width=width,
                                           delta=delta,
                                           block_freq=block_freq,
                                           plateau=plateau,
                                           tab=tab)
    wt = 2 * np.pi * (freq + delta) * t - (2 * np.pi * delta * t0 + phase)
    return Omega_x * np.cos(wt) + Omega_y * np.sin(wt)


DRAG_SINX = registerBaseFunc(_drag_sinx, _format_DRAG)


def drag_sinx(freq,
              width,
              plateau=0,
              delta=0,
              block_freq=None,
              phase=0,
              t0=0,
              tab=0.618):
    phase += pi * delta * (width + plateau)
    if isinstance(block_freq, float):
        block_freq = (block_freq, )
    return Waveform(seq=(_zero,
                         _basic_wave(DRAG_SINX, t0, freq, width, delta,
                                     block_freq, phase, plateau, tab), _zero),
                    bounds=(round(t0,
                                  NDIGITS), round(t0 + width + plateau,
                                                  NDIGITS), +inf))
