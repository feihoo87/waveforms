from collections import defaultdict

import numpy as np


def linFit(x, y):
    """use less memory than np.polyfit"""
    x, y = np.asarray(x), np.asarray(y)
    xm, ym = x.mean(), y.mean()
    N = len(x)
    a = (np.sum(x * y) - N * xm * ym) / ((x**2).sum() - N * xm * xm)
    b = ym - a * xm
    return np.array([a, b])


def fitCircle(x, y):
    u, v = x - x.mean(), y - y.mean()
    Suuu, Svvv = np.sum(u**3), np.sum(v**3)
    Suu, Svv = np.sum(u**2), np.sum(v**2)
    Suv = np.sum(u * v)
    Suuv, Suvv = np.sum(u**2 * v), np.sum(u * v**2)
    uc = (Suuv * Suv - Suuu * Svv - Suvv * Svv + Suv * Svvv)
    vc = (-Suu * Suuv + Suuu * Suv + Suv * Suvv - Suu * Svvv)
    uc /= 2 * (Suv**2 - Suu * Svv)
    vc /= 2 * (Suv**2 - Suu * Svv)
    xc, yc = uc + x.mean(), vc + y.mean()
    R = np.sqrt(np.mean((x - xc)**2 + (y - yc)**2))
    return xc, yc, R


def fitCrossPoint(x1, y1, x2, y2):
    a1, b1 = linFit(x1, y1)
    a2, b2 = linFit(x2, y2)
    return (b2 - b1) / (a1 - a2), (a1 * b2 - a2 * b1) / (a1 - a2)


def fitPole(x, y):
    a, b, c = np.polyfit(x, y, 2)
    return -0.5 * b / a, c - 0.25 * b**2 / a


def goodnessOfFit(pnum, ydata, fvec, sigma=None):
    """
    拟合优度

    Args:
        pnum (int): 拟合参数的个数
        ydata (1d-array): 原始数据
        fvec (1d-array): 残差
        sigma (1d-array, optional): 标准差. Defaults to None.

    Returns:
        tuple: (SSE, R_square, Adj_R_sq, RMSE)
    """
    n = len(ydata)
    dfe = n - pnum
    SSE = (fvec**2).sum()
    if sigma is not None:
        SST = (((ydata - ydata.mean()) / sigma)**2).sum()
    else:
        std = np.sqrt(1.0 * n / dfe * (fvec**2).mean())
        SST = (((ydata - ydata.mean()) / std)**2).sum()
    RMSE = np.sqrt(SSE / dfe)
    R_square = 1 - SSE / SST
    Adj_R_sq = 1 - (SSE / (dfe - 1)) / (SST / (n - 1))

    return SSE, R_square, Adj_R_sq, RMSE


def countState(state):
    ret = defaultdict(lambda: 0)
    for s in state:
        ret[tuple(s)] += 1
    return dict(ret)


def count_to_diag(count, shape=None):
    state = list(count.keys())
    if shape is None:
        shape = (2, ) * len(state[0])
    n = np.asarray(list(count.values()))
    p = n / np.sum(n)
    state = np.ravel_multi_index(np.asarray(state).T, shape)
    ret = np.zeros(shape).reshape(-1)
    ret[state] = p
    return ret


def _atleast_type(a, dtype):
    if dtype == np.double and a.dtype.type == np.int8:
        return a.astype(np.double)
    elif dtype == complex and a.dtype.type in [np.int8, np.double]:
        return a.astype(complex)
    else:
        return a


def classifyData(data, measure_gates, avg=False):
    assert data.shape[-1] == len(
        measure_gates), 'number of qubits must equal to the size of last axis'

    ret = np.zeros_like(data, dtype=np.int8)

    for i, g in enumerate(measure_gates):
        signal = g['params'].get('signal', 'state')
        thr = g['params'].get('threshold', 0)
        phi = g['params'].get('phi', 0)

        if signal == 'state':
            ret[..., i] = (data[..., i] * np.exp(-1j * phi)).real > thr
        elif signal == 'amp':
            ret = _atleast_type(ret, np.double)
            ret[..., i] = (data[..., i] * np.exp(-1j * phi)).real
        if signal == 'raw':
            ret = _atleast_type(ret, complex)
            ret[..., i] = data[..., i]
        elif signal == 'real':
            ret = _atleast_type(ret, np.double)
            ret[..., i] = data[..., i].real
        elif signal == 'imag':
            ret = _atleast_type(ret, np.double)
            ret[..., i] = data[..., i].imag
        elif signal == 'abs':
            ret = _atleast_type(ret, np.double)
            ret[..., i] = np.abs(data[..., i])
        else:
            pass
    if avg:
        ret = ret.mean(axis=-2)
    return ret
