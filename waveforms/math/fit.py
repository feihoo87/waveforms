import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import erf
from sklearn import svm


def lin_fit(x, y):
    """use less memory than np.polyfit"""
    x, y = np.asarray(x), np.asarray(y)
    xm, ym = x.mean(), y.mean()
    N = len(x)
    a = (np.sum(x * y) - N * xm * ym) / ((x**2).sum() - N * xm * xm)
    b = ym - a * xm
    return np.array([a, b])


def fit_circle(x, y):
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


def fit_cross_point(x1, y1, x2, y2):
    a1, b1 = lin_fit(x1, y1)
    a2, b2 = lin_fit(x2, y2)
    return (b2 - b1) / (a1 - a2), (a1 * b2 - a2 * b1) / (a1 - a2)


def fit_pole(x, y):
    a, b, c = np.polyfit(x, y, 2)
    return -0.5 * b / a, c - 0.25 * b**2 / a


def goodness_of_fit(pnum, ydata, fvec, sigma=None):
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


def count_state(state):
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


__classify_methods = {}


def install_classify_method(method: str, func: callable):
    __classify_methods[method] = func


def uninstall_classify_method(method: str):
    if method in __classify_methods:
        del __classify_methods[method]


def classify(data, method, params):
    if method in __classify_methods:
        return __classify_methods[method](data, params)
    else:
        raise ValueError("method not found")


def default_classify(data, params):
    """
    默认的分类方法
    """
    thr = params.get('threshold', 0)
    phi = params.get('phi', 0)
    return (data * np.exp(-1j * phi)).real > thr


install_classify_method("state", default_classify)


def classify_data(data, measure_gates, avg=False):
    assert data.shape[-1] == len(
        measure_gates), 'number of qubits must equal to the size of last axis'

    ret = np.zeros_like(data, dtype=np.int8)

    for i, g in enumerate(measure_gates):
        signal = g['params'].get('signal', 'state')

        if signal in __classify_methods:
            ret[..., i] = classify(data[..., i], signal, g['params'])
        elif signal == 'amp':
            phi = g['params'].get('phi', 0)
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


def cdf(t, data):
    data.sort()
    x = data
    y = np.linspace(0, 1, len(data))
    if t is None:
        return x, y
    else:
        return np.interp(t, x, y, left=0, right=1)


def gaussian_pdf(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 *
                                                     ((x - mu) / sigma)**2)


def gaussian_cdf(x, mu, sigma):
    return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))


def mult_gaussian_pdf(x, mu, sigma, amp):
    amp /= np.sum(amp)
    ret = np.zeros_like(x)
    for i in range(len(mu)):
        ret += amp[i] * gaussian_pdf(x, mu[i], sigma[i])
    return ret


def mult_gaussian_cdf(x, mu, sigma, amp):
    amp /= np.sum(amp)
    ret = np.zeros_like(x)
    for i in range(len(mu)):
        ret += amp[i] * gaussian_cdf(x, mu[i], sigma[i])
    return ret


def fit_readout_distribution(s0, s1):
    def loss(params, s0, s1):
        c0, c1, r0, r1, p0, p1 = params
        x0, y0 = cdf(None, s0)
        x1, y1 = cdf(None, s1)
        Y0 = mult_gaussian_cdf(x0, [c0, c1], [r0, r1], [p0, 1 - p0])
        Y1 = mult_gaussian_cdf(x1, [c0, c1], [r0, r1], [p1, 1 - p1])

        return np.sum((Y0 - y0)**2 + (Y1 - y1)**2)

    res = minimize(loss,
                   [s0.mean(), s1.mean(),
                    s0.std(), s1.std(), 1, 0],
                   args=(s0, s1),
                   bounds=[(None, None), (None, None), (0, None), (0, None),
                           (0, 1), (0, 1)])

    return res


def get_threshold_info(s0, s1):
    s0, s1 = np.asarray(s0), np.asarray(s1)

    data = np.hstack([s0, s1])
    scale = 0.2 * np.abs(data).max()
    data /= scale
    target = np.hstack([np.zeros_like(s0), np.ones_like(s1)])
    X = np.c_[np.real(data), np.imag(data)]
    clf = svm.LinearSVC()
    clf.fit(X, target)
    A, B, C = clf.coef_[0, 0], clf.coef_[0, 1], clf.intercept_[0]
    phi = np.arctan2(B, A)
    #thr = -scale * C / np.sqrt(A**2 + B**2)

    re0 = (s0 * np.exp(-1j * phi)).real
    re1 = (s1 * np.exp(-1j * phi)).real
    im0 = (s0 * np.exp(-1j * phi)).imag
    im1 = (s1 * np.exp(-1j * phi)).imag

    fit_readout_distribution(re0, re1)

    x = np.unique(np.hstack([re0, re1]))
    x.sort()
    a = cdf(x, re0)
    b = cdf(x, re1)
    c = a - b

    visibility = c.max()
    thr = x[c == visibility].mean()

    c0, a0, b0 = np.mean(s0), np.std(re0), np.std(im0)
    c1, a1, b1 = np.mean(s1), np.std(re1), np.std(im1)

    params_r = fit_readout_distribution(re0, re1).x
    params_i = fit_readout_distribution(im0, im1).x

    return {
        'threshold': thr,
        'phi': phi,
        'visibility': (visibility, cdf(thr, re0), cdf(thr, re1)),
        'signal': (re0, re1),
        'idle': (im0, im1),
        'center': (c0, c1),
        'params': (params_r, params_i),
        'std': (a0, b0, a1, b1),
        'cdf': (x, a, b, c)
    }


def getThresholdInfo(s0, s1):
    warnings.warn('getThresholdInfo is deprecated, use get_threshold_info',
                  DeprecationWarning, 2)
    return get_threshold_info(s0, s1)


def classifyData(data, measure_gates, avg=False):
    warnings.warn('classifyData is deprecated, use classify_data',
                  DeprecationWarning, 2)
    return classify_data(data, measure_gates, avg=avg)


def countState(state):
    warnings.warn('countState is deprecated, use count_state',
                  DeprecationWarning, 2)
    return count_state(state)