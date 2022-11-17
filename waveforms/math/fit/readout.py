import warnings
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import erf, erfinv

from .geo import EPS, point_in_polygon


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


def classify_svm(data, params):
    """
    分类方法：SVM
    """
    raise NotImplementedError
    from sklearn import svm

    clf = svm.SVC(kernel='rbf',
                  gamma=params.get('gamma', 1),
                  C=params.get('C', 1))
    clf.fit(data, data)
    return clf.predict(data)


def classify_kmeans(data, params):
    """
    分类方法：KMeans
    """
    from sklearn.cluster import KMeans

    centers = params.get('centers', None)
    if isinstance(centers, list):
        centers = np.asarray(centers)

    k = params.get('k', None)
    if k is None and centers is not None:
        k = np.asarray(centers).shape[0]
    cur_shape = data.shape

    flatten_init = np.array([np.real(centers), np.imag(centers)]).T

    flatten_data = data.flatten()
    ret_ans = KMeans(n_clusters=k, init=flatten_init).fit_predict(
        np.array([np.real(flatten_data),
                  np.imag(flatten_data)]).T)
    return 2**ret_ans.reshape(cur_shape)


def classify_nearest(data, params):
    """
    分类方法：最近邻
    """
    centers = params.get('centers', None)
    if centers is None:
        raise ValueError("centers not found")
    return 2**np.argmin([np.abs(data - c) for c in centers], axis=0)


def classify_range(data, params):
    """
    分类方法：范围
    """
    centers = params.get('centers', None)
    radians = params.get('radians', None)
    if centers is None:
        raise ValueError("centers not found")
    if radians is None:
        return 2**np.argmin([np.abs(data - c) for c in centers], axis=0)

    ret = np.full_like(data, 0, dtype=int)
    for i, (c, r) in enumerate(zip(centers, radians)):
        ret[np.abs(data - c) <= r] += 2**i
    return ret


def classify_polygon(data, params):
    """
    分类方法: 多边形内
    """
    polygons = params.get('polygons', None)
    eps = params.get('eps', EPS)
    if polygons is None:
        raise ValueError("polygons not found")

    ret = np.full_like(data, 0, dtype=int)
    for i, polygon in enumerate(polygons):
        ret[point_in_polygon(data, polygon, eps)] += 2**i
    return ret


install_classify_method("state", default_classify)
install_classify_method("nearest", classify_nearest)
install_classify_method("range", classify_range)
install_classify_method("kmeans", classify_kmeans)
install_classify_method("polygon", classify_polygon)


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


def gaussian_pdf_2d(z, mu, cov):
    z = z - mu
    v = np.moveaxis(np.array([z.real, z.imag]), 0, -1).reshape(*z.shape, 2, 1)
    vT = np.moveaxis(v, -1, -2)
    m = np.linalg.inv(cov)
    return 1 / (2 * np.pi * np.sqrt(np.linalg.det(cov))) * np.exp(
        -0.5 * vT @ m @ v)[..., 0, 0]


def gaussian_cdf(x, mu, sigma):
    return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))


def gaussian_cdf_inv(y, mu, sigma):
    return np.sqrt(2) * sigma * erfinv(2 * y - 1) + mu


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


def readout_distribution(s, p, c0, c1, cov0, cov1):
    return gaussian_pdf_2d(s, c0, cov0) * p + gaussian_pdf_2d(s, c1,
                                                              cov1) * (1 - p)


def fit_readout_distribution(s0, s1):
    center = 0.5 * (s0.mean() + s1.mean())
    s0, s1 = s0 - center, s1 - center

    scale = np.max([np.abs(s0.mean()), np.abs(s1.mean()), s0.std(), s1.std()])

    s0, s1 = s0 / scale, s1 / scale

    def a_b_phi_2_cov(a, b, phi):
        m = np.array([[np.cos(phi) * a, -np.sin(phi) * b],
                      [np.sin(phi) * a, np.cos(phi) * b]])

        return m @ m.T

    def cov_2_a_b_phi(cov):
        x, y, z = cov[0, 0], cov[1, 1], cov[0, 1]
        a = (x - y) / z
        b = np.sqrt(4 + a**2)
        c = np.sqrt(8 + 2 * a**2 + (2 * a**3 + 8 * a) / b)
        t = -a / 2 - b / 2 + c / 2
        d = 1 - 6 * t**2 + t**4
        phi = np.arctan2(2 * t, 1 - t**2)
        a = np.sqrt(-(-x + 2 * t**2 * x - t**4 * x + 4 * t**2 * y) / d)
        b = np.sqrt(-(4 * t**2 * x - y + 2 * t**2 * y - t**4 * y) / d)
        return a, b, phi

    def loss(params, s0, s1):
        cr0, cr1, rr0, rr1, ci0, ci1, ri0, ri1, p0, p1, phi0, phi1 = params

        c0, c1 = cr0 + 1j * ci0, cr1 + 1j * ci1

        cov0 = a_b_phi_2_cov(rr0, ri0, phi0)
        cov1 = a_b_phi_2_cov(rr1, ri1, phi1)

        eps = 1e-20
        return (
            -np.log(readout_distribution(s0, p0, c0, c1, cov0, cov1) +
                    eps).sum() -
            np.log(readout_distribution(s1, p1, c0, c1, cov0, cov1) +
                   eps).sum())

    res = minimize(loss, [
        s0.real.mean(),
        s1.real.mean(),
        s0.real.std(),
        s1.real.std(),
        s0.imag.mean(),
        s1.imag.mean(),
        s0.imag.std(),
        s1.imag.std(), 1, 0, 0, 0
    ],
                   args=(s0, s1),
                   bounds=[(None, None), (None, None), (1e-6, None),
                           (1e-6, None), (None, None), (None, None),
                           (1e-6, None), (1e-6, None), (0, 1), (0, 1),
                           (0, 2 * np.pi), (0, 2 * np.pi)])

    cr0, cr1, rr0, rr1, ci0, ci1, ri0, ri1, p0, p1, phi0, phi1 = res.x
    c0, c1 = cr0 + 1j * ci0, cr1 + 1j * ci1
    c0, c1 = c0 * scale + center, c1 * scale + center
    cov0 = a_b_phi_2_cov(rr0, ri0, phi0)
    cov1 = a_b_phi_2_cov(rr1, ri1, phi1)

    return (c0, c1, rr0 * scale, rr1 * scale, ri0 * scale, ri1 * scale, p0, p1,
            phi0, phi1, cov0 * scale**2, cov1 * scale**2)


def fit_readout_distribution2(s0, s1):
    center = 0.5 * (s0.mean() + s1.mean())
    s0, s1 = s0 - center, s1 - center

    scale = np.max([np.abs(s0.mean()), np.abs(s1.mean()), s0.std(), s1.std()])

    s0, s1 = s0 / scale, s1 / scale

    def loss(params, s0, s1):
        cr0, cr1, rr0, rr1, ci0, ci1, ri0, ri1, p0, p1 = params

        x0, y0 = cdf(None, s0.real)
        x1, y1 = cdf(None, s1.real)
        x2, y2 = cdf(None, s0.imag)
        x3, y3 = cdf(None, s1.imag)

        Y0 = mult_gaussian_cdf(x0, [cr0, cr1], [rr0, rr1], [p0, 1 - p0])
        Y1 = mult_gaussian_cdf(x1, [cr0, cr1], [rr0, rr1], [p1, 1 - p1])
        Y2 = mult_gaussian_cdf(x2, [ci0, ci1], [ri0, ri1], [p0, 1 - p0])
        Y3 = mult_gaussian_cdf(x3, [ci0, ci1], [ri0, ri1], [p1, 1 - p1])

        return (np.sum((Y0 - y0)**2) + np.sum((Y1 - y1)**2) + np.sum(
            (Y2 - y2)**2) + np.sum((Y3 - y3)**2))

    res = minimize(loss, [
        s0.real.mean(),
        s1.real.mean(),
        s0.real.std(),
        s1.real.std(),
        s0.imag.mean(),
        s1.imag.mean(),
        s0.imag.std(),
        s1.imag.std(), 1, 0
    ],
                   args=(s0, s1),
                   bounds=[(None, None), (None, None), (1e-6, None),
                           (1e-6, None), (None, None), (None, None),
                           (1e-6, None), (1e-6, None), (0, 1), (0, 1)])

    cr0, cr1, rr0, rr1, ci0, ci1, ri0, ri1, p0, p1 = res.x
    c0, c1 = cr0 + 1j * ci0, cr1 + 1j * ci1
    c0, c1 = c0 * scale + center, c1 * scale + center

    return (c0, c1, rr0 * scale, rr1 * scale, ri0 * scale, ri1 * scale, p0, p1,
            0, 0)


def get_threshold_info(s0, s1, thr=None, phi=None):
    from sklearn import svm

    s0, s1 = np.asarray(s0), np.asarray(s1)

    if phi is None:
        data = np.hstack([s0, s1])
        scale = 0.2 * np.abs(data).max()
        data /= scale
        target = np.hstack(
            [np.zeros_like(s0, dtype=float),
             np.ones_like(s1, dtype=float)])
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

    x = np.unique(np.hstack([re0, re1]))
    x.sort()
    a = cdf(x, re0)
    b = cdf(x, re1)
    c = a - b

    visibility = c.max()

    if thr is None:
        thr = x[c == visibility]
        thr = 0.5 * (thr.min() + thr.max())

    (c0, c1, rr0, rr1, ri0, ri1, p0, p1, phi0, phi1, cov0,
     cov1) = fit_readout_distribution(re0 + 1j * im0, re1 + 1j * im1)

    params_r = np.array([c0.real, c1.real, rr0, rr1, p0, p1, phi0])
    params_i = np.array([c0.imag, c1.imag, ri0, ri1, p0, p1, phi1])

    return {
        'threshold': thr,
        'phi': phi,
        'visibility': (visibility, cdf(thr, re0), cdf(thr, re1)),
        'signal': (re0, re1),
        'idle': (im0, im1),
        'center': (c0, c1),
        'params': (params_r, params_i),
        'std': (rr0, ri0, rr1, ri1, cov0, cov1),
        'cdf': (x, a, b, c)
    }


def bayesian_correction(state, correction_matrices, subspace):
    """Apply a correction matrix to a state.

    Args:
        state (np.array, dtype=int): The state to be corrected.
        correction_matrices (np.array): A list of correction matrices.
        subspace (np.array, dtype=int): The basis of subspace.

    Returns:
        np.array: The corrected state.

    Examples:
        >>> state = np.random.randint(2, size = (101, 1024, 4))
        >>> PgPe = np.array([[0.1, 0.8], [0.03, 0.91], [0.02, 0.87], [0.05, 0.9]])
        >>> correction_matrices = np.array(
            [np.array([[Pe, Pe - 1], [-Pg, 1 - Pg]]) / (Pe - Pg) for Pg, Pe in PgPe])
        >>> subspace = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0]])
        >>> result = bayesian_correction(state, correction_matrices, subspace)
        >>> result.shape
        (101, 1024, 5)
    """
    num_qubits = state.shape[-1]
    site_index = np.arange(num_qubits)

    shape = state.shape
    state = state.reshape(-1, num_qubits)

    if len(subspace) < len(state):
        ret = []
        for target_state in subspace:
            A = np.prod(correction_matrices[site_index, target_state, state],
                        axis=-1)
            ret.append(A)
        ret = np.array(ret).T.reshape(shape)
    else:
        ret = []
        for bit_string in state:
            A = np.prod(correction_matrices[site_index, subspace, bit_string],
                        axis=-1)
            ret.append(A)
        ret = np.array(ret).reshape(shape)
    ret = ret.mean(axis=-2)
    return ret


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
