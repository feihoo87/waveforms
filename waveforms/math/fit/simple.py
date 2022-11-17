import numpy as np


def lin_fit(x, y, axis=None):
    """use less memory than np.polyfit"""
    x, y = np.asarray(x), np.asarray(y)
    xm, ym = x.mean(axis=axis), y.mean(axis=axis)
    N = len(x)
    a = (np.sum(x * y, axis=axis) - N * xm * ym) / (
        (x**2).sum(axis=axis) - N * xm * xm)
    b = ym - a * xm
    return np.array([a, b])


def poly_fit(x_data, y_data, degree=20, data_filter=None):
    try:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    except ImportError:
        raise ImportError('scikit-learn is required for poly_fit')

    poly = PolynomialFeatures(degree=degree)
    poly.fit(x_data.reshape((-1, 1)))
    x_train = poly.transform(x_data.reshape((-1, 1)))

    model = Pipeline([
        ('sca', StandardScaler()),
        ('ridge', Ridge(solver='cholesky')),
    ])

    model.fit(x_train, y_data)
    y_fit = model.predict(x_train)

    # select data
    for weight in [(0.01, 1), (30, 0.8), (30, 0.9)]:
        err = (y_fit - y_data)**2
        thr = np.mean(err) * weight[0] + np.median(err) * weight[1]
        mask = err < thr
        model.fit(x_train[mask], np.array(y_data)[mask])
        y_fit = model.predict(x_train)

    x, y = x_data[mask].reshape(-1), np.array(y_data)[mask]
    xx = np.linspace(x[0], x[-1], 5001)
    poly = PolynomialFeatures(degree=degree)
    poly.fit(xx.reshape((-1, 1)))
    xxx = poly.transform(xx.reshape((-1, 1)))
    a = np.polyfit(xx, model.predict(xxx), degree)

    return a, [np.min(x_data), np.max(x_data)], mask


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


def find_cross_point(z1, z2, z3, z4):
    v1 = z2 - z1
    v2 = z4 - z3
    v3 = z3 - z1
    a = (v1.real * v2.imag - v1.imag * v2.real)
    t1 = (v3.real * v2.imag - v3.imag * v2.real) / a
    t2 = (v3.real * v1.imag - v3.imag * v1.real) / a

    return z1 + t1 * v1, t1, t2


def fit_pole(x, y):
    a, b, c = np.polyfit(x, y, 2)
    return -0.5 * b / a, c - 0.25 * b**2 / a


def fit_cosine(data, repeat=1, weight=None, x=None):
    """
    Find the amplitude and phase of the data.
    solve equations:
        data[i] = offset + R * cos(phi + i * repeat * 2 * pi / N)
        where i = 0, 1, ..., N-1 and N = len(data)

    Args:
        data (array): data
        repeat (int, optional): Number of cycles. Defaults to 1.
        weight (array, optional): weight. Defaults to None.
    
    Returns:
        tuple: (R, offset, phi)
    
    Examples:
        >>> import numpy as np
        >>> np.random.seed(1234)
        >>> N = 100
        >>> x = 2 * np.pi * np.linspace(0, 1, N, endpoint=False)
        >>> y = 1.2 * np.cos(x - 0.9) + 0.4 + 0.01*np.random.randn(N)
        >>> R, offset, phi = fit_cosine(y)
        >>> R, offset, phi
        (1.20002782555664, 0.4003511228312543, -0.9002458177749354)
    """
    data = np.asarray(data)
    N = data.shape[0]
    if x is None:
        x = np.linspace(0, 2 * np.pi * repeat, N, endpoint=False)
    else:
        weight = np.ones_like(x)
    if weight is None:
        offset = data.mean(axis=0)
        data = data - offset
        e = np.sum(np.moveaxis(data, 0, -1) * np.cos(x), axis=-1)
        f = np.sum(np.moveaxis(data, 0, -1) * np.sin(x), axis=-1)
        R = 2 * np.sqrt(e**2 + f**2) / N
        phi = np.arctan2(-f, e)
    else:
        if weight.ndim == 1:
            assert weight.shape[0] == N, "len(weight) != data.shape[0]"
        elif weight.ndim == 2:
            assert weight.shape == data.shape, "weight.shape != data.shape"
        else:
            raise ValueError("weight must be 1d or 2d array")
        weight = weight / weight.sum(axis=0)
        y = np.sum(np.moveaxis(data, 0, -1) * np.moveaxis(weight, 0, -1),
                   axis=-1)

        a = np.sum(np.moveaxis(weight, 0, -1) * np.cos(x), axis=-1)
        b = np.sum(np.moveaxis(weight, 0, -1) * np.sin(x), axis=-1)
        c = np.sum(np.moveaxis(weight, 0, -1) * np.cos(2 * x), axis=-1)
        d = np.sum(np.moveaxis(weight, 0, -1) * np.sin(2 * x), axis=-1)
        e = np.sum(np.moveaxis(data, 0, -1) * np.moveaxis(weight, 0, -1) *
                   np.cos(x),
                   axis=-1)
        f = np.sum(np.moveaxis(data, 0, -1) * np.moveaxis(weight, 0, -1) *
                   np.sin(x),
                   axis=-1)
        offset = (2 * a * c * e - 2 * a * e + 2 * b * d * e - 2 * b * f -
                  2 * b * c * f + 2 * a * d * f + y - c**2 * y -
                  d**2 * y) / (1 - 2 * a**2 - 2 * b**2 + 2 * a**2 * c -
                               2 * b**2 * c - c**2 + 4 * a * b * d - d**2)
        data = data - offset
        e = np.sum(np.moveaxis(data, 0, -1) * np.moveaxis(weight, 0, -1) *
                   np.cos(x),
                   axis=-1)
        f = np.sum(np.moveaxis(data, 0, -1) * np.moveaxis(weight, 0, -1) *
                   np.sin(x),
                   axis=-1)
        R = (2 * np.sqrt(((c - 1)**2 + d**2) * e**2 + (
            (c + 1)**2 + d**2) * f**2 - 4 * d * e * f)) / (c**2 + d**2 - 1)
        phi = np.arctan2(e - c * e - d * f, f + c * f - d * e) + np.pi / 2

    return R, offset, phi


def complex_amp_to_real(s, axis=None):
    k, _ = lin_fit(s.real, s.imag, axis=axis)
    phi = np.arctan(k)
    return np.real(s * np.exp(-1j * phi))


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
