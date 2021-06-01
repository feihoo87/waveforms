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
