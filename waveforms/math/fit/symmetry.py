import math

import numpy as np
from scipy.interpolate import griddata
from scipy.signal import fftconvolve


def _as_meshgrid(x, y, z):
    """
    Convert x, y, z data to meshgrid.

    Parameters
    ----------
    x : array_like
        x data.
    y : array_like
        y data.
    z : array_like
        z data.

    Returns
    -------
    tuple
        x, y, z meshgrid.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if z is None:
        z = y
        if z.ndim == 2:
            y = np.arange(z.shape[0])
        else:
            y = None
        if not np.allclose(np.diff(np.diff(x)), 0):
            xx = np.linspace(
                x[-1], x[0],
                round((np.max(x) - np.min(x)) / np.percentile(np.diff(x), 10)))
            if y is None:
                y = np.array([0])
                z = np.array([np.interp(xx, x, z)])
            else:
                grid_x, grid_y = np.meshgrid(xx, y)
                x_, y_ = np.meshgrid(x, y)
                points = np.array([x_.ravel(), y_.ravel()]).T
                z = griddata(points,
                             z.ravel(), (grid_x, grid_y),
                             method='linear')
            x = xx
        return x, y, z

    z = np.asarray(z)
    if x.ndim == y.ndim == z.ndim:
        xx = np.unique(x)
        yy = np.unique(y)
        z = griddata(np.vstack([x.ravel(), y.ravel()]).T,
                     z.ravel(),
                     np.meshgrid(xx, yy),
                     method='linear')
        x = xx
        y = yy

    if not np.allclose(np.diff(np.diff(x)), 0):
        xx = np.linspace(
            x[-1], x[0],
            round((np.max(x) - np.min(x)) / np.percentile(np.diff(x), 10)))
        grid_x, grid_y = np.meshgrid(xx, y)
        x_, y_ = np.meshgrid(x, y)
        points = np.array([x_.ravel(), y_.ravel()]).T
        z = griddata(points, z.ravel(), (grid_x, grid_y), method='linear')
        x = xx
    if not np.allclose(np.diff(np.diff(y)), 0):
        yy = np.linspace(
            y[-1], y[0],
            round((np.max(y) - np.min(y)) / np.percentile(np.diff(y), 10)))
        grid_x, grid_y = np.meshgrid(x, yy)
        x_, y_ = np.meshgrid(x, y)
        points = np.array([x_.ravel(), y_.ravel()]).T
        z = griddata(points, z.ravel(), (grid_x, grid_y), method='linear')
        y = yy

    return x, y, z


def find_axis_of_symmetry(x, y, z=None):
    """
    Find axis of symmetry of a curve.

    Parameters
    ----------
    x : array_like
        x data.
    y : array_like
        y data.

    Returns
    -------
    float
        Axis of symmetry.
    """
    x, y, z = _as_meshgrid(x, y, z)

    if z.ndim == 2:
        c = fftconvolve(z, z, mode='same', axes=1).sum(axis=0)
    else:
        c = fftconvolve(z, z, mode='same')
    i = np.argmax(c)
    x_center = len(c) / 2
    axis_of_symmetry_i = (i + x_center) / 2
    index = math.ceil(axis_of_symmetry_i)
    remainder = axis_of_symmetry_i - index
    return x[index] * (1 - remainder) + x[index + 1] * remainder


def find_center_of_symmetry(x, y, z):
    """
    Find center of symmetry of a surface.

    Parameters
    ----------
    x : array_like
        x data.
    y : array_like
        y data.
    z : array_like
        z data.

    Returns
    -------
    tuple
        Center of symmetry.
    """
    x, y, z = _as_meshgrid(x, y, z)

    c = fftconvolve(z, z, mode='same')
    j, i = np.unravel_index(np.argmax(c), c.shape)
    x_center = c.shape[1] / 2
    y_center = c.shape[0] / 2
    center_of_symmetry_i = (i + x_center) / 2
    center_of_symmetry_j = (j + y_center) / 2
    index_i = math.ceil(center_of_symmetry_i)
    remainder_i = center_of_symmetry_i - index_i
    index_j = math.ceil(center_of_symmetry_j)
    remainder_j = center_of_symmetry_j - index_j
    return (x[index_i] * (1 - remainder_i) + x[index_i + 1] * remainder_i,
            y[index_j] * (1 - remainder_j) + y[index_j + 1] * remainder_j)
