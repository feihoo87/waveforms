import numpy as np

EPS = 1e-17


def _cmp(x, goal=0, eps=EPS):
    return np.where(x - goal > eps, 1, 0) + np.where(goal - x > eps, -1, 0)


def _point_det(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a.real * b.imag - a.imag * b.real


def point_on_segment(points: np.ndarray,
                     a: np.ndarray,
                     b: np.ndarray,
                     eps: float = EPS) -> np.ndarray:
    """
    points: point array, 1d of complex
    a, b: endpoints of a segment
    """

    return (_cmp(_point_det(points - a, b - a), 0, eps)
            == 0) * (_cmp(_point_det(points - a, points - b), 0, eps) <= 0)


def point_in_polygon(points: np.ndarray,
                     polygon: np.ndarray,
                     eps: float = EPS) -> np.ndarray:
    """
    points: point array, 1d of complex
    polygon: vertex of a polygon, 1d of complex
    """

    ans = np.zeros_like(points)
    N = polygon.shape[0]

    for i in range(N):
        ans += point_on_segment(points, polygon[i], polygon[(i + 1) % N], eps)
        k = _cmp(
            _point_det(polygon[(i + 1) % N] - polygon[i], points - polygon[i]),
            0, eps)
        d1 = _cmp((polygon[i] - points).imag, 0, eps)
        d2 = _cmp((polygon[(i + 1) % N] - points).imag, 0, eps)
        ans += np.where((k > 0) * (d1 <= 0) * (d2 > 0), 1, 0) + np.where(
            (k < 0) * (d2 <= 0) * (d1 > 0), -1, 0)

    return ans > 0


def point_in_ellipse(data, c0, a, b, phi):
    """
    data: point array, 1d of complex
    c0 (complex): center of ellipse
    a, b (float): major and minor axis of ellipse
    phi (float): rotation angle of ellipse
    """
    data = data - c0
    data *= np.exp(-1j * phi)
    return (data.real / a)**2 + (data.imag / b)**2 < 1


def point_on_ellipse(data, c0, a, b, phi, eps=EPS):
    """
    data: point array, 1d of complex
    c0 (complex): center of ellipse
    a, b (float): major and minor axis of ellipse
    phi (float): rotation angle of ellipse
    """
    data = data - c0
    data *= np.exp(-1j * phi)
    x = (data.real / a)**2 + (data.imag / b)**2
    return (1.0 - eps < x) * (x < 1.0 + eps)


def point_out_ellipse(data, c0, a, b, phi):
    """
    data: point array, 1d of complex
    c0 (complex): center of ellipse
    a, b (float): major and minor axis of ellipse
    phi (float): rotation angle of ellipse
    """
    data = data - c0
    data *= np.exp(-1j * phi)
    return (data.real / a)**2 + (data.imag / b)**2 > 1
