import numpy as np

EPS = 1e-17


def _cmp(x, goal=0, eps=EPS):
    return np.where(x - goal > eps, 1, 0) + np.where(goal - x > eps, -1, 0)


def _point_det(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a.real * b.imag - a.imag * b.real


def point_on_segment(a: np.ndarray,
                     s: np.ndarray,
                     t: np.ndarray,
                     eps: float = EPS) -> np.ndarray:
    """
    a: point array, 1d of complex
    b, c: one point and another point of a segment
    """

    return (_cmp(_point_det(a - s, t - s), 0, eps)
            == 0) * (_cmp(_point_det(a - s, a - t), 0, eps) <= 0)


def point_in_polygon(a: np.ndarray,
                     b: np.ndarray,
                     eps: float = EPS) -> np.ndarray:
    """
    a: point array, 1d of complex
    b: vertex of a polygon, 1d of complex
    """

    ans = np.zeros_like(a)
    N = b.shape[0]

    for i in range(N):
        ans += point_on_segment(a, b[i], b[(i + 1) % N], eps)
        k = _cmp(_point_det(b[(i + 1) % N] - b[i], a - b[i]), 0, eps)
        d1 = _cmp((b[i] - a).imag, 0, eps)
        d2 = _cmp((b[(i + 1) % N] - a).imag, 0, eps)
        ans += np.where((k > 0) * (d1 <= 0) * (d2 > 0), 1, 0) + np.where(
            (k < 0) * (d2 <= 0) * (d1 > 0), -1, 0)

    return ans > 0
