import math

import numpy as np


def _SUn_generator(n, i):
    assert n > 1 and 0 <= i < n**2

    if i == 0:
        return np.eye(n, dtype=np.int8), 1
    mat = np.zeros((n, n), dtype=np.int8)

    k = math.isqrt(i)
    l = (i - k**2) // 2
    A = 1j if (i - k**2) % 2 else 1

    if k != l:
        mat[k, l] = 1
        if A == 1:
            mat[l, k] = 1
        else:
            mat[l, k] = -1
    else:
        for j in range(k):
            mat[j, j] = 1
        mat[k, k] = -k
        A = 1 / np.sqrt(k * (k + 1) / 2)

    return mat, A


class _SUGroup():

    def __init__(self, n: int):
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = n

    def __getitem__(self, i: int):
        if not 0 <= i < self.n**2:
            raise IndexError(f"i must be in [0, {self.n**2-1}]")
        mat, A = _SUn_generator(self.n, i)
        return A * mat

    def __repr__(self):
        return f"SU({self.n})"


def SU(n: int):
    return _SUGroup(n)
