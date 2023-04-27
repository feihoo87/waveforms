import math
import numpy as np


def SUn_generator(n, i):
    assert n > 1 and 0 <= i < 2**n

    if i == 0:
        return np.eye(n, dtype=complex)
    mat = np.zeros((n, n), dtype=complex)

    k = math.isqrt(i)
    l = (i - k**2) // 2
    v = 1j if (i - k**2) % 2 else 1

    if k != l:
        mat[k, l] = v
        mat[l, k] = np.conj(v)
    else:
        for j in range(k):
            mat[j, j] = 1
        mat[k, k] = -k

    return mat


class _SUGroup():

    def __init__(self, n):
        self.n = n

    def __getitem__(self, i):
        return SUn_generator(self.n, i)


def SU(n):
    return _SUGroup(n)