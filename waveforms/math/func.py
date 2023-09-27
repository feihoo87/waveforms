import numpy as np


def shift_poly(a, offset):
    if isinstance(a, np.poly1d):
        a = a.coef
    c = np.poly1d([1, -offset])
    b = np.poly1d([0])
    for i, aa in enumerate(a[::-1]):
        b = b + c**i * aa
    return b.coef