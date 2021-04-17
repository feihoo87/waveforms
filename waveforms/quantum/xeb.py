from typing import Callable, Iterable, Optional

import numpy as np
from numpy import euler_gamma
from scipy.special import polygamma


def uncorrelatedEntropy(D: int) -> float:
    return euler_gamma + polygamma(0, D)


def PTEntropy(N: int) -> float:
    D = 2**N
    return np.sum([1 / i for i in range(1, D + 1)]) - 1


def crossEntropy(P: np.array,
                 Q: np.array,
                 func: Callable = np.log,
                 eps: float = 1e-9) -> float:
    mask = (P > 0) * (Q > eps)
    return -np.sum(P[mask] * func(Q[mask]))


def Fxeb(Pm_lst: Iterable[np.array],
         Pe_lst: Iterable[np.array],
         Pi: Optional[np.array] = None) -> float:
    """
    XEB Fidelity

    Pm_lst: list of measured distribution
    Pe_lst: list of expected distribution
    """
    Si, Sm, Se = [], [], []
    if Pi is None:
        Pi = np.full_like(Pm_lst[0], 1.0 / Pm_lst[0].size)
    for Pm, Pe in zip(Pm_lst, Pe_lst):
        Si.append(crossEntropy(Pi, Pe))
        Sm.append(crossEntropy(Pm, Pe))
        Se.append(crossEntropy(Pe, Pe))
    return (np.mean(Si) - np.mean(Sm)) / (np.mean(Si) - np.mean(Se))


def specklePurity(Pm_lst: Iterable[np.array]):
    D = Pm_lst[0].size
    return np.asarray(Pm_lst).var()*D**2*(D+1)/(D-1)