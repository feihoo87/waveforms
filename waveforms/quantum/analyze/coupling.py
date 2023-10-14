import numpy as np


def effect_coupling(wc, w1, w2, eta, C12, C1, C2):
    """
    See also
    --------
    https://doi.org/10.1103/PhysRevApplied.10.054062
    """
    Delta1 = w1 - wc
    Delta2 = w2 - wc
    Sigma1 = w1 + wc
    Sigma2 = w2 + wc

    return 1 / 2 * (wc / 4 *
                    (1 / Delta1 + 1 / Delta2 - 1 / Sigma1 - 1 / Sigma2) * eta +
                    eta + 1) * C12 / np.sqrt(C1 * C2) * np.sqrt(w1 * w2)


def effect_coupling_2(wc, w1, w2, wq, wc_off, wc_2, g_2):
    """
    Parameters
    ----------
    wc : float
        frequency of coupler
    w1 : float
        frequency of qubit 1
    w2 : float
        frequency of qubit 2
    wq : float
        calibration point, frequency of qubit 1 and 2 when w1 = w2 = wq
    wc_off : float
        decoupling frequency of coupler when w1 = w2 = wq
    wc_2 : float
        second calibration frequency of coupler when w1 = w2 = wq
    g_2 : float
        coupling strength of coupler when w1 = w2 = wq and wc = wc_2

    See also
    --------
    https://doi.org/10.1103/PhysRevApplied.10.054062
    """
    Delta1 = w1 - wc
    Delta2 = w2 - wc
    Sigma1 = w1 + wc
    Sigma2 = w2 + wc
    eta = (wc_off / wq)**2 - 1

    a = g_2 / wq / (eta / (1 - (wc_2 / wq)**2) + 1)

    return a * (wc / 4 *
                (1 / Delta1 + 1 / Delta2 - 1 / Sigma1 - 1 / Sigma2) * eta +
                eta + 1) * np.sqrt(w1 * w2)


def decoupling_frequency_of_coupler(w1, w2, k):
    """
    calculate decoupling frequency of coupler

    Parameters
    ----------
    w1 : float
        frequency of qubit 1
    w2 : float
        frequency of qubit 2
    k : float
        k = wc / wq > 1 where wc is decoupling frequency of coupler when w1 = w2 = wq
    
    See also
    --------
    https://doi.org/10.1103/PhysRevApplied.10.054062
    
    """
    eta = k**2 - 1
    a = np.sqrt((-2 * w1**2 - eta * w1**2 - 2 * w2**2 - eta * w2**2)**2 - 8 *
                (2 * w1**2 * w2**2 + 2 * eta * w1**2 * w2**2))
    return 0.5 * np.sqrt(2 * w1**2 + eta * w1**2 + 2 * w2**2 + eta * w2**2 + a)
