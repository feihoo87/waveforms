import numpy as np


def dBm2Vpp(x, R0=50):
    mP = 10**(x / 10)
    Vrms = np.sqrt(mP * 1e-3 * R0)
    Vpp = 2 * Vrms * np.sqrt(2)
    return Vpp


def Vpp2dBm(x, R0=50):
    Vrms = x / np.sqrt(2) / 2
    mP = Vrms**2 / R0 * 1e3
    dBm = 10 * np.log10(mP)
    return dBm