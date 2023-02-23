import numpy as np


def dBm2Vpp(x, R0=50):
    """
    Convert dBm to Vpp

    Parameters
    ----------
    x : float
        Power in dBm
    R0 : float
        Load resistance in Ohms

    Returns
    -------
    Vpp : float
        Peak-to-peak voltage in Volts
    """
    mP = 10**(x / 10)
    Vrms = np.sqrt(mP * 1e-3 * R0)
    Vpp = 2 * Vrms * np.sqrt(2)
    return Vpp


def Vpp2dBm(x, R0=50):
    """
    Convert Vpp to dBm

    Parameters
    ----------
    x : float
        Peak-to-peak voltage in Volts
    R0 : float
        Load resistance in Ohms

    Returns
    -------
    dBm : float
        Power in dBm
    """
    Vrms = x / np.sqrt(2) / 2
    mP = Vrms**2 / R0 * 1e3
    dBm = 10 * np.log10(mP)
    return dBm


def dBm2Vrms(x, R0=50):
    """
    Convert dBm to Vrms

    Parameters
    ----------
    x : float
        Power in dBm
    R0 : float
        Load resistance in Ohms

    Returns
    -------
    Vrms : float
        RMS voltage in Volts
    """
    mP = 10**(x / 10)
    Vrms = np.sqrt(mP * 1e-3 * R0)
    return Vrms


def Vrms2dBm(x, R0=50):
    """
    Convert Vrms to dBm

    Parameters
    ----------
    x : float
        RMS voltage in Volts
    R0 : float
        Load resistance in Ohms

    Returns
    -------
    dBm : float
        Power in dBm
    """
    mP = x**2 / R0 * 1e3
    dBm = 10 * np.log10(mP)
    return dBm
