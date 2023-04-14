import inspect

import numpy as np
from scipy.optimize import curve_fit


def fit(func,
        xdata,
        ydata,
        p0=None,
        sigma=None,
        bounds=(-np.inf, np.inf),
        guess=None,
        static_params=None):
    """
    Fit data to a function.

    Args:
        func (function): fitting function
        xdata (np.ndarray): x data
        ydata (np.ndarray): y data
        p0 (dict): initial parameters
        sigma (np.ndarray): standard deviation of ydata
        bounds (tuple): lower and upper bounds of parameters
        guess (function): function to guess initial parameters
        static_params (dict): static parameters for fitting

    Returns:
        arg_names (list): parameter names
        popt (np.ndarray): optimized parameters
        pcov (np.ndarray): covariance matrix
        fitted (np.ndarray): fitted data
    """
    sig = inspect.signature(func)

    if static_params is None:
        static_params = {}

    arg_names = []
    for i, (arg_name, param) in enumerate(sig.parameters.items()):
        if i < 1:
            continue
        if arg_name not in static_params:
            arg_names.append(arg_name)

    def func_wrapper(x, *args):
        params = dict(zip(arg_names, args)) | static_params
        return func(x, **params)

    if guess is None and p0 is None:
        raise ValueError('Initial parameters are not specified!')

    if p0 is None:
        p0 = {}

    if guess is not None:
        p0 = p0 | guess(xdata, ydata, p0 | static_params)

    if isinstance(p0, dict):
        p0 = [p0[arg_name] for arg_name in arg_names]

    try:
        popt, pcov = curve_fit(func_wrapper,
                               xdata,
                               ydata,
                               p0=p0,
                               sigma=sigma,
                               absolute_sigma=True,
                               method='trf',
                               bounds=bounds)
        fitted = func_wrapper(xdata, *popt)
        return arg_names, popt, pcov, fitted
    except:
        raise ValueError('Fitting failed!')
