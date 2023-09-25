import numpy as np


def t1_residual(pars, x, data=None):
    A = pars['A']
    T1 = pars['T1']
    B = pars['B']
    model = A * np.exp(-x / T1) + B

    if data is None:
        return model
    if data.shape == x.shape:
        return model - data
    return np.concatenate((model - data[:len(x)], data[len(x):] - B))


def fit_t1(x, y, y_ref=None):
    """
    fit T1 data

    fit T1 data as exponential decay
    y = A * exp(-x / T1) + B

    Parameters
    ----------
    x : array_like
        time
    y : array_like
        T1 decay data
    y_ref : array_like
        reference data

    Returns
    -------
    result : lmfit.minimizer.MinimizerResult
        fit result
    """
    from lmfit import create_params, minimize
    from lmfit.models import ExponentialModel

    exp_mod = ExponentialModel()

    params = exp_mod.guess(y, x=x)
    A = params['amplitude'].value
    T1 = params['decay'].value
    B = params['offset'].value
    params = create_params(A=A, T1=T1, B=B)
    result = minimize(t1_residual, params, args=(x, ), kws={'data': y})
    A = result.params['A'].value
    T1 = result.params['T1'].value
    B = result.params['B'].value

    if y_ref is None:
        return result
    else:
        params = create_params(A=A, T1=T1, B=np.mean(y_ref))
        result = minimize(t1_residual,
                          params,
                          args=(x, ),
                          kws={'data': np.concatenate((y, y_ref))})
        return result


def ramsey_residual(pars, x, data=None):
    A = pars['A']
    T1 = pars['T1']
    T2 = pars['T2']
    B = pars['B']
    model = 0.5 * A * np.exp(-(x / (2 * T1)) - (x / T2)**2) * np.cos(
        2 * np.pi * pars['f'] * x) + 0.5 + B

    if data is None:
        return model
    if data.shape == x.shape:
        return model - data
    return np.concatenate(
        (model - data[:len(x)], t1_residual(pars, x, data[len(x):])))


def fit_ramsey(x, y, y_t1=None):
    """
    fit Ramsey data

    fit Ramsey data as exponential decay
    y = A * exp(-(x / (2 * T1)) - (x / T2) **2 ) * cos(2 * pi * f * x) + B

    Parameters
    ----------
    x : array_like
        time
    y : array_like
        Ramsey decay data
    y_t1 : array_like
        T1 decay data

    Returns
    -------
    result : lmfit.minimizer.MinimizerResult
        fit result
    """
    from scipy.signal import hilbert
    from lmfit import create_params, minimize
    from lmfit.models import ExponentialModel

    exp_mod = ExponentialModel()

    z = hilbert(y - np.mean(y))
    phase = np.unwrap(np.angle(z))
    f = np.mean(np.diff(phase)) / (2 * np.pi * np.mean(np.diff(x)))
    envelope = np.abs(z)
    params = exp_mod.guess(envelope, x=x)

