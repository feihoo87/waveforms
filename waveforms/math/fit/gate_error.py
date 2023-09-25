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


def rb_residual(pars, x, data=None):
    A = pars['A']
    tau_gate = pars['tau_gate']

    B = pars['B']
    model = A * (1 - np.exp(-x / tau_gate)) + B

    if data is None:
        return model
    if data.shape == x.shape:
        return model - data

    tau_ref = pars['tau_ref']
    model_ref = A * (1 - np.exp(-x / tau_ref)) + B
    return np.concatenate((model - data[1], model_ref - data[0]))


def fit_rb(N, cycle, y, y_ref=None):
    """
    fit RB data

    fit RB data as exponential decay
    y = A * (1 - exp(-cycle / tau)) + B
    extract gate error from tau

    Parameters
    ----------
    N : int
        number of qubits
    cycle : array_like
        number of cycles
    y : array_like
        interleaved sequence decay
    y_ref : array_like
        reference sequence decay

    Returns
    -------
    result : lmfit.minimizer.MinimizerResult
        fit result
    gate_error : float
        gate error
    """
    from lmfit import create_params, minimize
    from lmfit.models import ExponentialModel

    def gate_error(N, tau_gate, tau_ref=None):
        if tau_ref is None:
            r_gate = (1 - np.exp(-1 / tau_gate)) * (2**N - 1) / 2**N
        else:
            r_gate = (1 - np.exp(1 / tau_ref - 1 / tau_gate)) * (2**N -
                                                                 1) / 2**N
        return r_gate

    exp_mod = ExponentialModel()

    params = exp_mod.guess(y - 1 / 2**N, x=cycle)
    A = params['amplitude'].value
    tau_gate = params['decay'].value
    params = create_params(A=A, tau_gate=tau_gate, B=1 / 2**N)
    result = minimize(rb_residual, params, args=(cycle, ), kws={'data': y})
    A = result.params['amplitude'].value
    tau_gate = result.params['decay'].value
    B = result.params['B'].value

    if y_ref is None:
        return result, gate_error(N, tau_gate)
    else:
        params = exp_mod.guess(y_ref - 1 / 2**N, x=cycle)
        tau_ref = params['decay'].value
        params = create_params(A=A, tau_gate=tau_gate, B=B, tau_ref=tau_ref)

        result = minimize(rb_residual,
                          params,
                          args=(cycle, ),
                          kws={'data': [y_ref, y]})
        tau_gate = result.params['A_decay'].value
        tau_ref = result.params['tau_ref'].value
        return result, gate_error(N, tau_gate, tau_ref)


def single_qubit_gate_error(gate_time, T1, T2):
    """
    calculate single qubit gate error upper bound

    The error rate obtained from a randomized benchmarking experiment may be calculated as
    $r=1-\mathcal{F}$, where $\mathcal{F}$ is the average ﬁdelity
    $$
    \mathcal{F} = \frac{1}{6}\sum_i\langle\psi_i|\mathcal{E}(\psi_i)|\psi_i\rangle
    $$
    where $|\psi_i\rangle$ are eigenstates of X, Y , and Z operators.

    Parameters
    ----------
    gate_time : float
        gate time
    T1 : float
        T1 time
    T2 : float
        T2 time, defined as 1 / T2 = 1 / (2 * T1) + 1 / T_phi
        where T_phi is the pure dephasing time

    See Also
    --------
    https://doi.org/10.1103/PhysRevA.102.022220

    Returns
    -------
    r_gate : float
        gate error
    """
    r_gate = 1 / 2 - 1 / 6 * np.exp(-gate_time / T1) - 1 / 3 * np.exp(
        -gate_time / T2)
    return r_gate


def two_qubit_gate_error(gate_time, T1, T2, T1_2=None, T2_2=None):
    """
    calculate two qubit gate error upper bound

    Parameters
    ----------
    gate_time : float
        gate time
    T1 : float
        T1 time
    T2 : float
        T2 time, defined as 1 / T2 = 1 / (2 * T1) + 1 / T_phi
        where T_phi is the pure dephasing time
    T1_2 : float
        T1 time of the second qubit
    T2_2 : float
        T2 time of the second qubit

    See Also
    --------
    https://doi.org/10.1103/PhysRevA.102.022220
    https://doi.org/10.1038/npjqi.2017.1

    Returns
    -------
    r_gate : float
        gate error
    """
    if T1_2 is None:
        T1_2 = T1
    if T2_2 is None:
        T2_2 = T2

    r_gate = 1 - (1 - single_qubit_gate_error(gate_time, T1, T2)) * (
        1 - single_qubit_gate_error(gate_time, T1_2, T2_2))
    return r_gate


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
