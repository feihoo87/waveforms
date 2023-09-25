import numpy as np


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
