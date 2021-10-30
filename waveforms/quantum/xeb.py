import random
from typing import Callable, Iterable, Optional

import numpy as np
from waveforms.quantum.circuit.simulator import applySeq


def uncorrelatedEntropy(D: int) -> float:
    from scipy.special import polygamma

    return np.euler_gamma + polygamma(0, D)


def PTEntropy(N: int) -> float:
    D = 2**N
    return np.sum([1 / i for i in range(1, D + 1)]) - 1


def crossEntropy(P: np.array,
                 Q: np.array,
                 func: Callable = np.log,
                 eps: float = 1e-9) -> float:
    mask = (P > 0) * (Q > eps)
    if isinstance(P, (int, float, complex)):
        return -np.real(P) * np.sum(func(Q[mask]))
    else:
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
    for Pm, Pe in zip(Pm_lst, Pe_lst):
        if Pi is None:
            Pi = 1.0 / Pm.size
        Si.append(crossEntropy(Pi, Pe))
        Sm.append(crossEntropy(Pm, Pe))
        Se.append(crossEntropy(Pe, Pe))
    return (np.mean(Si) - np.mean(Sm)) / (np.mean(Si) - np.mean(Se))


def specklePurity(Pm_lst: Iterable[np.array]) -> float:
    D = Pm_lst[0].size
    return np.asarray(Pm_lst).var() * D**2 * (D + 1) / (D - 1)


def generateXEBCircuit(qubits, cycle, seed=None, base=['X/2', 'Y/2', 'W/2']):
    """
    Generate a quantum circuit for XEB.
    """

    MAX = len(base)

    ret = []
    rng = random.Random(seed)

    for _ in range(cycle):
        i = rng.randrange(MAX)
        for qubit in qubits:
            ret.append((rng.choice(base), qubit))
        ret.append(('Barrier', qubits))

    return ret


def xebCircuitStates(qubits, cycles, seed, base=['X/2', 'Y/2', 'W/2']):
    """
    XEB Fidelity

    qubits: qubits to measure
    cycles: number of cycles
    """
    circuit = generateXEBCircuit(qubits, cycles, seed, base)
    states = [np.array([1, 0], dtype=complex) for _ in qubits]
    index_map = {q: i for i, q in enumerate(qubits)}
    for gate, qubit in circuit:
        if qubit in index_map:
            states[index_map[qubit]] = applySeq([(gate, 0)],
                                                states[index_map[qubit]])
    return states


def xebProbabilityAndEntropy(states, count, shots):
    _Pe = [(psi * psi.conj()).real for psi in states]

    entropy = np.sum([crossEntropy(p, p) for p in _Pe])
    entropy_i = np.sum([crossEntropy(1 / 2, p) for p in _Pe])

    Pe, Pm = [], []
    for k, v in count.items():
        p = 1
        for i, state in enumerate(k):
            p *= _Pe[i][state]
        if p > 0:
            Pm.append(v / shots)
            Pe.append(p)
    return np.asarray(Pm), np.asarray(Pe), entropy, entropy_i


def Fxeb2(qubits, cycle, seeds, counts, shots, base=['X/2', 'Y/2', 'W/2']):
    Si, Sm, Se = [], [], []

    for seed, count in zip(seeds, counts):
        Pm, Pe, e, ei = xebProbabilityAndEntropy(
            xebCircuitStates(qubits, cycle, seed, base), count, shots)
        Sm.append(crossEntropy(Pm, Pe))
        Si.append(ei)
        Se.append(e)
    return (np.mean(Si) - np.mean(Sm)) / (np.mean(Si) - np.mean(Se))
