import random
from typing import Callable, Iterable, Optional

import numpy as np
from numpy import euler_gamma
from scipy.special import polygamma
from waveforms.quantum.circuit.simulator import applySeq


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


def xebProbability(states, count, shots):
    _Pe = [(psi * psi.conj()).real for psi in states]

    Pe, Pm = [], []
    for k, v in count.items():
        p = 1
        for i, state in enumerate(k):
            p *= _Pe[i][state]
        if p > 0:
            Pm.append(v / shots)
            Pe.append(p)
    return np.asarray(Pm), np.asarray(Pe)


def Fxeb2(qubits, cycle, seeds, counts, shots):
    Pe_lst, Pm_lst = [], []
    for seed, count in zip(seeds, counts):
        Pm, Pe = xebProbability(xebCircuitStates(qubits, cycle, seed), count,
                                shots)
        Pm_lst.append(Pm)
        Pe_lst.append(Pe)
    return Fxeb(Pe_lst, Pm_lst)