from concurrent.futures import process
import itertools
from functools import partial, reduce

import numpy as np

from .mat import U, fSim, make_immutable, rfUnitary

__matrix_of_gates = {}


def regesterGateMatrix(gate, mat, N=None):
    if isinstance(mat, np.ndarray):
        mat = make_immutable(mat)
    if N is None:
        N = int(np.log2(mat.shape[0]))
    __matrix_of_gates[gate] = (mat, N)


def gate_name(gate):
    if isinstance(gate, tuple):
        return gate_name(gate[0])
    elif isinstance(gate, str):
        return gate
    else:
        raise ValueError(f'Unexcept gate {gate}')


def gate2mat(gate):
    if isinstance(gate, str) and gate in __matrix_of_gates:
        return __matrix_of_gates[gate]
    elif isinstance(gate, tuple) and gate[0] in __matrix_of_gates:
        if callable(__matrix_of_gates[gate[0]][0]):
            return __matrix_of_gates[gate[0]][0](
                *gate[1:]), __matrix_of_gates[gate[0]][1]
        else:
            raise ValueError(
                f"Could not call {gate[0]}(*{gate[1:]}), `{gate[0]}` is not callable."
            )
    elif gate_name(gate) == 'C':
        U, N = gate2mat(gate[1])
        ret = np.eye(2 * U.shape[0], dtype=np.complex)
        ret[U.shape[0]:, U.shape[0]:] = U
        return ret, N + 1
    else:
        raise ValueError(f'Unexcept gate {gate}')


def splite_at(l, bits):
    """将 l 的二进制位于 bits 所列的位置上断开插上0
    
    如 splite_at(int('1111',2), [0,2,4,6]) == int('10101010', 2)
    bits 必须从小到大排列
    """
    r = l
    for n in bits:
        mask = (1 << n) - 1
        low = r & mask
        high = r - low
        r = (high << 1) + low
    return r


def place_at(l, bits):
    """将 l 的二进制位置于 bits 所列的位置上
    
    如 place_at(int('10111',2), [0,2,4,5,6]) == int('01010101', 2)
    """
    r = 0
    for index, n in enumerate(bits):
        b = (l >> index) & 1
        r += b << n
    return r


def reduceSubspace(targets, N, inputMat, func, args):
    innerDim = 2**len(targets)
    outerDim = 2**(N - len(targets))

    targets = tuple(reversed([N - i - 1 for i in targets]))

    def index(targets, i, j):
        return splite_at(j, sorted(targets)) | place_at(i, targets)

    if len(inputMat.shape) == 1:
        for k in range(outerDim):
            innerIndex = [index(targets, i, k) for i in range(innerDim)]
            inputMat[innerIndex] = func(inputMat[innerIndex], *args)
    else:
        for k, l in itertools.product(range(outerDim), repeat=2):
            innerIndex = np.asarray(
                [[index(targets, i, k),
                  index(targets, j, l)]
                 for i, j in itertools.product(range(innerDim), repeat=2)]).T
            sub = inputMat[innerIndex[0], innerIndex[1]].reshape(
                (innerDim, innerDim))
            inputMat[innerIndex[0], innerIndex[1]] = func(sub, *args).flatten()
    return inputMat


def _apply_gate(gate, inputMat, unitary_process, qubits, N):
    U, n = gate2mat(gate)
    if len(qubits) == n and all(isinstance(qubit, int) for qubit in qubits):
        reduceSubspace(qubits, N, inputMat, unitary_process, (U, ))
    elif n == 1 and all(isinstance(qubit, int) for qubit in qubits):
        for qubit in qubits:
            reduceSubspace([qubit], N, inputMat, unitary_process, (U, ))
    elif len(qubits) == n and all(
            isinstance(qubit, [tuple, list]) for qubit in qubits):
        for qubit_tuple in zip(*qubits):
            reduceSubspace(qubit_tuple, N, inputMat, unitary_process, (U, ))
    else:
        raise ValueError(f'Unexcept gate {gate} and qubits {qubits}')


def _measure_process(rho):
    return np.array([[rho[0, 0], 0], [0, rho[1, 1]]])


def _reset_process(rho):
    return np.array([[rho[0, 0] + rho[1, 1], 0], [0, 0]])


def _decohherence_process(rho, Gamma_t, gamma_t):
    rho00 = rho[0, 0] + rho[1, 1] * (1 - np.exp(-Gamma_t))
    rho11 = rho[1, 1] * np.exp(-Gamma_t)
    rho01 = rho[0, 1] * np.exp(-Gamma_t / 2 - gamma_t**2)
    rho10 = rho[1, 0] * np.exp(-Gamma_t / 2 - gamma_t**2)
    return np.array([[rho00, rho01], [rho10, rho11]])


def applySeq(seq, psi0=None):

    def _set_vector_to_rho(psi):
        psi = psi.reshape(-1, 1).conj() @ psi.reshape(1, -1)
        unitary_process = lambda psi, U: U @ psi @ U.T.conj()
        return psi, unitary_process, np.array([[1, 0], [0, 0]])

    if psi0 is None:
        psi = np.array([1, 0], dtype=complex)
        N = 1
    else:
        psi = psi0
        N = int(np.round(np.log2(psi.shape[0])))

    psi0 = np.array([1, 0])
    unitary_process = lambda psi, U: U @ psi

    if psi.ndim == 2:
        psi, unitary_process, psi0 = _set_vector_to_rho(psi)

    for gate, *qubits in seq:
        M = max(qubits)
        if M >= N:
            psi = reduce(np.kron, itertools.repeat(psi0, times=M - N + 1), psi)
            N = M + 1

        if gate_name(gate) in ['Barrier']:
            continue
        if gate_name(gate) in ['Delay']:
            if len(gate) == 2:
                continue
            else:
                if len(gate) == 3:
                    _, t, T1 = gate
                    Gamma_t = t / T1
                    gamma_t = 0
                else:
                    _, t, T1, Tphi = gate
                    Gamma_t = t / T1
                    gamma_t = t / Tphi
        if gate_name(gate) in ['Measure', 'Reset', 'Delay'] and psi.ndim == 1:
            psi, unitary_process, psi0 = _set_vector_to_rho(psi)

        if gate_name(gate) == 'Measure':
            reduceSubspace(qubits, N, psi, _measure_process, ())
        elif gate_name(gate) == 'Reset':
            reduceSubspace(qubits, N, psi, _reset_process, ())
        elif gate_name(gate) == 'Delay':
            reduceSubspace(qubits, N, psi, _decohherence_process,
                           (Gamma_t, gamma_t))
        else:
            _apply_gate(gate, psi, unitary_process, qubits, N)

    return psi


def seq2mat(seq, U=None):
    I = np.eye(2, dtype=np.complex)
    if U is None:
        U = np.eye(2, dtype=np.complex)
        N = 1
    else:
        N = int(np.round(np.log2(U.shape[0])))

    unitary_process = lambda U0, U: U @ U0

    for gate, *qubits in seq:
        M = max(qubits)
        if M >= N:
            U = reduce(np.kron, itertools.repeat(I, times=M - N + 1), U)
            N = M + 1
        if gate_name(gate) in ['Delay', 'Barrier']:
            continue
        if gate_name(gate) in ['Measure', 'Reset']:
            raise ValueError(
                'Measure and Reset must be applied to a state vector')
        else:
            _apply_gate(gate, U, unitary_process, qubits, N)
    return U


regesterGateMatrix('U', U, 1)
regesterGateMatrix('P', lambda p: U(theta=0, phi=0, lambda_=p), 1)
regesterGateMatrix('rfUnitary', rfUnitary, 1)
regesterGateMatrix('Rx', partial(rfUnitary, phi=0), 1)
regesterGateMatrix('Ry', partial(rfUnitary, phi=np.pi / 2), 1)
regesterGateMatrix('Rz', lambda p: U(theta=0, phi=0, lambda_=p), 1)
regesterGateMatrix('fSim', fSim, 2)

# one qubit
regesterGateMatrix('I', np.array([[1, 0], [0, 1]]))
regesterGateMatrix('X', np.array([[0, -1j], [-1j, 0]]))
regesterGateMatrix('Y', np.array([[0, -1], [1, 0]]))
regesterGateMatrix('X/2', np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2))
regesterGateMatrix('Y/2', np.array([[1, -1], [1, 1]]) / np.sqrt(2))
regesterGateMatrix('-X/2', np.array([[1, 1j], [1j, 1]]) / np.sqrt(2))
regesterGateMatrix('-Y/2', np.array([[1, 1], [-1, 1]]) / np.sqrt(2))
regesterGateMatrix('Z', np.array([[1, 0], [0, -1]]))
regesterGateMatrix('S', np.array([[1, 0], [0, 1j]]))
regesterGateMatrix('-S', np.array([[1, 0], [0, -1j]]))
regesterGateMatrix('H', np.array([[1, 1], [1, -1]]) / np.sqrt(2))

# non-clifford
regesterGateMatrix('T',
                   np.array([[1, 0], [0, 1 / np.sqrt(2) + 1j / np.sqrt(2)]]))
regesterGateMatrix('-T',
                   np.array([[1, 0], [0, 1 / np.sqrt(2) - 1j / np.sqrt(2)]]))
regesterGateMatrix('W/2', rfUnitary(np.pi / 2, np.pi / 4))
regesterGateMatrix('-W/2', rfUnitary(-np.pi / 2, np.pi / 4))
regesterGateMatrix('V/2', rfUnitary(np.pi / 2, 3 * np.pi / 4))
regesterGateMatrix('-V/2', rfUnitary(-np.pi / 2, 3 * np.pi / 4))

# two qubits
regesterGateMatrix(
    'CZ', np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]))
regesterGateMatrix(
    'Cnot', np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]))
regesterGateMatrix(
    'iSWAP',
    np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]))
regesterGateMatrix(
    'SWAP', np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))
regesterGateMatrix(
    'CR',
    np.array([[1, 1j, 0, 0], [1j, 1, 0, 0], [0, 0, 1, -1j], [0, 0, -1j, 1]]) /
    np.sqrt(2))

# non-clifford
regesterGateMatrix(
    'SQiSWAP',
    np.array([[1, 0, 0, 0], [0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0],
              [0, 1j / np.sqrt(2), 1 / np.sqrt(2), 0], [0, 0, 0, 1]]))

if __name__ == '__main__':
    # Porter-Thomas distribution

    def randomSeq(depth, N):
        seq = []
        for i in range(depth):
            for j in range(N):
                seq.append((np.random.choice(['X/2', 'Y/2', 'W/2']), j))
            for j in range(i % 2, N, 2):
                seq.append(('SQiSWAP', j, (j + 1) % N))
        return seq

    p = []
    # run 1000 random circuit on 6 qubits
    for i in range(1000):
        print('    ', i, end='')
        seq = randomSeq(50, 6)
        psi = applySeq(seq)
        p.extend(list(np.abs(psi)**2))
        print('    ', i)
    p = np.asarray(p)

    # plot distribution of probabilities
    N = 2**6
    y, x = np.histogram(N * p, bins=50, density=True)

    import matplotlib.pyplot as plt

    plt.semilogy((x[:-1] + x[1:]) / 2, y)
    plt.show()
