import itertools
from functools import partial, reduce

import numpy as np
from waveforms.quantum.math import U, fSim, rfUnitary

__matrix_of_gates = {}


def regesterGateMatrix(gate, mat, N=None):
    if N is None:
        N = int(np.log2(mat.shape[0]))
    __matrix_of_gates[gate] = (mat, N)


def gate2mat(gate):
    if isinstance(gate, str) and gate in __matrix_of_gates:
        return __matrix_of_gates[gate][0]
    elif isinstance(gate, tuple) and gate[0] in __matrix_of_gates:
        if callable(__matrix_of_gates[gate[0]][0]):
            return __matrix_of_gates[gate[0]][0](*gate[1:])
        else:
            raise ValueError(
                f"Could not call {gate[0]}(*{gate[1:]}), `{gate[0]}` is not callable."
            )
    elif isinstance(gate, tuple) and gate[0] == 'C':
        U = gate2mat(gate[1])
        ret = np.eye(2 * U.shape[0], dtype=np.complex)
        ret[U.shape[0]:, U.shape[0]:] = U
        return ret
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


def applySeq(seq, psi0=None):
    if psi0 is None:
        psi = np.array([1, 0], dtype=complex)
        N = 1
    else:
        psi = psi0
        N = int(np.round(np.log2(psi.shape[0])))

    if len(psi.shape) == 1:
        I = np.array([1, 0])
    else:
        I = np.eye(2)

    def func(psi, U):
        return U @ psi

    for gate, qubits in seq:
        if isinstance(qubits, tuple):
            M = max(qubits)
        else:
            M = qubits
            qubits = (qubits, )
        if M >= N:
            psi = reduce(np.kron, itertools.repeat(I, times=M - N + 1), psi)
            N = M + 1

        reduceSubspace(qubits, N, psi, func, (gate2mat(gate), ))

    return psi


def seq2mat(seq):
    return applySeq(seq, np.eye(2, dtype=complex))


regesterGateMatrix('U', U, 1)
regesterGateMatrix('P', partial(U, theta=0, phi=0), 1)
regesterGateMatrix('rfUnitary', rfUnitary, 1)
regesterGateMatrix('Rx', partial(rfUnitary, phi=0), 1)
regesterGateMatrix('Ry', partial(rfUnitary, phi=np.pi / 2), 1)
regesterGateMatrix('Rz', partial(U, theta=0, phi=0), 1)
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

    regesterGateMatrix('W', rfUnitary(np.pi / 2, np.pi / 4))

    def randomSeq(depth, N):
        seq = []
        for i in range(depth):
            for j in range(N):
                seq.append((np.random.choice(['X/2', 'Y/2', 'W']), j))
            for j in range(i % 2, N, 2):
                seq.append(('SQiSWAP', (j, (j + 1) % N)))
        return seq

    p = []
    # run 1000 random circuit on 6 qubits
    for i in range(1000):
        seq = randomSeq(50, 6)
        psi = applySeq(seq)
        p.extend(list(np.abs(psi)**2))
    p = np.asarray(p)

    # plot distribution of probabilities
    N = 2**6
    y, x = np.histogram(N * p, bins=50, density=True)

    import matplotlib.pyplot as plt

    plt.semilogy((x[:-1] + x[1:]) / 2, y)
    plt.show()
