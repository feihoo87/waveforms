import numpy as np


def extend(stablizers, sign, N):
    n = sign.size
    if n >= N:
        return stablizers, sign
    new_stablizers = np.eye(N, dtype=np.int8)
    new_sign = np.zeros(N, dtype=np.int8)
    new_sign[:n] = sign
    new_stablizers[:n, :n] = stablizers
    return new_stablizers, new_sign


def H(stablizers, sign, q):
    xz = stablizers[q]
    sign += (xz == 3) * 2
    stablizers[q] = (xz >> 1) | ((xz << 1) & 2)


def P(stablizers, sign, q):
    xz = stablizers[q]
    sign += (xz == 3) * 2
    stablizers[q] ^= xz >> 1


def X(stablizers, sign, q):
    xz = stablizers[q]
    sign += xz << 1


def Y(stablizers, sign, q):
    xz = stablizers[q]
    sign += (xz << 1) & xz


def Z(stablizers, sign, q):
    xz = stablizers[q]
    sign += xz & 2


def CZ(stablizers, sign, c, t):
    xz1 = stablizers[c]
    xz2 = stablizers[t]
    sign += (xz1 & xz2 & 2) & ((xz1 ^ xz2) << 1)
    stablizers[c] ^= xz2 >> 1
    stablizers[t] ^= xz1 >> 1


def measure(stablizers, sign, q):
    code = np.array([0, 0, 0, 0, 0, 0, 1, 3, 0, 3, 0, 1, 0, 1, 3, 0])
    if np.any(stablizers[q] & 2):
        index = np.argwhere(stablizers[q] & 2).reshape(-1)
        p = index[0]
        for i in index[1:]:
            stablizers[:, i] = stablizers[:, i] ^ stablizers[:, p]
            sign[i] += np.sum(code[(stablizers[:, i] << 2) | stablizers[:, p]])
        stablizers[:, p] = np.zeros_like(stablizers[:, p])
        stablizers[q, p] = 1
        return 2
    else:
        x = (np.sum(sign[stablizers[q] & 1 == 1]) & 2) >> 1
        return x


def run_circuit(circ, stablizers=None, sign=None):
    N = 1
    results = []
    if stablizers is None:
        stablizers = np.eye(N, dtype=np.int8)
        sign = np.zeros(N, dtype=np.int8)

    for gate, *qubits in circ:

        for q in qubits:
            stablizers, sign = extend(stablizers, sign, q + 1)
        if gate == 'I':
            continue
        elif gate == 'H':
            H(stablizers, sign, *qubits)
        elif gate == 'P' or gate == 'S':
            P(stablizers, sign, *qubits)
        elif gate == 'CZ':
            CZ(stablizers, sign, *qubits)
        elif gate == 'CX' or gate == 'Cnot':
            stablizers, sign, _ = run_circuit([('H', qubits[1]),
                                               ('CZ', *qubits),
                                               ('H', qubits[1])], stablizers,
                                              sign)
        elif gate == 'X':
            X(stablizers, sign, *qubits)
        elif gate == 'Y':
            Y(stablizers, sign, *qubits)
        elif gate == 'Z':
            Z(stablizers, sign, *qubits)
        elif gate == 'M' or gate == 'Measure':
            result = measure(stablizers, sign, *qubits)
            results.append(result)
        else:
            raise ValueError(f"Unknown gate {gate}")
    return stablizers, sign, results


def decode_stablizers(stablizers, sign):
    ret = []
    for s, S in zip(sign, stablizers.T):
        ret.append([' +', '+i', ' -', '-i'][s & 3] +
                   ''.join(['IZXY'[op] for op in S]))
    return ret
