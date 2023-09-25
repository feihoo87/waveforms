import functools
import random
import collections
from .paulis import imul_paulis


def H(a, i):
    sign = a & 3
    a >>= 2
    x, z = (a >> 2 * i) & 1, (a >> 2 * i + 1) & 1
    mask = x ^ z
    a ^= (mask << 2 * i + 1) | (mask << 2 * i)
    sign += 2 * (x & z)
    return (a << 2) | (sign & 3)


def P(a, i):
    sign = a & 3
    a >>= 2
    x, z = (a >> 2 * i) & 1, (a >> 2 * i + 1) & 1
    a ^= (x << 2 * i + 1)
    sign += 2 * (x & z)
    return (a << 2) | (sign & 3)


def CZ(a, i, j):
    sign = a & 3
    a >>= 2
    x1, z1, x2, z2 = (a >> 2 * i) & 1, (a >> 2 * i + 1) & 1, (
        a >> 2 * j) & 1, (a >> 2 * j + 1) & 1
    a ^= (x2 << 2 * i + 1) | (x1 << 2 * j + 1)
    sign += 2 * ((z1 ^ z2) & (x1 & x2))
    return (a << 2) | (sign & 3)


def measure(stablizers, a):
    """
    Measure a qubit.

    Args:
        stablizers: A list of stabilizers to measure the qubit in.
        a: The index of the qubit to measure.

    Returns:
        A tuple of the form (stablizers, result), where stablizers are lists
        of stabilizers after measurement, and result is the result of the
        measurement.
    """
    for p, stablizer in enumerate(stablizers):
        if (stablizer >> 2 * a + 2) & 1:
            break
    else:
        result = functools.reduce(imul_paulis, [
            stablizer
            for stablizer in stablizers if (stablizer >> 2 * a + 2 + 1) & 1
        ], 0) & 3
        result >>= 1
        return result, stablizers

    for i, s in enumerate(stablizers):
        if i != p and (s >> 2 * a + 2) & 1:
            stablizers[i] = imul_paulis(s, stablizer)
    stablizers[p] = (2 << 2 * a + 2)
    return 2, stablizers


def run_circuit(circ, stablizers=None, state=None):
    """
    Run a circuit on a list of stabilizers.

    Args:
        circ: A list of gates to apply. Each gate is a tuple of the form
            (gate, *qubits), where gate is one of 'H', 'P', 'CZ' and qubits
            is a list of qubits to apply the gate to.
        stablizers: A list of stabilizers to apply the circuit to. If None,
            a new list is created. Stabilizers are encoded as integers, where
            the first 2 * n bits encode the n qubits, and the last 2 bits
            encode the sign. Stabilizers are encoded as follows:
                I: 0
                X: 1
                Z: 2
                Y: 3
            The sign is encoded as follows:
                +: 0
                i: 1
                -: 2
                -i: 3
            The indices of the stabilizers are determined by the number of qubits
            in the circuit.
    
    Returns:
        A list of stabilizers after the circuit has been applied.
    """
    if stablizers is None:
        stablizers = []
    if state is None:
        state = []
    for gate, *qubits in circ:
        for i in qubits:
            if i + 1 > len(stablizers):
                for j in range(len(stablizers), i + 1):
                    stablizers.append(2 << 2 * j + 2)
        if gate == 'I':
            continue
        elif gate == 'H':
            stablizers = [H(a, *qubits) for a in stablizers]
        elif gate == 'P' or gate == 'S':
            stablizers = [P(a, *qubits) for a in stablizers]
        elif gate == 'CZ':
            stablizers = [CZ(a, *qubits) for a in stablizers]
        elif gate == 'CX' or gate == 'Cnot':
            stablizers, state = run_circuit([('H', qubits[1]), ('CZ', *qubits),
                                             ('H', qubits[1])], stablizers,
                                            state)
        elif gate == 'X':
            stablizers, state = run_circuit([('H', *qubits), ('P', *qubits),
                                             ('P', *qubits), ('H', *qubits)],
                                            stablizers, state)
        elif gate == 'M' or gate == 'Measure':
            flag, stablizers = measure(stablizers, *qubits)
            state.append((qubits[0], flag))
        else:
            raise ValueError(f"Unknown gate {gate}")
    trans = 0
    result = []
    for q, b in state:
        if b == 2:
            trans ^= random.randint(0, 1)
            b = 0
        result.append(b ^ trans)
    return stablizers, result
