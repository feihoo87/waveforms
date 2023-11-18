import functools
import itertools
import operator
import random

import numpy as np

from waveforms.math.group import CliffordGroup, PermutationGroup
from waveforms.math.group.clifford import (cliffordOrder,
                                           find_permutation_for_Unitary)
from waveforms.math.group.clifford.funtions import (
    one_qubit_clifford_matricies, one_qubit_clifford_mul_table,
    one_qubit_clifford_seq, one_qubit_clifford_seq2)
from waveforms.math.group.clifford.group import make_clifford_generators
from waveforms.math.matricies import synchronize_global_phase
from waveforms.qlisp.simulator.simple import applySeq, seq2mat


def make_circuit(gate, N):
    rest_qubits = set(range(N)) - set(gate[1:])
    circ = [gate]
    for qubit in rest_qubits:
        circ.append(('I', qubit))
    return circ


def random_circuit(N, depth, single_qubit_gate_set, two_qubit_gate_set):
    circ = []
    qubits = list(range(N))
    for i in range(depth):
        for q in range(N):
            circ.append((random.choice(single_qubit_gate_set), q))
        random.shuffle(qubits)
        for i in range(0, N - 1, 2):
            circ.append(
                (random.choice(two_qubit_gate_set), qubits[i], qubits[i + 1]))
    return circ


def test_cliffordOrder():
    assert cliffordOrder(0) == 1
    assert cliffordOrder(1) == 24
    assert cliffordOrder(2) == 11520
    assert cliffordOrder(3) == 92897280

    for N in [1, 2, 3]:
        clifford = PermutationGroup(list(make_clifford_generators(N).values()))
        assert clifford.order() == cliffordOrder(N)


def test_mul_table():
    group = CliffordGroup(1)
    lst = [
        group.circuit_to_permutation([(g, 0)]) for g in one_qubit_clifford_seq
    ]
    for i, j in itertools.product(range(24), repeat=2):
        assert lst[i] * lst[j] == lst[one_qubit_clifford_mul_table[i, j]]


def test_mul_table2():
    for i, j in itertools.product(range(24), repeat=2):
        assert np.allclose(
            synchronize_global_phase(one_qubit_clifford_matricies[j]
                                     @ one_qubit_clifford_matricies[i]),
            synchronize_global_phase(
                one_qubit_clifford_matricies[one_qubit_clifford_mul_table[i,
                                                                          j]]))


def test_seq2():
    for mat, seq in zip(one_qubit_clifford_matricies, one_qubit_clifford_seq2):
        assert np.allclose(
            synchronize_global_phase(mat),
            synchronize_global_phase(seq2mat([(g, 0) for g in seq])))


def test_elements():
    group = CliffordGroup(1)
    lst = [
        group.circuit_to_permutation([(g, 0)]) for g in one_qubit_clifford_seq
    ]
    assert len(group.elements) == 24
    assert set(group.elements) == set(lst)


def test_rb():
    group = CliffordGroup(2)

    circ = []
    for i in range(500):
        circ.extend(group.permutation_to_circuit(group.random()))

    circ2 = group.circuit_simplify(circ)
    circ3 = group.circuit_fullsimplify(circ)
    assert len(circ3) < len(circ)

    circ.extend(group.circuit_inv(circ3))
    circ2.extend(group.circuit_inv(circ3))
    circ3.extend(group.circuit_inv(circ3))

    psi = applySeq(circ)
    assert abs((psi * psi.conj()).real[0] - 1) < 1e-6

    psi = applySeq(circ2)
    assert abs((psi * psi.conj()).real[0] - 1) < 1e-6

    psi = applySeq(circ3)
    assert abs((psi * psi.conj()).real[0] - 1) < 1e-6


def test_express():
    N = 2
    DEPTH = 1000

    generators = make_clifford_generators(N)

    clifford = PermutationGroup(list(generators.values()))

    circuit = random_circuit(
        N, DEPTH, ['X/2', 'Y/2', 'X', 'Y', '-X/2', '-Y/2', 'Z', 'S', '-S'],
        ['CZ', 'iSWAP', 'Cnot'])
    U = seq2mat(circuit)

    circ_perm = find_permutation_for_Unitary(U, N)
    inv_circ = clifford.express(circ_perm.inv())

    assert (circ_perm *
            functools.reduce(operator.mul, inv_circ.expand())).is_identity()

    reversed_map = {v: k for k, v in generators.items()}
    inv_circuit = [reversed_map[c] for c in inv_circ.expand()]

    U = seq2mat(circuit + inv_circuit)

    assert np.allclose(synchronize_global_phase(U), np.eye(U.shape[0]))
