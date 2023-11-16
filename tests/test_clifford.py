import itertools

from waveforms.math.group import Cycles
from waveforms.math.group.clifford import CliffordGroup, cliffordOrder
from waveforms.math.group.clifford.funtions import (
    one_qubit_clifford_mul_table, one_qubit_clifford_seq)
from waveforms.qlisp.simulator.simple import applySeq


def test_cliffordOrder():
    assert cliffordOrder(0) == 1
    assert cliffordOrder(1) == 24
    assert cliffordOrder(2) == 11520
    assert cliffordOrder(3) == 92897280


def test_mul_table():
    group = CliffordGroup(1)
    lst = [
        group.circuit_to_permutation([(g, 0)]) for g in one_qubit_clifford_seq
    ]
    for i, j in itertools.product(range(24), repeat=2):
        assert lst[i] * lst[j] == lst[one_qubit_clifford_mul_table[i, j]]


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
