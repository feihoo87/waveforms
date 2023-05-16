import functools
import itertools
import operator
import random
from functools import reduce
from itertools import chain, product

import numpy as np

from waveforms.math.group import *
from waveforms.math.group.clifford.funtions import cliffordOrder
from waveforms.qlisp.simulator.simple import seq2mat


def make_circuit(gate, N):
    rest_qubits = set(range(N)) - set(gate[1:])
    circ = [gate]
    for qubit in rest_qubits:
        circ.append(('I', qubit))
    return circ


def find_permutation_for_Unitary(U, N):
    init = []
    final = []
    for base in product([
            SU(2)[0],
            SU(2)[1],
            SU(2)[2],
            SU(2)[3], -SU(2)[0], -SU(2)[1], -SU(2)[2], -SU(2)[3]
    ],
                        repeat=N):
        op = reduce(np.kron, base)
        init.append(op)
        final.append(U @ op @ U.T.conj())
    return find_permutation(init, final)


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


def expand_expr(perm):
    perm.simplify()
    expr = perm._expr
    return itertools.chain.from_iterable([[c] * n for c, n in expr])


def make_clifford_generators(N):
    gate_list = []

    for q in range(N):
        gate_list.append(('H', q))
        gate_list.append(('S', q))

    for q in range(N - 1):
        gate_list.append(('CZ', q, q + 1))

    generators = {}

    for gate in gate_list:
        U = seq2mat(make_circuit(gate, N))
        generators[gate] = find_permutation_for_Unitary(U, N)

    return generators


def test_init():
    G = PermutationGroup([Cycles((1, 2, 3)), Cycles((4, 5))])
    assert G.generators == [Cycles((1, 2, 3)), Cycles((4, 5))]


def test_order():
    G = PermutationGroup([Cycles((1, 2, 3)), Cycles((4, 5))])
    assert G.order() == 6

    G = PermutationGroup([
        Cycles((1, 2, 3, 4), (8, 17, 14, 11), (7, 20, 13, 10)),
        Cycles((5, 6, 7, 8), (1, 9, 21, 17), (2, 10, 22, 18)),
        Cycles((9, 10, 11, 12), (1, 13, 23, 5), (4, 16, 22, 8)),
        Cycles((13, 14, 15, 16), (3, 19, 23, 11), (4, 20, 24, 12)),
        Cycles((17, 18, 19, 20), (2, 6, 24, 14), (3, 7, 21, 15)),
        Cycles((21, 22, 23, 24), (5, 12, 15, 18), (6, 9, 16, 19))
    ])
    assert G.order() == 88179840

    for N in [1, 2, 3]:
        clifford = PermutationGroup(list(make_clifford_generators(N).values()))
        assert clifford.order() == cliffordOrder(N)


def test_orbit():
    G = SymmetricGroup(5)
    assert set(G.orbit("aaaaa")) == {"aaaaa"}
    assert set(
        G.orbit("aaaab")) == {"aaaab", "baaaa", "abaaa", "aabaa", "aaaba"}
    assert set(G.orbit("aaabb")) == {
        'aaabb', 'aabab', 'baaab', 'babaa', 'baaba', 'abbaa', 'bbaaa', 'aabba',
        'ababa', 'abaab'
    }


def test_contains():
    rot1 = Cycles((1, 3, 8, 6), (2, 5, 7, 4), (9, 48, 15, 12),
                  (10, 47, 16, 13), (11, 46, 17, 14))

    rot2 = Cycles((6, 15, 35, 26), (7, 22, 34, 19), (8, 30, 33, 11),
                  (12, 14, 29, 27), (13, 21, 28, 20))

    rot3 = Cycles((1, 12, 33, 41), (4, 20, 36, 44), (6, 27, 38, 46),
                  (9, 11, 26, 24), (10, 19, 25, 18))

    rot4 = Cycles((1, 24, 40, 17), (2, 18, 39, 23), (3, 9, 38, 32),
                  (41, 43, 48, 46), (42, 45, 47, 44))

    rot5 = Cycles((3, 43, 35, 14), (5, 45, 37, 21), (8, 48, 40, 29),
                  (15, 17, 32, 30), (16, 23, 31, 22))

    rot6 = Cycles((24, 27, 30, 43), (25, 28, 31, 42), (26, 29, 32, 41),
                  (33, 35, 40, 38), (34, 37, 39, 36))

    RubikGroup = PermutationGroup([rot1, rot2, rot3, rot4, rot5, rot6])

    assert RubikGroup.order() == 43252003274489856000

    assert Cycles(1, 9, 46) not in RubikGroup
    assert Cycles((1, 3), (9, 48), (17, 46)) not in RubikGroup
    assert Cycles(2, 47) not in RubikGroup
    assert Cycles((2, 47), (31, 37)) in RubikGroup
    assert Cycles((1, 9, 46), (3, 48, 17), (8, 15, 14)) in RubikGroup
    superflip = Cycles((2, 47), (4, 10), (7, 13), (5, 16), (20, 19), (21, 22),
                       (28, 34), (18, 44), (25, 36), (45, 23), (42, 39),
                       (31, 37))
    assert superflip in RubikGroup


def test_rank():
    G = SymmetricGroup(5)
    for i, g in enumerate(G.generate_schreier_sims()):
        assert G.coset_unrank(i) == g
        assert i == G.coset_rank(g)
    assert i + 1 == 120


def test_random():
    G = SymmetricGroup(10)
    for i in range(10):
        g = G.random()
        assert g in G


def test_subgroup():
    G = PermutationGroup([Cycles((1, 2, 3)), Cycles((4, 5))])
    assert G <= SymmetricGroup(6)
    assert G < SymmetricGroup(6)
    assert G <= G
    assert G == G
    assert G != SymmetricGroup(6)
    assert not G < SymmetricGroup(5)


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

    assert (
        circ_perm *
        functools.reduce(operator.mul, expand_expr(inv_circ))).is_identity()

    reversed_map = {v: k for k, v in generators.items()}
    inv_circuit = [reversed_map[c] for c in expand_expr(inv_circ)]

    U = seq2mat(circuit + inv_circuit)
    U = U * np.exp(-1j * np.angle(U[0, 0]))

    assert np.allclose(U, np.eye(U.shape[0]))
