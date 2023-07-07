import itertools
import random
from functools import reduce
from itertools import product

import numpy as np

from waveforms.qlisp.simulator.simple import seq2mat

from .._SU_n_ import SU
from ..permutation_group import find_permutation, PermutationGroup

_base = [SU(2)[i] for i in range(4)] + [-SU(2)[i] for i in range(4)]


def make_circuit(gate, N):
    rest_qubits = set(range(N)) - set(gate[1:])
    circ = [gate]
    for qubit in rest_qubits:
        circ.append(('I', qubit))
    return circ


def find_permutation_for_Unitary(U, N):
    init = []
    final = []

    for index in product(range(8), repeat=N):
        if all(i == 0 or i == 4 for i in index):
            continue
        op = reduce(np.kron, [_base[i] for i in index])
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


def make_clifford_generators(N,
                             one_qubit_gates=('H', 'S'),
                             two_qubit_gates=('CZ', ),
                             gate_list=None):
    if gate_list is None:
        gate_list = []

        for q in range(N):
            for gate in one_qubit_gates:
                gate_list.append((gate, q))

        for q in range(N - 1):
            for gate in two_qubit_gates:
                gate_list.append((gate, q, q + 1))

    generators = {}

    for gate in gate_list:
        U = seq2mat(make_circuit(gate, N))
        generators[gate] = find_permutation_for_Unitary(U, N)

    return generators


class CliffordGroup(PermutationGroup):

    def __init__(self,
                 N,
                 one_qubit_gates=('H', 'S'),
                 two_qubit_gates=('CZ', ),
                 gate_list=None):
        self.N = N
        generators = make_clifford_generators(N, one_qubit_gates,
                                              two_qubit_gates, gate_list)
        super().__init__(list(generators.values()))
        self.reversed_map = {v: k for k, v in generators.items()}

    def matrix_to_circuit(self, mat):
        perm = self.matrix_to_permutation(mat)
        return [self.reversed_map[c] for c in expand_expr(perm)]

    def matrix_to_permutation(self, mat):
        assert mat.shape == (
            2**self.N, 2**self.N
        ), f"mat.shape = {mat.shape} != (2**{self.N}, 2**{self.N})"
        perm = find_permutation_for_Unitary(mat, self.N)
        perm = self.express(perm)
        return perm
