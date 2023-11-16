import itertools
import random
from functools import reduce
from itertools import product

import numpy as np

from waveforms.cache import cache
from waveforms.qlisp.simulator.simple import seq2mat

from .._SU_n_ import SU
from ..permutation_group import Cycles, PermutationGroup, find_permutation
from .funtions import (cliffordOrder, one_qubit_clifford_mul_table,
                       one_qubit_clifford_seq, one_qubit_clifford_seq_inv)

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


@cache()
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
        self.gate_map = generators
        self.gate_map_inv = {v: k for k, v in generators.items()}
        for i in range(self.N):
            for g in one_qubit_clifford_seq:
                self.circuit_to_permutation([(g, i)])

    def __len__(self):
        return cliffordOrder(self.N)

    def matrix_to_circuit(self, mat):
        perm = self.matrix_to_permutation(mat)
        return [self.reversed_map[c] for c in perm.expand()]

    def matrix_to_permutation(self, mat):
        assert mat.shape == (
            2**self.N, 2**self.N
        ), f"mat.shape = {mat.shape} != (2**{self.N}, 2**{self.N})"
        perm = find_permutation_for_Unitary(mat, self.N)
        perm = self.express(perm)
        return perm

    def permutation_to_circuit(self, perm):
        perm = self.express(perm)
        return [self.reversed_map[c] for c in perm.expand()]

    def circuit_to_permutation(self, circuit):
        perm = Cycles()
        for gate in circuit:
            if gate not in self.gate_map:
                _, *qubits = gate
                circ = [('I', i) for i in range(self.N) if i not in qubits]
                circ.append(gate)
                mat = seq2mat(circ)
                self.gate_map[gate] = self.matrix_to_permutation(mat)
                self.gate_map_inv[self.gate_map[gate]] = gate
            perm = perm * self.gate_map[gate]
        return self.express(perm)

    def permutation_to_matrix(self, perm):
        return seq2mat(self.permutation_to_circuit(perm))

    def circuit_inv(self, circuit):
        perm = self.circuit_to_permutation(circuit).inv()
        return self.permutation_to_circuit(perm)

    def circuit_simplify(self, circuit):
        ret = []
        stack = {}
        for gate, *qubits in circuit:
            if len(qubits) > 1:
                for qubit in qubits:
                    ret.append((stack.pop(qubit,
                                          one_qubit_clifford_seq[0]), qubit))
                ret.append((gate, *qubits))
            else:
                qubit, = qubits
                i = one_qubit_clifford_seq_inv[stack.get(
                    qubit, one_qubit_clifford_seq[0])]
                j = one_qubit_clifford_seq_inv[gate]
                stack[qubit] = one_qubit_clifford_seq[
                    one_qubit_clifford_mul_table[i, j]]
        for qubit, gate in stack.items():
            ret.append((gate, qubit))
        return ret

    def circuit_fullsimplify(self, circuit):
        perm = self.circuit_to_permutation(circuit)
        return self.circuit_simplify(self.permutation_to_circuit(perm))
