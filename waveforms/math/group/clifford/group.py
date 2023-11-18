import itertools
import random
from functools import reduce
from itertools import islice, product

import numpy as np

from waveforms.cache import cache
from waveforms.qlisp.simulator.simple import seq2mat

from ...paulis import encode_paulis
from .._SU_n_ import SU
from ..permutation_group import Cycles, PermutationGroup, find_permutation
from .chp import CZ, H, P
from .funtions import (cliffordOrder, one_qubit_clifford_mul_table,
                       one_qubit_clifford_seq, one_qubit_clifford_seq_inv)


def find_permutation_for_Unitary(U, N):
    init = []
    final = []

    for s in islice(product([SU(2)[i] for i in range(4)], repeat=N), 1, None):
        op = reduce(np.kron, s)
        init.append(op)
        final.append(U @ op @ U.T.conj())
        init.append(-op)
        final.append(U @ (-op) @ U.T.conj())
    return find_permutation(init, final)


def make_clifford_generators(N, graph=None):
    if graph is None:
        graph = []
        for i in range(N - 1):
            graph.append((i, i + 1))

    stablizers = []
    for s in islice(product('IXYZ', repeat=N), 1, None):
        n = encode_paulis(''.join(s))
        stablizers.append(n)
        stablizers.append(n | 2)

    generators = {}

    for i in range(N):
        generators[('H', i)] = find_permutation(stablizers,
                                                [H(s, i) for s in stablizers])
        generators[('S', i)] = find_permutation(stablizers,
                                                [P(s, i) for s in stablizers])

    for i, j in graph:
        generators[('CZ', i,
                    j)] = find_permutation(stablizers,
                                           [CZ(s, i, j) for s in stablizers])

    return generators


class CliffordGroup(PermutationGroup):

    def __init__(self, N, graph=None):
        self.N = N
        generators = make_clifford_generators(N, graph)
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
