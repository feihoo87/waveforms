import random

import numpy as np

from waveforms.qlisp.simulator.simple import seq2mat

from .clifford import cliffordOrder
from .clifford.clifford import (generateTwoQubitCliffordSequence, inv,
                                mat2index, mul)
from .clifford.seq2mat import seq2qlisp

_index2seq = [[seq] for seq in generateTwoQubitCliffordSequence()]

from ..qlisp import mapping_qubits


def circuit_to_index(circuit: list) -> int:
    if not circuit:
        return 0
    mat = seq2mat(circuit)
    if mat.shape[0] == 2:
        mat = np.kron(np.eye(2), mat)
    return mat2index(mat)


def index_to_circuit(index: int, qubits=(0, ), base=None, rng=None) -> list:
    if len(qubits) > 2:
        raise ValueError('Only support 1 or 2 qubits')
    if rng is None:
        rng = random.Random()
    if base is None:
        base = _index2seq
    seq = rng.choice(base[index])
    if len(qubits) == 1:
        seq = (seq[1], )
    return seq2qlisp(seq, range(len(qubits)))


def generateRBCircuit(qubits, cycle, seed=None, interleaves=[], base=None):
    """Generate a random Clifford RB circuit.

    Args:
        qubits (list): The qubits to use.
        cycle (int): The cycles of clifford sequence.
        seed (int): The seed for the random number generator.
        interleaves (list): The interleaves to use.
        base (list): The basic two-qubit Clifford sequence.

    Returns:
        list: The RB circuit.
    """
    if isinstance(qubits, (str, int)):
        qubits = {0: qubits}
    else:
        qubits = {i: q for i, q in enumerate(qubits)}

    MAX = cliffordOrder(len(qubits))

    interleaves_index = circuit_to_index(interleaves)

    ret = []
    index = 0
    rng = random.Random(seed)

    for _ in range(cycle):
        i = rng.randrange(MAX)
        index = mul(i, index)
        ret.extend(index_to_circuit(i, qubits, base, rng))
        index = mul(interleaves_index, index)
        ret.extend(interleaves)

    ret.extend(index_to_circuit(inv(index), qubits, base, rng))

    return mapping_qubits(ret, qubits)
