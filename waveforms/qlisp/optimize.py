import numpy as np

from waveforms.math.matricies import Unitary2Angles
from waveforms.qlisp.simulator.simple import seq2mat


def circuit_simplify_level1(circuit):

    ret = []
    stack = {}
    for gate, *qubits in circuit:
        if len(qubits) > 1:
            for qubit in qubits:
                if qubit in stack:
                    U = stack.pop(qubit)
                    theta, phi, lam, *_ = Unitary2Angles(U)
                    ret.append((('u3', theta, phi, lam), qubit))
            ret.append((gate, *qubits))
        else:
            qubit, = qubits
            stack[qubit] = seq2mat([(gate, 0)]) @ stack.get(qubit, np.eye(2))
    for qubit, U in stack.items():
        theta, phi, lam, *_ = Unitary2Angles(U)
        ret.append((('u3', theta, phi, lam), qubit))
    return ret
