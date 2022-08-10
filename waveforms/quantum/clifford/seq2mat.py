from ...qlisp.simulator import seq2mat as _seq2mat
from .mat import normalize


def twoQubitGate(gates):
    return {
        ('CZ', 'CZ'): ('CZ', (0, 1)),
        ('C', 'Z'): ('CZ', (0, 1)),
        ('Z', 'C'): ('CZ', (0, 1)),
        ('CX', 'CX'): ('Cnot', (0, 1)),
        ('XC', 'XC'): ('Cnot', (1, 0)),
        ('CR', 'CR'): ('CR', (0, 1)),
        ('RC', 'RC'): ('CR', (1, 0)),
        ('C', 'X'): ('Cnot', (0, 1)),
        ('X', 'C'): ('Cnot', (1, 0)),
        ('C', 'R'): ('CR', (0, 1)),
        ('R', 'C'): ('CR', (1, 0)),
        ('iSWAP', 'iSWAP'): ('iSWAP', (0, 1)),
        ('SWAP', 'SWAP'): ('SWAP', (0, 1)),
        ('SQiSWAP', 'SQiSWAP'): ('SQiSWAP', (0, 1)),
    }[gates]


def seq2qlisp(seq, qubits):
    if len(seq) > 2:
        raise ValueError("Only support 1 or 2 bits.")
    if len(seq) != len(qubits):
        raise ValueError("seq size and qubit num mismatched.")

    qlisp = []
    for gates in zip(*seq):
        try:
            qlisp.append(twoQubitGate(gates))
        except:
            for gate, i in zip(gates, qubits):
                qlisp.append((gate, i))
    return qlisp


def seq2mat(seq):
    N = len(seq)
    if N > 2:
        raise ValueError("Only support 1 or 2 bits.")
    if N == 1:
        qubits = (1, )
    else:
        qubits = (0, 1)

    return normalize(_seq2mat(seq2qlisp(seq, qubits)))
