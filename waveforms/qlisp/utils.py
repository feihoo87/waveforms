from itertools import repeat

import numpy as np


def DD(qubit, t, gates, pos, f=0):
    seq = [('X/2', qubit)]
    i = 0
    for gate in gates:
        gap = t * (pos[i] - pos[i - 1]) if i > 0 else t * pos[0]
        seq.append((('Delay', gap), qubit))
        seq.append((gate, qubit))
        i += 1
    gap = t * (1 - pos[-1]) if len(pos) > 0 else t
    seq.append((('Delay', gap), qubit))
    if f != 0:
        seq.append((('P', 2 * np.pi * f * t), qubit))
    seq.append(('X/2', qubit))
    return seq


def XY4(qubit, t, f=0):
    pos = np.arange(1, 5) / 5
    return DD(qubit, t, ['X', 'Y', 'X', 'Y'], pos, f)


def XY8(qubit, t, f=0):
    pos = np.arange(1, 9) / 9
    return DD(qubit, t, ['X', 'Y', 'X', 'Y', 'Y', 'X', 'Y', 'X'], pos, f)


def XY16(qubit, t, f=0):
    pos = np.arange(1, 17) / 17
    return DD(qubit, t, [
        'X', 'Y', 'X', 'Y', 'Y', 'X', 'Y', 'X', 'X', 'Y', 'X', 'Y', 'Y', 'X',
        'Y', 'X'
    ], pos, f)


def UDD(qubit, n, t, f=0):
    j = np.arange(n) + 1
    return DD(qubit, t, repeat('Y', times=n),
              np.sin(np.pi * j / (2 * n + 2))**2, f)


def CPMG(qubit, n, t, f=0):
    j = np.arange(n) + 1
    return DD(qubit, t, repeat('Y', times=n), (j - 0.5) / n, f)


def CP(qubit, n, t, f=0):
    j = np.arange(n) + 1
    return DD(qubit, t, repeat('X', times=n), (j - 0.5) / n, f)


def Ramsey(qubit, t, f=0):
    return [('X/2', qubit), (('Delay', t), qubit),
            (('rfUnitary', np.pi / 2, 2 * np.pi * f * t), qubit)]


def SpinEcho(qubit, t, f=0):
    return [('X/2', qubit), (('Delay', t / 2), qubit),
            (('rfUnitary', np.pi, np.pi * f * t), qubit),
            (('Delay', t / 2), qubit), ('X/2', qubit)]


_ALLXYSeq = [('I', 'I'), ('X', 'X'), ('Y', 'Y'), ('X', 'Y'), ('Y', 'X'),
             ('X/2', 'I'), ('Y/2', 'I'), ('X/2', 'Y/2'), ('Y/2', 'X/2'),
             ('X/2', 'Y'), ('Y/2', 'X'), ('X', 'Y/2'), ('Y', 'X/2'),
             ('X/2', 'X'), ('X', 'X/2'), ('Y/2', 'Y'), ('Y', 'Y/2'),
             ('X', 'I'), ('Y', 'I'), ('X/2', 'X/2'), ('Y/2', 'Y/2')]


def ALLXY(qubit, i):
    assert 0 <= i < len(
        _ALLXYSeq), f"i={i} is out of range(0, {len(_ALLXYSeq)})"
    return [(gate, qubit) for gate in _ALLXYSeq[i]]
