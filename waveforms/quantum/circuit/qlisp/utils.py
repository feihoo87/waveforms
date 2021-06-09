from itertools import repeat

import numpy as np


def DD(t, gates, pos, Delta=0):
    seq = ['X/2']
    i = 0
    for gate in gates:
        gap = t * (pos[i] - pos[i - 1]) if i > 0 else t * pos[0]
        seq.append(('Delay', gap))
        seq.append(gate)
        i += 1
    gap = t * (1 - pos[-1]) if len(pos) > 0 else t
    seq.append(('Delay', gap))
    if Delta != 0:
        seq.append(('P', Delta * t))
    seq.append('X/2')
    seq.append('Measure')
    return seq


def XY4(t, Delta=0):
    pos = np.arange(1, 5) / 5
    return DD(t, ['X', 'Y', 'X', 'Y'], pos, Delta)


def XY8(t, Delta=0):
    pos = np.arange(1, 9) / 9
    return DD(t, ['X', 'Y', 'X', 'Y', 'Y', 'X', 'Y', 'X'], pos, Delta)


def XY16(t, Delta=0):
    pos = np.arange(1, 17) / 17
    return DD(t, [
        'X', 'Y', 'X', 'Y', 'Y', 'X', 'Y', 'X', 'X', 'Y', 'X', 'Y', 'Y', 'X',
        'Y', 'X'
    ], pos, Delta)


def UDD(n, t, Delta=0):
    j = np.arange(n) + 1
    return DD(t, repeat('Y', times=n),
              np.sin(np.pi * j / (2 * n + 2))**2, Delta)


def CPMG(n, t, Delta=0):
    j = np.arange(n) + 1
    return DD(t, repeat('Y', times=n), (j - 0.5) / n, Delta)


def CP(n, t, Delta=0):
    j = np.arange(n) + 1
    return DD(t, repeat('X', times=n), (j - 0.5) / n, Delta)


def Ramsey(t, f=0):
    return [
        'X/2', ('Delay', t), ('rfUnitary', np.pi / 2, 2 * np.pi * f * t),
        'Measure'
    ]


def SpinEcho(t, f=0):
    return [
        'X/2', ('Delay', t / 2), ('rfUnitary', np.pi, np.pi * f * t),
        ('Delay', t / 2), 'X/2', 'Measure'
    ]


ALLXYSeq = [('I', 'I'), ('X', 'X'), ('Y', 'Y'), ('X', 'Y'), ('Y', 'X'),
            ('X/2', 'I'), ('Y/2', 'I'), ('X/2', 'Y/2'), ('Y/2', 'X/2'),
            ('X/2', 'Y'), ('Y/2', 'X'), ('X', 'Y/2'), ('Y', 'X/2'),
            ('X/2', 'X'), ('X', 'X/2'), ('Y/2', 'Y'), ('Y', 'Y/2'), ('X', 'I'),
            ('Y', 'I'), ('X/2', 'X/2'), ('Y/2', 'Y/2')]
