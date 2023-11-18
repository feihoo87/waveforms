import operator
from functools import lru_cache, reduce

import numpy as np
from numpy import pi

from waveforms.cache import cache

from ...matricies import U


def cliffordOrder(n: int) -> int:
    """
    Order of complex Clifford group of degree 2^n arising in quantum coding theory.
    
    Sloane, N. J. A. (ed.). "Sequence A003956 (Order of Clifford group)".
    The On-Line Encyclopedia of Integer Sequences. OEIS Foundation.
    https://oeis.org/A003956
    """
    return reduce(operator.mul, (((1 << (2 * j)) - 1) << 2 * j + 1
                                 for j in range(1, n + 1)), 1)


one_qubit_clifford_seq = [
    # Paulis
    ("u3", 0   ,  0   ,  0   ), # I
    ("u3", pi  , -pi  ,  0   ), # X
    ("u3", pi  ,  0   ,  0   ), # Y
    ("u3", 0   ,  pi/2,  pi/2), # Z

    # 2 pi / 3 rotations
    ("u3", pi/2, -pi/2,  0   ),
    ("u3", pi/2, -pi/2,  pi  ),
    ("u3", pi/2,  pi/2,  0   ),
    ("u3", pi/2,  pi/2, -pi  ),
    ("u3", pi/2,  0   ,  pi/2),
    ("u3", pi/2,  0   , -pi/2),
    ("u3", pi/2, -pi  ,  pi/2),
    ("u3", pi/2,  pi  , -pi/2),

    # pi / 2 rotations
    ("u3", pi/2, -pi/2,  pi/2), #  X/2
    ("u3", pi/2,  pi/2, -pi/2), # -X/2
    ("u3", pi/2,  0   ,  0   ), #  Y/2
    ("u3", pi/2,  pi  , -pi  ), # -Y/2
    ("u3", 0   ,  0   ,  pi/2), #  Z/2
    ("u3", 0   ,  0   , -pi/2), # -Z/2

    # Hadamard-like
    ("u3", pi/2, -pi  ,  0   ),
    ("u3", pi/2,  0   ,  pi  ), # Hadamard
    ("u3", pi/2,  pi/2,  pi/2),
    ("u3", pi/2, -pi/2, -pi/2),
    ("u3", pi  , -pi/2,  0   ),
    ("u3", pi  ,  pi/2,  0   )
] #yapf: disable

one_qubit_clifford_seq2 = [
    # Paulis
    [],                              # I
    ['H', 'S', 'S', 'H'],            # X
    ['H', 'S', 'S', 'H', 'S', 'S'],  # Y
    ['S', 'S'],                      # Z

    # 2 pi / 3 rotations
    ['S', 'S', 'H', 'S', 'S', 'S'],
    ['H', 'S', 'S', 'S'],
    ['S', 'S', 'H', 'S'],
    ['H', 'S'],
    ['S', 'S', 'S', 'H'],
    ['S', 'H'],
    ['S', 'S', 'S', 'H', 'S', 'S'],
    ['S', 'H', 'S', 'S'],

    # pi / 2 rotations
    ['S', 'S', 'S', 'H', 'S', 'S', 'S'], #  X/2
    ['S', 'H', 'S'],                     # -X/2
    ['S', 'S', 'H'],                     #  Y/2
    ['H', 'S', 'S'],                     # -Y/2
    ['S'],                               #  Z/2
    ['S', 'S', 'S'],                     # -Z/2

    # Hadamard-like
    ['S', 'S', 'H', 'S', 'S'],
    ['H'],
    ['S', 'S', 'S', 'H', 'S'],
    ['S', 'H', 'S', 'S', 'S'],
    ['H', 'S', 'S', 'H', 'S'],
    ['H', 'S', 'S', 'H', 'S', 'S', 'S']
] #yapf: disable

one_qubit_clifford_seq_inv = {
    g: i
    for i, g in enumerate(one_qubit_clifford_seq)
}
one_qubit_clifford_seq_inv['H'] = 19
one_qubit_clifford_seq_inv['S'] = 16
one_qubit_clifford_seq_inv['I'] = 0
one_qubit_clifford_seq_inv['X'] = 1
one_qubit_clifford_seq_inv['Y'] = 2
one_qubit_clifford_seq_inv['Z'] = 3
one_qubit_clifford_seq_inv['X/2'] = 12
one_qubit_clifford_seq_inv['-X/2'] = 13
one_qubit_clifford_seq_inv['Y/2'] = 14
one_qubit_clifford_seq_inv['-Y/2'] = 15
one_qubit_clifford_seq_inv['-S'] = 17
one_qubit_clifford_seq_inv.update({f'C1_{i}': i for i in range(24)})

one_qubit_clifford_mul_table = np.array([
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
    [1,0,3,2,6,7,4,5,11,10,9,8,13,12,18,19,22,23,14,15,21,20,16,17],
    [2,3,0,1,7,6,5,4,10,11,8,9,20,21,15,14,23,22,19,18,12,13,17,16],
    [3,2,1,0,5,4,7,6,9,8,11,10,21,20,19,18,17,16,15,14,13,12,23,22],
    [4,7,5,6,11,8,9,10,2,3,1,0,22,17,21,12,14,18,13,20,23,16,15,19],
    [5,6,4,7,10,9,8,11,1,0,2,3,23,16,12,21,19,15,20,13,22,17,18,14],
    [6,5,7,4,8,11,10,9,3,2,0,1,16,23,20,13,18,14,12,21,17,22,19,15],
    [7,4,6,5,9,10,11,8,0,1,3,2,17,22,13,20,15,19,21,12,16,23,14,18],
    [8,9,11,10,1,3,2,0,7,4,5,6,19,14,22,16,20,12,23,17,15,18,13,21],
    [9,8,10,11,2,0,1,3,6,5,4,7,14,19,23,17,13,21,22,16,18,15,20,12],
    [10,11,9,8,3,1,0,2,4,7,6,5,18,15,17,23,12,20,16,22,14,19,21,13],
    [11,10,8,9,0,2,3,1,5,6,7,4,15,18,16,22,21,13,17,23,19,14,12,20],
    [12,13,21,20,18,19,14,15,22,17,23,16,1,0,4,5,8,10,6,7,2,3,11,9],
    [13,12,20,21,14,15,18,19,16,23,17,22,0,1,6,7,11,9,4,5,3,2,8,10],
    [14,19,15,18,22,16,23,17,20,21,12,13,8,9,2,0,6,4,1,3,10,11,7,5],
    [15,18,14,19,17,23,16,22,12,13,20,21,10,11,0,2,5,7,3,1,8,9,4,6],
    [16,23,22,17,12,21,20,13,19,14,15,18,5,6,8,11,3,0,10,9,7,4,1,2],
    [17,22,23,16,21,12,13,20,14,19,18,15,4,7,9,10,0,3,11,8,6,5,2,1],
    [18,15,19,14,16,22,17,23,21,20,13,12,11,10,3,1,4,6,0,2,9,8,5,7],
    [19,14,18,15,23,17,22,16,13,12,21,20,9,8,1,3,7,5,2,0,11,10,6,4],
    [20,21,13,12,19,18,15,14,17,22,16,23,3,2,7,6,10,8,5,4,0,1,9,11],
    [21,20,12,13,15,14,19,18,23,16,22,17,2,3,5,4,9,11,7,6,1,0,10,8],
    [22,17,16,23,13,20,21,12,15,18,19,14,7,4,11,8,2,1,9,10,5,6,0,3],
    [23,16,17,22,20,13,12,21,18,15,14,19,6,5,10,9,1,2,8,11,4,7,3,0],
], dtype=np.int8) #yapf: disable


_one_qubit_clifford_index = {
    (0, 0, 0): 0,
    (0, 0, 1): 16,
    (0, 0, 2): 3,
    (0, 0, 3): 17,
    (0, 1, 0): 16,
    (0, 1, 1): 3,
    (0, 1, 2): 17,
    (0, 1, 3): 0,
    (0, 2, 0): 3,
    (0, 2, 1): 17,
    (0, 2, 2): 0,
    (0, 2, 3): 16,
    (0, 3, 0): 17,
    (0, 3, 1): 0,
    (0, 3, 2): 16,
    (0, 3, 3): 3,
    (1, 0, 0): 14,
    (1, 0, 1): 8,
    (1, 0, 2): 19,
    (1, 0, 3): 9,
    (1, 1, 0): 6,
    (1, 1, 1): 20,
    (1, 1, 2): 7,
    (1, 1, 3): 13,
    (1, 2, 0): 18,
    (1, 2, 1): 10,
    (1, 2, 2): 15,
    (1, 2, 3): 11,
    (1, 3, 0): 4,
    (1, 3, 1): 12,
    (1, 3, 2): 5,
    (1, 3, 3): 21,
    (2, 0, 0): 2,
    (2, 0, 1): 22,
    (2, 0, 2): 1,
    (2, 0, 3): 23,
    (2, 1, 0): 23,
    (2, 1, 1): 2,
    (2, 1, 2): 22,
    (2, 1, 3): 1,
    (2, 2, 0): 1,
    (2, 2, 1): 23,
    (2, 2, 2): 2,
    (2, 2, 3): 22,
    (2, 3, 0): 22,
    (2, 3, 1): 1,
    (2, 3, 2): 23,
    (2, 3, 3): 2,
}


def one_qubit_clifford_index(gate):
    if gate in one_qubit_clifford_seq_inv:
        return one_qubit_clifford_seq_inv[gate]
    else:
        match gate:
            case ('R', phi):
                return one_qubit_clifford_index(
                    ('U', pi / 2, phi - pi / 2, pi / 2 - phi))
            case ('rfUnitary', theta, phi):
                return one_qubit_clifford_index(
                    ('U', theta, phi - pi / 2, pi / 2 - phi))
            case ('u3', theta, phi, lam) | ('U', theta, phi, lam):
                theta_i = round(4 * np.mod(theta / (2 * pi), 1), 12)
                if np.mod(theta_i, 1) > 1e-9:
                    return -1
                theta_i = round(theta_i)
                if theta_i == 1:
                    pass
                elif theta_i == 0:
                    phi, lam = phi + lam, 0
                elif theta_i == 2:
                    phi, lam = phi - lam, 0
                elif theta_i == 3:
                    theta_i = 1
                    phi, lam = phi + pi, lam + pi
                else:
                    return -1
                phi_i = round(4 * np.mod(phi / (2 * pi), 1), 12)
                lam_i = round(4 * np.mod(lam / (2 * pi), 1), 12)
                if np.mod(phi_i, 1) < 1e-9 and np.mod(lam_i, 1) < 1e-9:
                    return _one_qubit_clifford_index[theta_i,
                                                     round(phi_i),
                                                     round(lam_i)]
                else:
                    return -1
            case ('u2', phi, lam):
                return one_qubit_clifford_index(('U', pi / 2, phi, lam))
            case ('u1', lam) | ('Rz', lam) | ('P', lam):
                return one_qubit_clifford_index(('U', 0, 0, lam))
            case 'P':
                return 16
            case ('Rx', theta):
                return one_qubit_clifford_index(('U', theta, -pi / 2, pi / 2))
            case ('Ry', theta):
                return one_qubit_clifford_index(('U', theta, 0, 0))
            case _:
                return -1


def twoQubitCliffordSequence(n):
    """
    生成第 n 个群元对应的操作序列
    """
    S1 = [0, 8, 7]  #  I, Rot(2/3 pi, (1,1,1)), Rot(2/3 pi, (1,-1,-1))
    if n < 576:
        i, j = np.unravel_index(n, (24, 24))
        return ((i, ), (j, ))
    elif n < 5760:
        n -= 576
        i, j, k, l = np.unravel_index(n, (24, 24, 3, 3))
        return ((i, 'CX', S1[k]), (j, 'CX', S1[l]))
    elif n < 10944:
        n -= 5760
        i, j, k, l = np.unravel_index(n, (24, 24, 3, 3))
        return ((i, 'iSWAP', S1[k]), (j, 'iSWAP', S1[l]))
    else:
        n -= 10944
        i, j = np.unravel_index(n, (24, 24))
        return ((i, 'SWAP'), (j, 'SWAP'))


@cache()
def _matricies():
    one_qubit_clifford_matricies = [U(*g[1:]) for g in one_qubit_clifford_seq]
    two_qubit_clifford_matricies = []
    for n in range(11520):
        seq = twoQubitCliffordSequence(n)
        mat = np.eye(4, dtype=complex)
        for a, b in zip(*seq):
            if isinstance(a, str):
                mat = {
                    'CX':
                    np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1],
                              [0, 0, 1, 0]]),
                    'iSWAP':
                    np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0],
                              [0, 0, 0, 1]]),
                    'SWAP':
                    np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                              [0, 0, 0, 1]])
                }[a] @ mat
            else:
                mat = np.kron(one_qubit_clifford_matricies[a],
                              one_qubit_clifford_matricies[b]) @ mat

        two_qubit_clifford_matricies.append(mat)
    return one_qubit_clifford_matricies, two_qubit_clifford_matricies


one_qubit_clifford_matricies, two_qubit_clifford_matricies = _matricies()
