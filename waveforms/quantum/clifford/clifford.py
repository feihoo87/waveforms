import pickle
import struct
from itertools import chain, product
from pathlib import Path

import numpy as np
from numpy import pi

from ...cache import cache_dir
from .mat import mat2num, num2mat
from .seq2mat import seq2mat

one_qubit_clifford_seq = [
    # Paulis
    ('I', ),
    ('X', ),
    ('Y', ),
    ('Y', 'X'),

    # 2 pi / 3 rotations
    ('X/2', 'Y/2'),
    ('X/2', '-Y/2'),
    ('-X/2', 'Y/2'),
    ('-X/2', '-Y/2'),
    ('Y/2', 'X/2'),
    ('Y/2', '-X/2'),
    ('-Y/2', 'X/2'),
    ('-Y/2', '-X/2'),

    # pi / 2 rotations
    ('X/2', ),
    ('-X/2', ),
    ('Y/2', ),
    ('-Y/2', ),
    ('-X/2', 'Y/2', 'X/2'),
    ('-X/2', '-Y/2', 'X/2'),

    # Hadamard-like
    ('X', 'Y/2'),
    ('X', '-Y/2'), # Hadamard
    ('Y', 'X/2'),
    ('Y', '-X/2'),
    ('X/2', 'Y/2', 'X/2'),
    ('-X/2', 'Y/2', '-X/2')
] #yapf: disable


one_qubit_clifford_seq_2 = [
    # Paulis
    ('I',),
    ('X',),
    ('Y',),
    ('Z',),

    # 2 pi / 3 rotations
    ('Y/2', '-S'),
    ('-Y/2', 'S'),
    ('Y/2', 'S'),
    ('-Y/2', '-S'),
    ('X/2', 'S'),
    ('-X/2', '-S'),
    ('X/2', '-S'),
    ('-X/2', 'S'),

    # pi / 2 rotations
    ('X/2',),
    ('-X/2',),
    ('Y/2',),
    ('-Y/2',),
    ('S',),
    ('-S',),

    # Hadamard-like
    ('Y/2', 'Z'),
    ('-Y/2', 'Z'), # Hadamard
    ('X/2', 'Z'),
    ('-X/2', 'Z'),
    ('X', 'S'),
    ('Y', 'S')
] #yapf: disable


one_qubit_clifford_seq_3 = [
    # Paulis
    ("u3", 0 / 6 * pi,  0 / 6 * pi,  0 / 6 * pi), # I
    ("u3", 6 / 6 * pi, -6 / 6 * pi,  0 / 6 * pi), # X
    ("u3", 6 / 6 * pi,  0 / 6 * pi,  0 / 6 * pi), # Y
    ("u3", 0 / 6 * pi,  3 / 6 * pi,  3 / 6 * pi), # Z

    # 2 pi / 3 rotations
    ("u3", 3 / 6 * pi, -3 / 6 * pi,  0 / 6 * pi),
    ("u3", 3 / 6 * pi, -3 / 6 * pi,  6 / 6 * pi),
    ("u3", 3 / 6 * pi,  3 / 6 * pi,  0 / 6 * pi),
    ("u3", 3 / 6 * pi,  3 / 6 * pi, -6 / 6 * pi),
    ("u3", 3 / 6 * pi,  0 / 6 * pi,  3 / 6 * pi),
    ("u3", 3 / 6 * pi,  0 / 6 * pi, -3 / 6 * pi),
    ("u3", 3 / 6 * pi, -6 / 6 * pi,  3 / 6 * pi),
    ("u3", 3 / 6 * pi,  6 / 6 * pi, -3 / 6 * pi),

    # pi / 2 rotations
    ("u3", 3 / 6 * pi, -3 / 6 * pi,  3 / 6 * pi), #  X/2
    ("u3", 3 / 6 * pi,  3 / 6 * pi, -3 / 6 * pi), # -X/2
    ("u3", 3 / 6 * pi,  0 / 6 * pi,  0 / 6 * pi), #  Y/2
    ("u3", 3 / 6 * pi,  6 / 6 * pi, -6 / 6 * pi), # -Y/2
    ("u3", 0 / 6 * pi,  0 / 6 * pi,  3 / 6 * pi), #  Z/2
    ("u3", 0 / 6 * pi,  0 / 6 * pi, -3 / 6 * pi), # -Z/2

    # Hadamard-like
    ("u3", 3 / 6 * pi, -6 / 6 * pi,  0 / 6 * pi),
    ("u3", 3 / 6 * pi,  0 / 6 * pi,  6 / 6 * pi), # Hadamard
    ("u3", 3 / 6 * pi,  3 / 6 * pi,  3 / 6 * pi),
    ("u3", 3 / 6 * pi, -3 / 6 * pi, -3 / 6 * pi),
    ("u3", 6 / 6 * pi, -3 / 6 * pi,  0 / 6 * pi),
    ("u3", 6 / 6 * pi,  3 / 6 * pi,  0 / 6 * pi)
] #yapf: disable


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


def _fill_seq_pair(A, B):
    """
    填充必要的 I 作为占位符
    """
    if len(A) > len(B):
        B = ('I', ) * (len(A) - len(B)) + B
    elif len(B) > len(A):
        A = ('I', ) * (len(B) - len(A)) + A
    return A, B


def _short_seq_pair(A, B):
    """
    去掉多余的 I
    """
    C, D = [], []
    for x, y in zip(A, B):
        if x == 'I' and y == 'I':
            continue
        else:
            C.append(x)
            D.append(y)
    if len(C) == 0:
        return ('I', ), ('I', )
    else:
        return tuple(C), tuple(D)


def generateTwoQubitCliffordSequence():
    """
    生成全部 11520 个群元对应的操作序列
    """
    C1 = one_qubit_clifford_seq
    S1 = [0, 8, 7]  #  I, Rot(2/3 pi, (1,1,1)), Rot(2/3 pi, (1,-1,-1))

    # single qubit class 0 ~ 576
    for i, j in product(range(24), repeat=2):
        A, B = _fill_seq_pair(C1[i], C1[j])
        yield _short_seq_pair(A, B)

    # CNOT-like class 576 ~ 5760
    for i, j in product(range(24), repeat=2):
        for k, l in product(S1, S1):
            A, B = _fill_seq_pair(C1[i], C1[j])
            C, D = _fill_seq_pair(C1[k], C1[l])
            yield _short_seq_pair(
                A + ('CX', ) + C,
                B + ('CX', ) + D,
            )

    # iSWAP-like class 5760 ~ 10944
    for i, j in product(range(24), repeat=2):
        for k, l in product(S1, S1):
            A, B = _fill_seq_pair(C1[i], C1[j])
            C, D = _fill_seq_pair(C1[k], C1[l])
            yield _short_seq_pair(A + ('iSWAP', ) + C, B + ('iSWAP', ) + D)

    # SWAP-like class 10944 ~ 11520
    for i, j in product(range(24), repeat=2):
        A, B = _fill_seq_pair(C1[i], C1[j])
        yield _short_seq_pair(A + ('SWAP', ), B + ('SWAP', ))


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


def two_qubit_clifford_num():
    return [
        mat2num(seq2mat(seq)) for seq in generateTwoQubitCliffordSequence()
    ]


NUMBEROFELEMENTS = 11520  # cliffordOrder(2)

__base = cache_dir.parent / "clifford"
__mul_table_file = __base / "clifford_2qb_mul_table_unsigned_short.dat"
__index_file = __base / "clifford_2qb_index.pickle"
__seq_lib_file = __base / "clifford_2qb_seq_lib.db"
__mul_table_packer = struct.Struct('<H')

if not __base.exists():
    __base.mkdir(parents=True)

if not __mul_table_file.exists():
    __mul_table = bytearray(NUMBEROFELEMENTS * NUMBEROFELEMENTS * 2)
    for offset in range(0, len(__mul_table), 2):
        __mul_table_packer.pack_into(__mul_table, offset, 0x8000)
    __mul_table_file.write_bytes(__mul_table)
else:
    __mul_table = bytearray(__mul_table_file.read_bytes())

if not __index_file.exists():
    __index2num = two_qubit_clifford_num()
    __num2index = {n: i for i, n in enumerate(__index2num)}
    __index2mat = [num2mat(n) for n in __index2num]

    with open(__index_file, 'wb') as f:
        pickle.dump((__index2num, __num2index, __index2mat), f)

else:
    with open(__index_file, 'rb') as f:
        (__index2num, __num2index, __index2mat) = pickle.load(f)

# if not __seq_lib_file.exists():
#     __index2seq = [{seq} for seq in generateTwoQubitCliffordSequence()]
#     __seq2index = {
#         seq: i
#         for i, seq in enumerate(generateTwoQubitCliffordSequence())
#     }
#     with open(__seq_lib_file, 'wb') as f:
#         pickle.dump(__index2seq, __seq2index, f)
# else:
#     with open(__seq_lib_file, 'rb') as f:
#         __index2seq, __seq2index = pickle.load(f)

# def _elms(seq):
#     base = {'I'}
#     for g1, g2 in zip(*seq):
#         if (g1, g2) in [('CZ', 'CZ'), ('iSWAP', 'iSWAP'),
#                         ('SQiSWAP', 'SQiSWAP')]:
#             base.add(g1)
#         elif (g1, g2) in [('C', 'Z'), ('Z', 'C')]:
#             base.add('CZ')
#         elif g1 == 'C' or g2 == 'C':
#             base.add(g1 + g2)
#         else:
#             base.add(g1)
#             base.add(g2)
#     return base

# def _betterSeq(a, b):
#     """
#     a is better than b
#     """
#     return (len(a[0]) < len(b[0]) and _elms(a) <= _elms(b)
#             or len(a[0]) == len(b[0]) and _elms(a) < _elms(b))

# def _updateSeqLib(i, seq):
#     if all(_betterSeq(s, seq) for s in __index2seq[i]):
#         return
#     l = set()
#     for s in __index2seq[i]:
#         if _betterSeq(seq, s):
#             continue
#         l.add(s)
#     l.add(seq)
#     __index2seq[i] = l
#     with open(__seq_lib_file, 'wb') as f:
#         pickle.dump(__index2seq, f)


def mat2index(mat: np.ndarray) -> int:
    """
    convert matrix to index

    Args:
        mat ([type]): unitary matrix

    Returns:
        int: index of Clifford gate
    """
    return num2index(mat2num(mat))


def index2mat(i: int) -> np.ndarray:
    """
    convert index to matrix

    Args:
        i (int): index of Clifford gate

    Returns:
        np.ndarray: matrix
    """
    return __index2mat[i]


def index2num(i: int) -> int:
    return __index2num[i]


def num2index(num: int) -> int:
    return __num2index[num]


def seq2num(seq):
    return mat2num(seq2mat(seq))


def seq2index(seq):
    # if seq in __seq2index:
    #     return __seq2index[seq]
    i = mat2index(seq2mat(seq))
    # _updateSeqLib(i, seq)
    return i


# def index2seq(i, base=None):
#     if base is None:
#         return __index2seq[i]
#     ret = []
#     for seq in __index2seq[i]:
#         if _elms(seq) <= set(base) | {'I'}:
#             ret.append(seq)
#     return ret


def mul(i1: int, i2: int) -> int:
    offset = 2 * (NUMBEROFELEMENTS * i1 + i2)
    ret = __mul_table_packer.unpack_from(__mul_table, offset)[0]

    if ret & 0x8000:
        ret = mat2index(index2mat(i1) @ index2mat(i2))
        __mul_table_packer.pack_into(__mul_table, offset, ret)
        with open(__mul_table_file, 'r+b') as f:
            f.seek(offset)
            f.write(__mul_table_packer.pack(ret))
    return ret


def inv(i: int) -> int:
    return mat2index(index2mat(i).T.conj())


def _genSeq(i, gate=('CZ', 'CZ')):
    pulses = ['I', 'X', 'Y', 'X/2', 'Y/2', '-X/2', '-Y/2']
    phases = ['I', 'Z', 'S', '-S']

    for k in [2, 4, 6, 8]:
        num = len(pulses)**k * len(phases)**2
        if i >= num:
            i -= num
            continue
        index = np.unravel_index(i, (len(pulses), ) * k + (len(phases), ) * 2)
        a = [pulses[n] for n in index[:-2]]
        p = [phases[n] for n in index[-2:]]
        return _short_seq_pair(
            tuple(
                chain(*[[a[j], gate[0]] for j in range(0, k - 3, 2)],
                      [a[-2], p[0]])),
            tuple(
                chain(*[[a[j], gate[1]] for j in range(1, k - 2, 2)],
                      [a[-1], p[1]])))
    else:
        raise IndexError(f'i={i} should be less than 94158400.')


def _countTwoQubitGate(seq, gate):
    count = 0
    for a, b in zip(*seq):
        if (a, b) == gate:
            count += 1
    return count


def genSeqForGate(db, gate=('CZ', 'CZ')):
    db = Path(db)
    if db.exists():
        with open(db, 'rb') as f:
            start, index2seq = pickle.load(f)
    else:
        start, index2seq = 0, [list() for i in range(NUMBEROFELEMENTS)]
    try:
        while True:
            try:
                seq = _genSeq(start, gate)
            except IndexError:
                break
            i = seq2index(seq)
            if len(index2seq[i]) == 0:
                index2seq[i].append(seq)
            else:
                s = index2seq[i].pop()
                if (len(s[0]) == len(seq[0])
                        and _countTwoQubitGate(s, gate) == _countTwoQubitGate(
                            seq, gate)):
                    index2seq[i].append(seq)
                    index2seq[i].append(s)
                elif (len(s[0]) > len(seq[0])
                      or _countTwoQubitGate(s, gate) > _countTwoQubitGate(
                          seq, gate)):
                    index2seq[i] = [seq]
                else:
                    index2seq[i].append(s)
            start += 1
            if start % 10000 == 0:
                with open(db, 'wb') as f:
                    pickle.dump((start, index2seq), f)
    finally:
        with open(db, 'wb') as f:
            pickle.dump((start, index2seq), f)


# def build():
#     pulses = ['I', 'X', 'Y', 'X/2', 'Y/2', '-X/2', '-Y/2']
#     phases = ['I', 'Z', 'S', '-S']
#     gates = [('CZ', 'CZ'), ('C', 'X'), ('X', 'C'), ('iSWAP', 'iSWAP')]

#     def genSeq(gate):
#         for a in product(pulses, repeat=4):
#             for p1, p2 in product(phases, repeat=2):
#                 yield _short_seq_pair((a[0], gate[0], a[2], p1),
#                                       (a[1], gate[1], a[3], p2))
#         for a in product(pulses, repeat=6):
#             for p1, p2 in product(phases, repeat=2):
#                 yield _short_seq_pair((a[0], gate[0], a[2], gate[0], a[4], p1),
#                                       (a[1], gate[1], a[3], gate[1], a[5], p2))
#         for a in product(pulses, repeat=8):
#             for p1, p2 in product(phases, repeat=2):
#                 yield _short_seq_pair(
#                     (a[0], gate[0], a[2], gate[0], a[4], gate[0], a[6], p1),
#                     (a[1], gate[1], a[3], gate[1], a[5], gate[1], a[7], p2))

#     for gate in gates:
#         for seq in genSeq(gate):
#             seq2index(seq)
