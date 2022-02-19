from itertools import count, product

import numpy as np


def normalize(mat):
    """
    将第一个非零的矩阵元相位转成 0，以保证仅相差一个全局相位的矩阵具有相同的表示。
    优先处理对角元，再依次处理下三角阵。
    """
    assert mat.shape[0] == mat.shape[1]
    for i in range(mat.shape[0]):
        for j in range(mat.shape[0] - i):
            if np.abs(mat[j + i, j]) > 1e-3:
                mat = mat * np.abs(mat[j + i, j]) / mat[j + i, j]
                return mat
    return 0 * mat


# 相位规范化后，2 比特 Clifford 群元的矩阵元仅可能取以下值
elms = [
    0, 0, 0, 0, 1, 1j, -1, -1j, 1 / np.sqrt(2), 1j / np.sqrt(2),
    -1 / np.sqrt(2), -1j / np.sqrt(2), 0.5, 0.5j, -0.5, -0.5j
]

# 速查表，key 为模与相位组成的元组
# 对于模，0,1,2,3 分别对应 0, 1, 1/sqrt(2), 1/2
# 对于相位， 0,1,2,3 分别对应 0, pi/2, pi, 3pi/2
elms_map = {k: v for k, v in zip(product(range(4), repeat=2), elms)}


def mat2num(mat, norm=True):
    """
    将一个 2 比特 Clifford 群元对应的矩阵转换为 64 位的整数
    
    由于规范化后矩阵元只有 13 种可能取值，故每个矩阵元可用 4 位二进制整数表示，
    4 x 4 的矩阵可以用不超过 64 位的整数表示
    """

    # 仅相隔一个全局相位的操作等价，故令第一个非零的矩阵元相位为 0，保证操作与矩阵一一对应
    if norm:
        mat = normalize(mat)

    absData, phaseData = 0, 0
    for index, (i, j) in zip(count(start=0, step=2), product(range(4),
                                                             repeat=2)):
        for k, v in elms_map.items():
            if abs(v - mat[i, j]) < 1e-3:
                a, phase = k
                break
        else:
            raise ValueError(f"Element {mat[i, j]} not allowed.")
        absData |= a << index
        phaseData |= phase << index
    return absData | (phaseData << 32)


def num2mat(num):
    """
    将 64 位整数还原成矩阵
    """
    absData, phaseData = num & 0xffffffff, num >> 32
    mat = np.zeros((4, 4), dtype=complex)

    for index, (i, j) in zip(count(start=0, step=2), product(range(4),
                                                             repeat=2)):
        a, phase = (absData >> index) & 0x3, (phaseData >> index) & 0x3
        mat[i, j] = elms_map[(a, phase)]
    return mat
