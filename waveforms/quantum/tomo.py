import operator
from functools import lru_cache, reduce
from itertools import chain, combinations, product, repeat

import numpy as np
from scipy import linalg, optimize
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import inv, lsqr

from waveforms.math.matricies import (sigmaI, sigmaM, sigmaP, sigmaX, sigmaY,
                                      sigmaZ)
from waveforms.qlisp.simulator.simple import _matrix_of_gates

from ..cache import cache
from .math import dagger, normalize, randomUnitary, unitary2v, v2unitary

__base_op = {
    gate: mat
    for gate, (mat, *_) in _matrix_of_gates.items()
    if isinstance(mat, np.ndarray)
}

__base_op['sigmaX'] = sigmaX()
__base_op['sigmaY'] = sigmaY()
__base_op['sigmaZ'] = sigmaZ()
__base_op['sigmaP'] = sigmaP()
__base_op['sigmaM'] = sigmaM()

qst_gates = ['-Y/2', 'X/2', 'I']
ocqst_gates = ['I', 'X/2', 'Y/2', '-X/2', '-Y/2', 'X']
qpt_init_gates = ['I', 'X', 'Y/2', 'X/2']
pauli_basis = ['I', 'sigmaX', 'sigmaY', 'sigmaZ']
real_pauli_basis = ['I', 'sigmaX', 'Y', 'Z']
raise_lower_basis = ['I', 'sigmaP', 'sigmaM', 'sigmaZ']


def tensorMatrix(transform):
    """transform 所对应变换矩阵
    """
    return reduce(np.kron, (__base_op[k] for k in transform))


@lru_cache()
def tensorElement(opList, r, c):
    """直积矩阵第 r 行、第 c 列元素
    """
    N = len(opList)
    return reduce(
        operator.mul,
        (__base_op[k][int(x)][int(y)]
         for k, x, y in zip(opList, f'{r:b}'.zfill(N), f'{c:b}'.zfill(N))))


def rhoToV(rho):
    """密度矩阵转成一个纯实数向量
    """
    N = rho.shape[0]
    indices = np.triu_indices(N, 1)
    return np.hstack(
        (rho[indices].real, rho[indices].imag, np.diag(rho)[1:].real))


def vToRho(V):
    """rhoToV 的逆函数"""
    N = int(np.sqrt(len(V) + 1))
    assert N**2 - 1 == len(V)

    X = V[:(N**2 - N) // 2]
    Y = V[(N**2 - N) // 2:(N**2 - N)]
    Z = V[(N**2 - N):]
    rho = np.diag(np.hstack((1 - np.sum(Z), Z)) / 2).astype(complex)

    rho[np.triu_indices(N, 1)] = X + 1j * Y
    rho += dagger(rho)

    return normalize(rho)


def qstOpList(N, gates=qst_gates):
    """State tomography 对应的全部操作
    """
    return product(gates, repeat=N)


def qptInitList(N, gates=qpt_init_gates):
    return product(gates, repeat=N)


def qptBases(N, gates=pauli_basis):
    return product(gates, repeat=N)


def xFactor(i, transform, j, k):
    """经过 transform 变换后的密度矩阵第 i 个对角元表达式中 x_jk 的系数
    """
    return 2 * np.real(
        np.conj(tensorElement(transform, i, k)) *
        tensorElement(transform, i, j))


def yFactor(i, transform, j, k):
    """经过 transform 变换后的密度矩阵第 i 个对角元表达式中 y_jk 的系数
    """
    return -2 * np.imag(
        np.conj(tensorElement(transform, i, k)) *
        tensorElement(transform, i, j))


def zFactor(i, transform, j):
    """经过 transform 变换后的密度矩阵第 i 个对角元表达式中 z_j 的系数
    """
    return np.real(
        np.conj(tensorElement(transform, i, j)) *
        tensorElement(transform, i, j) -
        np.conj(tensorElement(transform, i, 0)) *
        tensorElement(transform, i, 0))


def constFactor(i, transform):
    """经过 transform 变换后的密度矩阵第 i 个对角元表达式中的常数项
    """
    return np.real(
        np.conj(tensorElement(transform, i, 0)) *
        tensorElement(transform, i, 0))


@cache()
def formUMatrix(n, gates):
    """构造从相干矢到测量结果的转移矩阵"""
    dim = 2**n
    A_data, A_i, A_j = [], [], []
    C = []

    for row, (U,
              i) in enumerate(product(qstOpList(n, gates=gates), range(1,
                                                                       dim))):
        C.append(constFactor(i, U))
        for col, (jk, func) in enumerate(
                chain(zip(combinations(range(dim), 2), repeat(xFactor)),
                      zip(combinations(range(dim), 2), repeat(yFactor)),
                      zip(combinations(range(1, dim), 1), repeat(zFactor)))):
            A = func(i, U, *jk)

            if np.abs(A) > 1e-9:
                A_i.append(row)
                A_j.append(col)
                A_data.append(A)
    A, b = coo_matrix((A_data, (A_i, A_j))), np.asarray(C)
    return A, b


def qst(diags, gates=qst_gates):
    """Convert a set of diagonal measurements into a density matrix.
    
    diags - measured probabilities (diagonal elements) after acting
            on the state with each of the unitaries from the qst
            protocol.
    gates - qst protocol
            yield unitaries from product(gates, repeat=N).
            gates should be choosed from {I, X, Y, Z, H, S, -S,
            X/2, Y/2, -X/2, -Y/2}.
    """
    diags = np.asarray(diags)
    N = len(diags[0])  # size of density matrix
    n = int(np.log2(N))  # number of qubits

    A, b = formUMatrix(n, gates=gates)

    P = np.asarray(diags)[:, 1:]
    v, *_ = lsqr(A, P.flatten() - b)
    return vToRho(v)


@cache()
def _UUds(N, gates):
    return [(tensorMatrix(ops), dagger(tensorMatrix(ops)))
            for ops in qstOpList(N, gates=gates)]


def _qst_mle(diags, UUds: list[tuple[np.ndarray, np.ndarray]],
             rho0: np.ndarray, F: np.ndarray):
    """State tomography with maximum-likelihood estimation.
    
    diags - measured probabilities (diagonal elements) after acting
            on the state with each of the unitaries from the qst
            protocol.
    UUds - list of unitary pairs (U, U^\\dagger)
    rho0 - initial density matrix
    F - fidelity of the measurement
    """
    # convert the initial guess into vector
    v_guess = rhoToV(rho0)

    def log(x):
        """Safe version of log that returns -Inf when x < 0, rather than NaN.
        
        This is good for our purposes since negative probabilities are infinitely unlikely.
        """
        return np.log(x.real * (x.real > 0))

    def unlikelihood(v):  # negative of likelihood function
        rho = vToRho(v)
        pxis = np.array([F @ np.diag(U @ rho @ Ud) for U, Ud in UUds])
        pxis = pxis * (pxis.real > 0)
        terms = (diags * log(pxis + (diags == 0)) +
                 (1 - diags) * log(1 - pxis + (diags == 1)))
        return -np.sum(terms.real)

    #minimize
    #tis = optimize.fmin(unlikelihood, tis_guess)
    #tis = optimize.fmin_bfgs(unlikelihood, tis_guess)
    v = optimize.minimize(unlikelihood, v_guess, method='BFGS')['x']
    return vToRho(v)


def qst_mle(diags, gates=qst_gates, F=None, rho0=None):
    """State tomography with maximum-likelihood estimation.
    
    diags - a 2D array of measured probabilites.  The first index indicates which
            operation from the qst protocol was applied, while the second index
            tells which measurement result this was (e.g. 000, 001, etc.).
    gates - qst protocol
            yield unitaries from product(gates, repeat=N).
            gates should be choosed from {I, X, Y, Z, H, S, -S,
            X/2, Y/2, -X/2, -Y/2}.
    rho0 - an initial guess for the density matrix, e.g. from linear tomography.
    """
    diags = np.asarray(diags)
    N = len(diags[0])  # size of density matrix
    n = int(np.log2(N))  # number of qubits

    # make an initial guess using linear tomography
    if rho0 is None:
        rho0 = normalize(qst(diags, gates))

    # precompute conjugate transposes of matrices
    UUds = _UUds(n, gates)

    if F is None:
        F = np.eye(N)

    return _qst_mle(diags, UUds, rho0, F)


def chi0(rhosIn, rhosOut):
    """Calculates the pointer-basis chi-matrix.
    
    rhosIn - array of input density matrices
    rhosOut - array of output density matrices.
    
    Uses linalg.lstsq to calculate the closest fit
    when the chi-matrix is overdetermined by the data.
    """
    Din = rhosIn[0].size
    Dout = rhosOut[0].size
    n = len(rhosIn)

    rhosIn_mat = np.array(rhosIn).reshape((n, Din))
    rhosOut_mat = np.array(rhosOut).reshape((n, Dout))

    chi0, resids, rank, s = linalg.lstsq(rhosIn_mat,
                                         rhosOut_mat,
                                         overwrite_a=True,
                                         overwrite_b=True)
    return chi0


@cache()
def chi0_to_chi_mat(N, basis=pauli_basis):
    return inv(chi_to_chi0_mat(N, basis))


@cache()
def chi_to_chi0_mat(N, basis=pauli_basis):
    dim = 2**N
    A_data, A_i, A_j = [], [], []
    for row, (k, l, i, j) in enumerate(product(range(dim), repeat=4)):
        for col, (m,
                  n) in enumerate(product(qptBases(N, gates=basis), repeat=2)):
            elm = tensorElement(m, i, k) * np.conj(tensorElement(n, j, l))
            if abs(elm) > 1e-12:
                A_i.append(row)
                A_j.append(col)
                A_data.append(elm)
    return csc_matrix((A_data, (A_i, A_j)), dtype=complex)


def chi0_to_chi(chi0, basis=pauli_basis):
    dim = int(round(np.sqrt(chi0.shape[0])))
    N = int(round(np.log2(dim)))
    M = chi0_to_chi_mat(N, basis)
    chi = M @ chi0.reshape(dim**4)
    return chi.reshape((dim**2, dim**2))


def chi_to_chi0(chi, basis=pauli_basis):
    dim = int(round(np.sqrt(chi.shape[0])))
    N = int(round(np.log2(dim)))
    M = chi_to_chi0_mat(N, basis)
    chi0 = M @ chi.reshape(dim**4)
    return chi0.reshape((dim**2, dim**2))


#def qpt(a, b=None, /, *, init_gates=qpt_init_gates, basis=pauli_basis):
def qpt(a, b=None, *, init_gates=qpt_init_gates, basis=pauli_basis):
    if b is None:
        rfList = a
        rho0 = np.zeros_like(rfList[0], dtype=complex)
        rho0[0, 0] = 1
        N = int(round(np.log2(rfList[0].shape[0])))
        Us = [tensorMatrix(seq) for seq in qptInitList(N, gates=init_gates)]
        riList = [U @ rho0 @ dagger(U) for U in Us]
    else:
        riList, rfList = a, b
    chi = chi0(np.array(riList), np.array(rfList))
    return chi0_to_chi(chi, basis)


def applyOp(opList, rho=None):
    if rho is None:
        rho = reduce(np.kron,
                     repeat(np.array([[1, 0], [0, 0]]), times=len(opList)))
    U = tensorMatrix(opList)
    return U @ rho @ dagger(U)


def applyChi(chi, rho, basis=pauli_basis):
    dim = rho.shape[0]
    chi0 = chi_to_chi0(chi, basis)
    rhoF = rho.reshape(dim * dim) @ chi0
    return rhoF.reshape((dim, dim))


def chain_process(chi1, chi2, basis=pauli_basis):
    chi0 = chi_to_chi0(chi2, basis) @ chi_to_chi0(chi1, basis)
    return chi0_to_chi(chi0, basis)


def U_to_chi(U, basis=pauli_basis):
    riList = []
    rfList = []

    N = int(round(np.log2(U.shape[0])))

    for Ui in qptInitList(N):
        # 分别制备不同的初态
        ri = applyOp(Ui)
        # 做完操作后测末态的 QST
        rf = U @ ri @ dagger(U)
        riList.append(ri)
        rfList.append(rf)

    # 计算 chi 超算符
    chi = chi0(np.asarray(riList), np.asarray(rfList))
    chi = chi0_to_chi(chi, basis)
    return chi


def chi_to_U(chi, basis=pauli_basis, U0=None):
    from waveforms.quantum.clifford.mat import normalize

    def unlikelihood(v):
        U = v2unitary(v)
        chi1 = U_to_chi(U, basis)
        return -np.abs(np.trace(chi @ dagger(chi1))) / np.sqrt(
            np.trace(chi @ dagger(chi)).real *
            np.trace(chi1 @ dagger(chi1)).real)
        #return (np.abs(chi - chi1)**2).sum()

    dim = int(round(np.sqrt(chi.shape[0])))

    if U0 is None:
        v_guess = unitary2v(randomUnitary(dim))
    else:
        v_guess = unitary2v(U0)
    v = optimize.minimize(unlikelihood, v_guess, method='BFGS')['x']
    return normalize(v2unitary(v))


__all__ = ["qst", "qst_mle", "qpt", "qstOpList", "qptInitList"]
