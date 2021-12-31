from functools import reduce
from itertools import chain, product
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh, expm, logm, sqrtm
from waveforms.math import (fit_circle, fit_cross_point, fit_pole, getFTMatrix,
                            lin_fit)
from waveforms.math.signal import decay, oscillation


def issparse(qob):
    """Checks if ``qob`` is explicitly sparse.
    """
    return isinstance(qob, sp.spmatrix)


def make_immutable(mat):
    """Make array read only, in-place.
    Parameters
    ----------
    mat : sparse or dense array
        Matrix to make immutable.
    """
    if issparse(mat):
        mat.data.flags.writeable = False
        if mat.format in {'csr', 'csc', 'bsr'}:
            mat.indices.flags.writeable = False
            mat.indptr.flags.writeable = False
        elif mat.format == 'coo':
            mat.row.flags.writeable = False
            mat.col.flags.writeable = False
    else:
        mat.flags.writeable = False
    return mat


# Paulis
sigmaI = lambda: make_immutable(np.eye(2, dtype=complex))
sigmaX = lambda: make_immutable(np.array([[0, 1], [1, 0]], dtype=complex))
sigmaY = lambda: make_immutable(np.array([[0, -1j], [1j, 0]], dtype=complex))
sigmaZ = lambda: make_immutable(np.array([[1, 0], [0, -1]], dtype=complex))
s0, s1, s2, s3 = sigmaI(), sigmaX(), sigmaY(), sigmaZ()

# Bell states
BellPhiP = lambda: make_immutable(
    np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2))
BellPhiM = lambda: make_immutable(
    np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2))
BellPsiP = lambda: make_immutable(
    np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2))
BellPsiM = lambda: make_immutable(
    np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2))
phiplus, phiminus = BellPhiP(), BellPhiM()
psiplus, psiminus = BellPsiP(), BellPsiM()


def tensor(matrixes: np.ndarray) -> np.ndarray:
    """Compute the tensor product of a list (or array) of matrixes"""
    return reduce(np.kron, matrixes)


def sigma(j: int, N: int = 1) -> np.ndarray:
    """
    """
    s = [s0, s1, s2, s3]
    dims = [4] * N
    idx = np.unravel_index(j, dims)
    return tensor(s[x] for x in idx)


def basis(x: Union[int, str], dim: int = 2) -> np.ndarray:
    '''
    Returns a single element of a state vector.  Each component is
    either a digit (0,1,2,...) representing the computational basis,
    or one or two letters representing a special state (H,V,D,A,R,L,
    X,Y,Z,Xm,Ym,Zm.  B1..4 are the 4 bell states
    '''
    d = {
        'H': [1, 0],
        'V': [0, 1],
        'D': [1, 1],
        'A': [1, -1],
        'R': [1, 1j],
        'L': [1, -1j],
        'X': [1, 1],
        'Xp': [1, 1],
        'Xm': [1, -1],
        'Y': [1, 1j],
        'Yp': [1, 1j],
        'Ym': [1, -1j],
        'Z': [1, 0],
        'Zp': [1, 0],
        'Zm': [0, 1],
        'B1': phiminus,
        'B2': phiplus,
        'B3': psiminus,
        'B4': psiplus
    }
    if isinstance(x, int) or x.isdigit():
        rv = np.zeros((dim), dtype=complex)
        rv[int(x)] = 1
    else:
        rv = np.array(d[x], dtype=complex)
        rv = rv / np.linalg.norm(rv)
    return rv


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute the commutator of two matrices"""
    return A @ B - B @ A


def dagger(vector: np.ndarray) -> np.ndarray:
    """Compute the hermitian conjugate of a vector or matrix"""
    return vector.conjugate().transpose()


def normalize(X: np.ndarray) -> np.ndarray:
    """Normalize the representation of X.  This means:
    For state vectors, X' * X = 1 and the first non-zero
    element of X is positive real.  For density matricies,
    X is hermetian, unit trace, and non-negative"""
    if X.ndim == 1:
        nz = X[np.nonzero(X)][0]
        X *= nz.conj()
        amp = np.abs(dagger(X) @ X)
        X /= np.sqrt(amp)
    else:
        (d, U) = eigh(X)
        d = d.real
        d[d < 0] = 0
        X = U @ np.diag(d) @ dagger(U)
        X = dagger(X) + X
        X /= np.trace(X)
    return X


def psi2rho(psi: np.ndarray) -> np.ndarray:
    '''
    Converts input state vector to density matrix represenation.  If psi
    is already a density matrix, leave it alone, so this function can be
    used to coerce input arguments to the desired type.
    '''
    if psi.ndim == 1:
        return normalize(np.outer(psi, psi.conj()))
    else:  # Already a density matrix
        return normalize(psi)


def rho2v(rho):
    """密度矩阵转相干矢
    """
    N = rho.shape[0]
    indices = np.tril_indices(N, -1)
    z = np.diag(rho).real
    Z = np.cumsum(z[:-1]) - z[1:] * np.arange(1, N)
    return np.hstack((rho[indices].real, rho[indices].imag, Z))


def v2rho(V):
    """相干矢转密度矩阵"""
    N = int(np.sqrt(len(V) + 1))
    assert N**2 - 1 == len(V)

    X = V[:(N**2 - N) // 2]
    Y = V[(N**2 - N) // 2:(N**2 - N)]
    Z = V[(N**2 - N):]

    A = np.hstack([Z[:0:-1] - Z[-2::-1], Z[0]])
    zn = (1 - A.sum()) / N
    A = A / np.arange(N - 1, 0, -1)
    z = np.cumsum(np.hstack([zn, A]))[::-1]

    rho = np.diag(z / 2).astype(complex)

    rho[np.tril_indices(N, -1)] = X + 1j * Y
    rho += dagger(rho)

    return normalize(rho)


def Hermitian2v(H):
    N = H.shape[0]
    indices = np.triu_indices(N, 1)
    return np.hstack((H[indices].real, H[indices].imag, np.diag(H).real))


def v2Hermitian(V):
    N = int(round(np.sqrt(len(V))))

    X = V[:(N**2 - N) // 2]
    Y = V[(N**2 - N) // 2:(N**2 - N)]
    Z = V[(N**2 - N):]
    H = np.diag(Z / 2).astype(np.complex)

    H[np.triu_indices(N, 1)] = X + 1j * Y
    H += dagger(H)
    return H


def unitary2v(U):
    H = -1j * logm(U)
    return Hermitian2v(H)


def v2unitary(V):
    H = v2Hermitian(V)
    return expm(1j * H)


def rho2bloch(rho):
    """
    与 rho2v 类似，只不过是选择以 sigma_i \otimes sigma_j ... 为基底来展开,
    只适用于N比特态,即 rho 的形状只能是 (2 ** N, 2 ** N)
    """
    bases = [s0, s1, s2, s3]
    N = int(np.log2(rho.shape[0]))
    return np.asarray([
        np.sum(rho * reduce(np.kron, s).T).real
        for s in product(bases, repeat=N)
    ][1:])


def bloch2rho(V):
    """
    rho2bloch 的逆函数
    """
    bases = [s0, s1, s2, s3]
    N = int(np.log2(len(V) + 1) / 2)
    rho = reduce(np.add,
                 (a * reduce(np.kron, s)
                  for (a, s) in zip(chain([1], V), product(bases, repeat=N))))
    return rho / rho.shape[0]


def traceDistance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Compute the trace distance between matrixes rho and sigma
    See Nielsen and Chuang, p. 403
    """
    A = rho - sigma
    return np.real(np.trace(sqrtm(dagger(A) @ A))) / 2.0


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    '''
    The fidelity of two quantum states.
    '''
    rho = psi2rho(rho)
    sigma = psi2rho(sigma)
    rhosqrt = sqrtm(rho)
    return np.real(np.trace(sqrtm(rhosqrt @ sigma @ rhosqrt)))


def entropy(rho: np.ndarray) -> float:
    '''von Neumann / Shannon entropy function'''
    if rho.ndim == 1:  # pure states have no entropy
        return 0
    v = np.linalg.eigvalsh(rho).real
    return -np.sum(v[v > 0] * np.log2(v[v > 0]))


def concurrence(rho: np.ndarray) -> float:
    """Concurrence of a two-qubit density matrix.
    see http://qwiki.stanford.edu/wiki/Entanglement_of_Formation
    """
    yy = np.array([[ 0, 0, 0,-1],
                   [ 0, 0, 1, 0],
                   [ 0, 1, 0, 0],
                   [-1, 0, 0, 0]], dtype=complex) #yapf: disable
    m = rho @ yy @ rho.conj() @ yy
    eigs = [np.abs(e) for e in np.linalg.eig(m)[0]]
    e = [np.sqrt(x) for x in sorted(eigs, reverse=True)]
    return max(0, e[0] - e[1] - e[2] - e[3])


def eof(rho: np.ndarray) -> float:
    """Entanglement of formation of a two-qubit density matrix.
    see http://qwiki.stanford.edu/wiki/Entanglement_of_Formation
    """
    def h(x):
        if x <= 0 or x >= 1:
            return 0
        return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

    C = concurrence(rho)
    arg = max(0, np.sqrt(1 - C**2))
    return h((1 + arg) / 2.0)


def randomUnitary(N: int) -> np.ndarray:
    """Random unitary matrix of dimension N."""
    H = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    H = (H + dagger(H)) / 2
    U = expm(-1j * H)
    return U


def randomState(N: int) -> np.ndarray:
    """Generates a random pure state."""
    psi = np.random.randn(N) + 1j * np.random.randn(N)
    return normalize(psi)


def randomDensity(N: int, rank: Optional[int] = None) -> np.ndarray:
    """Generates a random density matrix.  The distribution does not
    necessarily mean anything.  N is the dimension and rank is the
    number of non-zero eigenvalues."""

    if rank is None or rank > N:
        rank = N
    A = np.random.randn(N, rank) + 1j * np.random.randn(N, rank)
    rho = A @ A.T.conj()
    return rho / np.trace(rho)


##########################################################


def U(theta, phi, lambda_, delta=0):
    """general unitary
    
    Any general unitary could be implemented in 2 pi/2-pulses on hardware

    U(theta, phi, lambda_, delta) = \
        np.exp(1j * delta) * \
        U(0, 0, theta + phi + lambda_) @ \
        U(np.pi / 2, p2, -p2) @ \
        U(np.pi / 2, p1, -p1))

    or  = \
        np.exp(1j * delta) * \
        U(0, 0, theta + phi + lambda_) @ \
        rfUnitary(np.pi / 2, p2 + pi / 2) @ \
        rfUnitary(np.pi / 2, p1 + pi / 2)
    
    where p1 = -lambda_ - pi / 2
          p2 = pi / 2 - theta - lambda_
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    a, b = (phi + lambda_) / 2, (phi - lambda_) / 2
    d = np.exp(1j * delta)
    return d * np.array([[c * np.exp(-1j * a), -s * np.exp(-1j * b)],
                         [s * np.exp(1j * b), c * np.exp(1j * a)]])


def Unitary2Angles(U: np.ndarray) -> np.ndarray:
    if U[0, 0] == 0:
        delta = (np.angle(U[1, 0]) + np.angle(U[0, 1])) / 2
        U /= np.exp(1j * delta)
        theta = np.pi
        phi = np.angle(U[1, 0])
        lambda_ = -phi
    else:
        delta = np.angle(U[0, 0])
        U = U / np.exp(1j * delta)
        theta = 2 * np.arccos(U[0, 0])
        phi = np.angle(U[1, 0])
        lambda_ = np.angle(-U[0, 1])
        delta += (phi + lambda_) / 2
    return np.array([theta, phi, lambda_, delta]).real


def rfUnitary(theta, phi):
    """
    Gives the unitary operator for an ideal microwave gate.
    phi gives the rotation axis on the plane of the bloch sphere (RF drive phase)
    theta is the rotation angle of the gate (pulse area)

    rfUnitary(theta, phi) := expm(-1j * theta / 2 * \
        (sigmaX() * cos(phi) + sigmaY() * sin(phi)))

    rfUnitary(theta, phi + pi/2) == U(theta, phi, -phi)
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s * np.exp(-1j * phi)],
                     [-1j * s * np.exp(1j * phi), c]])


def fSim(theta, phi):
    c, s = np.cos(theta), np.sin(theta)
    p = np.exp(-1j * phi)
    return np.array([
        [1,     0,     0,     0],
        [0,     c, -1j*s,     0],
        [0, -1j*s,     c,     0],
        [0,     0,     0,     p]
    ]) #yapf: disable


def rabi(t, TR, Omega, A, offset):
    return oscillation(t, [(1, Omega)], A, offset) * decay(t, [TR])


def cpmg(t, T1, Tphi, Delta, A, offset, phi=0):
    return oscillation(t, [(np.exp(1j * phi), Delta)], A, offset) * decay(
        t, [2 * T1, Tphi])


def relaxation(t, T1, A, offset):
    return A * decay(t, [T1]) + offset
