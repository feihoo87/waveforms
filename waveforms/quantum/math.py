#%%
from functools import reduce
from itertools import chain, cycle, product
from typing import Generator, Optional, Sequence, Union

import numpy as np
from scipy.linalg import eigh, expm, sqrtm

# Paulis
sigmaI = lambda: np.eye(2, dtype=complex)
sigmaX = lambda: np.array([[0, 1], [1, 0]], dtype=complex)
sigmaY = lambda: np.array([[0, -1j], [1j, 0]], dtype=complex)
sigmaZ = lambda: np.array([[1, 0], [0, -1]], dtype=complex)
s0, s1, s2, s3 = sigmaI(), sigmaX(), sigmaY(), sigmaZ()

# Bell states
BellPhiP = lambda: np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
BellPhiM = lambda: np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
BellPsiP = lambda: np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
BellPsiM = lambda: np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
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
        theta, phi, lambda_, delta = np.pi, 0, np.angle(U[1, 0]), 0
    else:
        delta = np.angle(U[0, 0])
        U = U / np.exp(1j * delta)
        theta = 2 * np.arccos(U[0, 0])
        phi = np.angle(U[1, 0])
        lambda_ = np.angle(-U[0, 1])
        delta += (phi + lambda_) / 2
    return np.array([theta, phi, lambda_, delta])


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


def lorentz(x, x0, gamma):
    """lorentz peak"""
    return 1 / (1 + ((x - x0) / gamma)**2)


def lorentzComplex(x, x0, gamma):
    """complex lorentz peak
    
    lorentz(x, x0, gamma) = lorentzComplex(x, x0, gamma) * conj(lorentzComplex(x, x0, gamma))
    """
    return 1 / (1 + 1j * (x - x0) / gamma)


def gaussian(x, x0, sigma):
    """gaussian peak"""
    return np.exp(-0.5 * ((x - x0) / sigma)**2)


def peaks(x, peaks, background=0):
    """
    peaks: list of (center, width, amp, shape)
           shape should be either 'gaussian' or 'lorentz'
    background: a float, complex or ndarray with the same shape of `x`
    """
    ret = np.zeros_like(x)
    for center, width, amp, shape in peaks:
        if shape == 'gaussian':
            ret += amp * gaussian(x, center, width)
        else:
            ret += amp * lorentz(x, center, width)

    return ret + background


def complexPeaks(x, peaks, background=0):
    """
    peaks: list of (center, width, amp)
    background: a float, complex or ndarray with the same shape of `x`
    """
    ret = np.zeros_like(x, dtype=np.complex)
    for x0, gamma, A, *_ in peaks:
        ret += A * lorentzComplex(x, x0, gamma)
    return ret + background


def decay(t, tau):
    """
    exponential decay
    """
    a = -(1 / np.asarray(tau))**(np.arange(len(tau)) + 1)
    a = np.hstack([a[::-1], [0]])
    return np.exp(np.poly1d(a)(t))


def oscillation(t, spec=((1, 1), ), amplitude=1, offset=0):
    """
    oscillation
    """
    ret = np.zeros_like(t, dtype=np.complex)
    for A, f in spec:
        ret += A * np.exp(2j * np.pi * f * t)
    return amplitude * np.real(ret) + offset


def rabi(t, TR, Omega, A, offset):
    return oscillation(t, [(1, Omega)], A, offset) * decay(t, [TR])


def cpmg(t, T1, Tphi, Delta, A, offset, phi=0):
    return oscillation(t, [(np.exp(1j * phi), Delta)], A, offset) * decay(
        t, [2 * T1, Tphi])


def relaxation(t, T1, A, offset):
    return A * decay(t, [T1]) + offset


def linFit(x, y):
    """use less memory than np.polyfit"""
    x, y = np.asarray(x), np.asarray(y)
    xm, ym = x.mean(), y.mean()
    N = len(x)
    a = (np.sum(x * y) - N * xm * ym) / ((x**2).sum() - N * xm * xm)
    b = ym - a * xm
    return np.array([a, b])


def fitCircle(x, y):
    u, v = x - x.mean(), y - y.mean()
    Suuu, Svvv = np.sum(u**3), np.sum(v**3)
    Suu, Svv = np.sum(u**2), np.sum(v**2)
    Suv = np.sum(u * v)
    Suuv, Suvv = np.sum(u**2 * v), np.sum(u * v**2)
    uc = (Suuv * Suv - Suuu * Svv - Suvv * Svv + Suv * Svvv)
    vc = (-Suu * Suuv + Suuu * Suv + Suv * Suvv - Suu * Svvv)
    uc /= 2 * (Suv**2 - Suu * Svv)
    vc /= 2 * (Suv**2 - Suu * Svv)
    xc, yc = uc + x.mean(), vc + y.mean()
    R = np.sqrt(np.mean((x - xc)**2 + (y - yc)**2))
    return xc, yc, R


def fitCrossPoint(x1, y1, x2, y2):
    a1, b1 = linFit(x1, y1)
    a2, b2 = linFit(x2, y2)
    return (b2 - b1) / (a1 - a2), (a1 * b2 - a2 * b1) / (a1 - a2)


def fitPole(x, y):
    a, b, c = np.polyfit(x, y, 2)
    return -0.5 * b / a, c - 0.25 * b**2 / a


def getFTMatrix(f_list: Sequence[float],
                numOfPoints: int,
                phase_list: Optional[Sequence[float]] = None,
                weight: Optional[np.ndarray] = None,
                sampleRate: float = 1e9) -> np.ndarray:
    """
    get a matrix for Fourier transform

    Args:
        f_list (Sequence[float]): list of frequencies
        numOfPoints (int): size of signal frame
        phase_list (Optional[Sequence[float]], optional): list of phase. Defaults to None.
        weight (Optional[np.ndarray], optional): weight or list of weight. Defaults to None.
        sampleRate (float, optional): sample rate of signal. Defaults to 1e9.

    Returns:
        numpy.ndarray: exp matrix
    
    >>> shots, numOfPoints, sampleRate = 100, 1000, 1e9
    >>> f1, f2 = -12.7e6, 32.8e6
    >>> signal = np.random.randn(shots, numOfPoints)
    >>> e = getFTMatrix([f1, f2], numOfPoints, sampleRate=sampleRate)
    >>> ret = signal @ e
    >>> ret.shape
    (100, 2)
    >>> t = np.arange(numOfPoints) / sampleRate
    >>> signal = 0.8 * np.sin(2 * np.pi * f1 * t) + 0.2 * np.cos(2 * np.pi * f2 * t)
    >>> signal @ e
    array([-0.00766509-0.79518987j,  0.19531432+0.00207068j])
    >>> spec = 2 * np.fft.fft(signal) / numOfPoints
    >>> freq = np.fft.fftfreq(numOfPoints)
    >>> e = getFTMatrix(freq, numOfPoints, sampleRate=1)
    >>> np.allclose(spec, signal @ e)
    True
    """
    e = []
    t = np.linspace(0, numOfPoints / sampleRate, numOfPoints, endpoint=False)
    if weight is None or len(weight) == 0:
        weight = np.full(numOfPoints, 2 / numOfPoints)
    if phase_list is None or len(phase_list) == 0:
        phase_list = np.zeros_like(f_list)
    if weight.ndim == 1:
        weightList = cycle(weight)
    else:
        weightList = weight
    for f, phase, weight in zip(f_list, phase_list, weightList):
        e.append(weight * np.exp(-1j * (2 * np.pi * f * t + phase)))
    return np.asarray(e).T


# %%
