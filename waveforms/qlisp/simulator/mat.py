import numpy as np
import scipy.sparse as sp


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
