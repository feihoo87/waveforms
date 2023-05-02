from functools import lru_cache, reduce

import numpy as np
from scipy.linalg import eigh_tridiagonal, eigvalsh_tridiagonal
from scipy.optimize import minimize

CAP_UNIT = 1e-15
FREQ_UNIT = 1e9
"""
常见超导材料及其对应的超导能隙：

| 超导材料 | 超导能隙 (Δ, $\mu$eV)|
|---------|--------------------|
| 铝 (Al) |         170        |
| 钽 (Ta) |         700-1,000  |
| 铌 (Nb) |         1500       |
| 汞 (Hg) |         16         |
| 铍 (Be) |         300-450    |
| 镍 (Ni) |         200        |
| 锡 (Sn) |         1100       |
| 铅 (Pb) |         1400       |
| 镧钡铜氧化物 (LBCO) |  20,000-30,000   |
| 高温钡镧铜氧化物 (BSCCO) | 10,000-15,000 |

请注意，这些值可能因材料的纯度、制备条件等因素而有所变化。
"""


class Transmon():

    def __init__(self, **kw):
        self.Ec = 0.2
        self.EJ = 20

        self.d = 0
        if kw:
            self._set_params(**kw)

    def _set_params(self, **kw):
        if {"EJ", "Ec", "d"} <= set(kw):
            return self._set_params_EJS_Ec_d(kw['EJ'], kw['Ec'], kw['d'])
        elif {"EJ", "Ec"} <= set(kw):
            return self._set_params_EJ_Ec(kw['EJ'], kw['Ec'])
        elif {"f01", "alpha"} <= set(kw):
            if 'ng' not in kw:
                return self._set_params_f01_alpha(kw['f01'], kw['alpha'])
            else:
                return self._set_params_f01_alpha(kw['f01'], kw['alpha'],
                                                  kw['ng'])
        elif {"f01_max", "f01_min"} <= set(kw):
            if {"alpha1", "alpha2"} <= set(kw):
                return self._set_params_f01_max_min_alpha(
                    kw['f01_max'], kw['f01_min'], kw['alpha1'], kw['alpha2'],
                    kw.get('ng', 0))
            elif {"alpha"} <= set(kw):
                return self._set_params_f01_max_min_alpha(
                    kw['f01_max'], kw['f01_min'], kw['alpha'], kw['alpha'],
                    kw.get('ng', 0))
            elif {"alpha1"} <= set(kw):
                return self._set_params_f01_max_min_alpha(
                    kw['f01_max'], kw['f01_min'], kw['alpha1'], kw['alpha1'],
                    kw.get('ng', 0))
        raise TypeError('_set_params() got an unexpected keyword arguments')

    def _set_params_EJ_Ec(self, EJ, Ec):
        self.Ec = Ec
        self.EJ = EJ

    def _set_params_EJS_Ec_d(self, EJS, Ec, d):
        self.Ec = Ec
        self.EJ = EJS
        self.d = d

    def _set_params_f01_alpha(self, f01, alpha, ng=0):
        Ec = -alpha
        EJ = (f01 - alpha)**2 / 8 / Ec

        def err(x, target=(f01, alpha)):
            EJ, Ec = x
            levels = self._levels(Ec, EJ, ng=ng)
            f01 = levels[1] - levels[0]
            f12 = levels[2] - levels[1]
            alpha = f12 - f01
            return (target[0] - f01)**2 + (target[1] - alpha)**2

        ret = minimize(err, x0=[EJ, Ec])
        self._set_params_EJ_Ec(*ret.x)

    def _set_params_f01_max_min_alpha(self,
                                      f01_max,
                                      f01_min,
                                      alpha1,
                                      alpha2=None,
                                      ng=0):
        if alpha2 is None:
            alpha2 = alpha1

        Ec = -alpha1
        EJS = (f01_max - alpha1)**2 / 8 / Ec
        d = (f01_min + Ec)**2 / (8 * EJS * Ec)

        def err(x, target=(f01_max, alpha1, f01_min, alpha2)):
            EJS, Ec, d = x
            levels = self._levels(Ec, self._flux_to_EJ(0, EJS, d), ng=ng)
            f01_max = levels[1] - levels[0]
            f12 = levels[2] - levels[1]
            alpha1 = f12 - f01_max

            levels = self._levels(Ec, self._flux_to_EJ(0.5, EJS, d), ng=ng)
            f01_min = levels[1] - levels[0]
            f12 = levels[2] - levels[1]
            alpha2 = f12 - f01_min

            return (target[0] - f01_max)**2 + (target[1] - alpha1)**2 + (
                target[2] - f01_min)**2 + (target[3] - alpha2)**2

        ret = minimize(err, x0=[EJS, Ec, d])
        self._set_params_EJS_Ec_d(*ret.x)

    @staticmethod
    def _flux_to_EJ(flux, EJS, d=0):
        F = np.pi * flux
        EJ = EJS * np.sqrt(np.cos(F)**2 + d**2 * np.sin(F)**2)
        return EJ

    @staticmethod
    def _levels(Ec, EJ, ng=0.0, gridSize=51, select_range=(0, 10)):
        n = np.arange(gridSize) - gridSize // 2
        w = eigvalsh_tridiagonal(4 * Ec * (n - ng)**2,
                                 -EJ / 2 * np.ones(gridSize - 1),
                                 select='i',
                                 select_range=select_range)
        return w

    @lru_cache(maxsize=128)
    def levels(self, flux=0, ng=0):
        return self._levels(self.Ec, self._flux_to_EJ(flux, self.EJ, self.d),
                            ng)

    @property
    def EJ1_EJ2(self):
        return (1 + self.d) / (1 - self.d)

    def chargeParityDiff(self, flux=0, ng=0, k=0):
        a = self.levels(flux, ng=0 + ng)
        b = self.levels(flux, ng=0.5 + ng)

        return (a[1 + k] - a[k]) - (b[1 + k] - b[k])


def mass(C):
    """
    C: capacitance matrix in fF

    return: mass matrix in GHz^-1
    """
    from scipy.constants import e, h

    a = np.diag(C)
    b = C - np.diag(a)
    c = np.sum(b, axis=0)
    C = np.diag(a + c) - b

    # convert unit of capacitance, make sure energy unit is GHz
    M = C * CAP_UNIT / (4 * e**2 / h / FREQ_UNIT)
    return M


def Rn_to_EJ(Rn, gap=200e-6, T=0.01):
    """
    Rn: normal resistance in Ohm
    gap: superconducting gap in ueV
    T: temperature in K

    return: EJ in GHz
    """
    from scipy.constants import e, h, hbar, k, pi

    Delta = gap * e
    Ic = pi * Delta / (2 * e * Rn) * np.tanh(Delta / (2 * k * T))
    EJ = Ic * hbar / (2 * e)
    return EJ / h / 1e9


def flux_to_EJ(flux, EJS, d=0):
    """
    flux: flux in Phi_0
    EJS: symmetric Josephson energy in GHz
    d: asymmetry parameter
        EJ1 / EJ2 = (1 + d) / (1 - d)
    """
    F = np.pi * flux
    EJ = EJS * np.sqrt(np.cos(F)**2 + d**2 * np.sin(F)**2)
    return EJ


def n_op(N=5):
    return np.diag(np.arange(-N, N + 1))


# def cos_phi_op(N=5):
#     from scipy.sparse import diags
#     return diags(
#         [np.full((2 * N, ), 0.5),
#          np.full(
#              (2 * N, ), 0.5), [0.5], [0.5]], [1, -1, 2 * N, -2 * N]).toarray()


def cos_phi_op(N=5):
    from scipy.sparse import diags
    return diags([np.full(
        (2 * N, ), 0.5), np.full((2 * N, ), 0.5)], [1, -1]).toarray()


# def sin_phi_op(N=5):
#     from scipy.sparse import diags
#     return diags(
#         [np.full((2 * N, ), 0.5j),
#          np.full((2 * N, ), -0.5j), [-0.5j], [0.5j]],
#         [1, -1, 2 * N, -2 * N]).toarray()


def sin_phi_op(N=5):
    from scipy.sparse import diags
    return diags([np.full(
        (2 * N, ), 0.5j), np.full((2 * N, ), -0.5j)], [1, -1]).toarray()


def phi_op(N=5):
    from scipy.fft import fft, ifft, ifftshift

    k = ifftshift(np.arange(-N, N + 1) * np.pi / N)
    psi = np.eye(k.shape[0])
    T = fft(psi, overwrite_x=True)
    T *= k
    return ifft(T, overwrite_x=True)


def H_C(C, N=5):
    num_qubits = C.shape[0]

    A = np.linalg.inv(mass(C))

    n = n_op(N)
    I = np.eye(n.shape[0])

    n_ops = []
    for i in range(num_qubits):
        n_ops.append(
            reduce(np.kron, [n if j == i else I for j in range(num_qubits)]))

    ret = np.zeros_like(n_ops[0], dtype=float)

    for i in range(num_qubits):
        for j in range(num_qubits):
            ret += n_ops[i] * A[i, j] / 2 * n_ops[j]
    return ret


def H_phi(Rn, flux, d=0, N=5):
    num_qubits = Rn.shape[0]
    EJ = flux_to_EJ(flux, Rn_to_EJ(Rn), d)
    op = cos_phi_op(N)
    I = np.eye(op.shape[0])

    ret = np.zeros((op.shape[0]**num_qubits, op.shape[0]**num_qubits),
                   dtype=float)
    for i in range(num_qubits):
        ret -= EJ[i] * reduce(np.kron,
                              [op if j == i else I for j in range(num_qubits)])

    return ret


def _eig_singal_qubit(M, EJ, ng=0.0, levels=5, eps=1e-6):
    E = None

    for N in range(levels, 100):
        n = n_op(N)
        cos_phi = cos_phi_op(N)
        H = 0.5 / M * (n - ng)**2 - EJ * cos_phi
        w, v = np.linalg.eigh(H)

        if E is not None:
            if np.all(np.abs(E[:levels] - w[:levels]) < eps):
                break
        E = w

    return w, v.T @ n @ v, v, N


def spectrum(C, Rn, flux, ng=0.0, d=0, N=5):
    if isinstance(C, (int, float)):
        C = np.array([[C]])
    num_qubits = C.shape[0]
    if isinstance(Rn, (int, float)):
        Rn = Rn * np.ones(num_qubits)

    A = np.linalg.inv(mass(C))
    M = 1 / np.diag(A)
    EJ = flux_to_EJ(flux, Rn_to_EJ(Rn), d)

    H0 = []
    n_ops = []
    Udgs = []

    for m, Ej in zip(M, EJ):
        E, n, Udg, Ns = _eig_singal_qubit(m, Ej, ng=ng, levels=N)
        H0.append(E[:2 * Ns + 1])
        n_ops.append(n[:2 * Ns + 1, :2 * Ns + 1])
        Udgs.append(Udg[:2 * Ns + 1, :2 * Ns + 1])

    tenser_n_ops = []
    for i in range(num_qubits):
        tenser_n_ops.append(
            reduce(np.kron, [
                n_ops[i] if j == i else np.eye(n_ops[j].shape[0])
                for j in range(num_qubits)
            ]))

    H = np.zeros((N**num_qubits, N * num_qubits), dtype=float)

    H0 = np.diag(reduce(np.kron, [H0[i] for i in range(num_qubits)]))
    H = np.zeros_like(H0)
    H += H0

    for i in range(num_qubits):
        for j in range(i):
            H += A[i, j] * tenser_n_ops[i] * tenser_n_ops[j]
    w = np.linalg.eigvalsh(H)
    return w
