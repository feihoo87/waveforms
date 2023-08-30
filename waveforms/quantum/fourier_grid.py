# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:41:35 2016

@author: feihoo87
"""
__all__ = [
    'FourierGrid',
    'QuantumSystem',
    'Noise',
]

import numpy as np
from scipy import constants as const
from scipy.fftpack import fftn, ifftn, ifftshift, fftshift, ifft
from scipy.linalg import eigh
from scipy.integrate import ode
from scipy.interpolate import interp1d


class FourierGrid():
    """FourierGrid

    Write Hamiltonian in the form of H = T(k1, k2, ...) + U(x1, x2, ...) where
    [xj, kj] = i , and apply the function `T' to the Fourier grid object.
    Makesure T is time independent.
    """

    def __init__(self,
                 dims=1,
                 size=1001,
                 bound=[-5, 5],
                 T=lambda k, *b: 0.5 * k**2):
        """
        example:
        grid1d = FourierGrid(dims=1, size=101, bound=[-1,1])
        grid2d = FourierGrid(dims=2, size=[101, 41], bound=[[-3,3],[-1,1]])
        """
        self._check_size_and_bound(dims, size, bound)
        self.dims = dims
        self.size = size
        self.bound = bound
        self.T_func = [T]
        self.__T = None
        self.__T_diag = None
        self.__H = None
        self.__H_distroyed = False
        self.__diag_mask = None
        self._init_step()

    def _check_size_and_bound(self, dims, size, bound):
        if dims == 1:
            if not isinstance(size, int):
                raise Exception('size must be an integer')
            if len(bound) != 2:
                raise Exception('len(bound)!=2')
        else:
            if len(size) != dims:
                raise Exception('len(size) != dims')
            if len(bound) != dims:
                raise Exception('len(bound) != dims')
            for b in bound:
                if len(b) != 2:
                    raise Exception(
                        'all length of elements in bound must be 2')

    def _init_step(self):
        if self.dims == 1:
            self.x_step = [(self.bound[1] - self.bound[0]) / self.size]
            self.k_step = [2 * np.pi / (self.x_step[0] * self.size)]
        else:
            self.x_step = [(self.bound[i][1] - self.bound[i][0]) / self.size[i]
                           for i in range(self.dims)]
            self.k_step = [
                2 * np.pi / (self.x_step[i] * self.size[i])
                for i in range(self.dims)
            ]

    def x(self, indexing='xy', no_grid=False, sparse=True):
        size = [self.size] if self.dims == 1 else self.size
        bound = [self.bound] if self.dims == 1 else self.bound
        x = [
            np.linspace(bound[i][0], bound[i][1], size[i])
            for i in range(self.dims)
        ]
        if no_grid:
            return x
        else:
            return np.meshgrid(*x, indexing=indexing, sparse=sparse)

    def k(self, indexing='xy', no_grid=False, fft_shift=False, sparse=True):
        size = [self.size] if self.dims == 1 else self.size
        k = [
            np.linspace(-(size[i] - 1) / 2,
                        (size[i] - 1) / 2, size[i]) * self.k_step[i]
            for i in range(self.dims)
        ]
        if fft_shift:
            for i, l in enumerate(k):
                k[i] = ifftshift(l)
        if no_grid:
            return k
        else:
            return np.meshgrid(*k, indexing=indexing, sparse=sparse)

    def reset(self):
        del self.__T
        del self.__H
        del self.__T_diag
        del self.__diag_mask
        self.__T = None
        self.__H = None
        self.__T_diag = None
        self.__diag_mask = None

    def T(self):
        if self.__T is None:
            self._make_T()
        return self.__T

    def _make_T(self):
        T_k = self.T_func[0](*self.k(indexing='ij', fft_shift=True))
        hilbert_dims = np.multiply.reduce(self.size)
        shape = [self.size, self.size
                 ] if self.dims == 1 else list(self.size) * 2
        psi = np.eye(hilbert_dims).reshape(*shape)
        axes = tuple([i - self.dims for i in range(self.dims)])
        T = fftn(psi, axes=axes, overwrite_x=True)
        T *= T_k
        T = ifftn(T, axes=axes, overwrite_x=True)
        self.__T = T.reshape(hilbert_dims, hilbert_dims)
        self.__T_diag = np.diag(self.__T)
        self.__diag_mask = np.eye(len(self.__T_diag), dtype=bool)

    def U(self,
          U_func,
          clip=True,
          vmin=None,
          vmax=None,
          indexing='xy',
          time=None):
        """calculate potential energy by given function U_func at grid.x()

        indexing: 'xy' is used for plotting, 'ij' for generate Hamiltonian.
        if clip = True, the values of potential will clip into range [vmin, vmax].
        if vim / vmax is not given, min(U_func(x)) / min(U_func(at bounder of grid))
        will be used.
        """
        if time is not None:
            U = U_func(*self.x(indexing=indexing), **dict(time=time))
        else:
            U = U_func(*self.x(indexing=indexing))
        if not clip:
            return U
        if vmin is None:
            vmin = U.min()
        if vmax is None:
            s = slice(-1)
            ret = []
            for i in range(self.dims):
                index = [s] * self.dims
                index[i] = 0
                ret.append(np.min(U[index]))
                index[i] = -1
                ret.append(np.min(U[index]))
            vmax = np.min(ret)
        return U.clip(vmin, vmax)

    def _H(self):
        return self.__H

    def H(self, U_func, time_dependent=False, time=None):
        """Hamiltonian
        """
        U = self.U(U_func,
                   indexing='ij',
                   clip=False,
                   time=(time if time_dependent else None)).flatten()
        if self.__H is None or self.__H_distroyed:
            self.__H = np.diag(U).astype(complex)
            self.__H += self.T()
            self.__H_distroyed = False
        elif time_dependent:
            self.__H[self.__diag_mask] = 0
            self.__H += np.diag(U + self.__T_diag)
        else:
            pass
        return self.__H

    def E(self, U):
        """Return the eigenvalues"""
        #print(H.shape)
        #return np.linalg.eigvalsh(self.H(U))
        """
        MKL Error:
        return eigvalsh(H), overwrite_a=True, overwrite_b=True, check_finite=False)

        return eigh(H, b=None, lower=True, eigvals_only=True,
                overwrite_a=True, overwrite_b=True,
                turbo=True, eigvals=None, type=1,
                check_finite=False)
        """
        w, v = eigh(self.H(U),
                    b=None,
                    lower=True,
                    eigvals_only=False,
                    overwrite_a=True,
                    overwrite_b=True,
                    turbo=True,
                    eigvals=None,
                    type=1,
                    check_finite=False)
        self.__H_distroyed = True
        return w

    def States(self, U):
        """Return the eigenvalues and eigenvectors

        Returns :
        w : (..., M) ndarray
            The eigenvalues in ascending order, each repeated according to its
            multiplicity.
        v : {(..., M, M) ndarray, (..., M, M) matrix}
            The column v[:, i] is the normalized eigenvector corresponding to
            the eigenvalue w[i]. Will return a matrix object if a is a matrix
            object.
        """
        w, v = eigh(self.H(U),
                    overwrite_a=True,
                    overwrite_b=True,
                    check_finite=False)
        self.__H_distroyed = True
        return w, v


class QuantumSystem():

    def __init__(self, dim=1):
        self.dim = dim
        self.grid = None
        self.energy_unit = 1.0
        self.time_unit = 1.0

    def set_grid(self, bound=[-1, 1], size=501, step=None):
        self.grid = FourierGrid(dims=self.dim,
                                size=size,
                                bound=bound,
                                T=self.T)

    def T(self, *k):
        return 0.5 * np.add.reduce(list(map(lambda k: k**2, k)))

    def U(self, *x):
        return 0.0 * x[0]

    def States(self):
        return self.grid.States(self.U)

    def Dynamic(self, psi0, tlist, time_dependent=False, with_U=False):
        """psi(t[i]) = ret[i,:]"""

        def func(t, psi):
            """
            dpsi/dt = -1j*H.dot(psi.T)/const.hbar
            """
            H = self.grid.H(self.U, time_dependent, t)
            dpsi = H.dot(psi.T)
            dpsi *= (-1j * self.energy_unit * self.time_unit / const.hbar)
            return dpsi

        sol = ode(func).set_integrator('zvode').set_initial_value(
            psi0, tlist[0])
        #ret = [sol.integrate(t) for t in tlist[1:]]
        #ret.insert(0, psi0)
        ret = [psi0]
        for t in tlist[1:]:
            if not sol.successful():
                break
            ret.append(sol.integrate(t))
        if with_U:
            U_list = [
                self.U(*self.grid.x(), **dict(time=t))
                for t in tlist[:len(ret)]
            ]
            return np.asarray(ret), np.asarray(U_list)
        else:
            return np.asarray(ret)


class Noise():

    def __init__(self, max_time, max_freq, amp=1e-4, T=0.01, seed=None):
        """Generate noise by given spectral density, default spectral density is
        black body radiation.

        min_freq depends on max_time
        """
        if seed is not None:
            np.random.seed(seed)
        N = 2 * int(max_time * max_freq)
        self.max_time = max_time
        self.max_freq = max_freq

        def planck(f, T):
            return (2 * const.h * f**3 /
                    const.c**2) / (np.exp(const.h * f / const.k / T) - 1)

        f = np.linspace(-max_freq, max_freq, N)
        phase = np.random.random(N)
        spec = planck(f, T) * np.exp(2j * phase * np.pi)
        self._seq = np.real(ifft(fftshift(spec)))
        self._seq -= self._seq.mean()
        self._seq *= np.sqrt(2) * amp / np.sum(np.abs(self._seq))
        self._seq = np.asarray(list(self._seq) + [self._seq[0]])
        self._tlist = np.linspace(0, max_time, N + 1)
        self.interp = [interp1d(self._tlist, self._seq, kind='cubic')]

    def value(self, t):
        t = t % self.max_time
        return self.interp[0](t)
