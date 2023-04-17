from functools import lru_cache

import numpy as np
from scipy.linalg import eigh_tridiagonal, eigvalsh_tridiagonal
from scipy.optimize import minimize


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
