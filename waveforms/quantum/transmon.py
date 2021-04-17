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
        if {"EJ", "Ec"} <= set(kw):
            self._set_params_EJ_Ec(kw['EJ'], kw['Ec'])
        elif {"f01", "alpha"} <= set(kw):
            if 'ng' not in kw:
                self._set_params_f01_alpha(kw['f01'], kw['alpha'])
            else:
                self._set_params_f01_alpha(kw['f01'], kw['alpha'], kw['ng'])
        else:
            raise TypeError(
                '_set_params() got an unexpected keyword arguments')

    def _set_params_EJ_Ec(self, EJ, Ec):
        self.Ec = Ec
        self.EJ = EJ

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

    def chargeParityDiff(self, flux=0, ng=0, k=0):
        a = self.levels(flux, ng=0 + ng)
        b = self.levels(flux, ng=0.5 + ng)

        return (a[1 + k] - a[k]) - (b[1 + k] - b[k])
        

if __name__ == "__main__":
    q = Transmon(f01=4.2, alpha=4.010-4.2)

    levels = q.levels()
    f01 = levels[1] - levels[0]
    f12 = levels[2] - levels[1]

    print("chargeParityDiff:")
    for k in range(4):
        diff = q.chargeParityDiff(k=k)
        print(f"  ({k},{k+1}) diff = {diff * 1e3:8.4f} MHz", f"(T = {1/np.abs(diff) / 2e3:.1f} us)")
    print(f"EJ = {q.EJ:.4f} GHz, Ec = {q.Ec*1e3:.4f} MHz, EJ/Ec={q.EJ/q.Ec:.2f}")
    print(f"f01 = {f01:.4f} GHz, alpha = {(f12-f01)*1e3:.1f} MHz")
