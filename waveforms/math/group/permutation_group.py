from __future__ import annotations

from itertools import chain, product
import functools
import numpy as np


@functools.total_ordering
class Permutation():

    __slots__ = ('_perm', '_max')

    def __init__(self, *perm):
        self._perm = tuple(perm)
        self._max = max(perm)
        assert len(set(perm)) == self._max + 1

    def __hash__(self):
        return hash(self._perm)

    def __eq__(self, value: Permutation) -> bool:
        return self._perm == value._perm

    def __lt__(self, value: Permutation) -> bool:
        return self._perm < value._perm

    def __getitem__(self, i: int) -> int:
        try:
            return self._perm[i]
        except:
            return i

    def __repr__(self):
        return f'Permutation{tuple(self._perm)!r}'

    def __mul__(self,
                other: Permutation | Cycles | tuple | list) -> Permutation:
        """Returns the product of two permutation.

        The product of permutations a, b is understood to be the permutation
        resulting from applying a, then b.
        """
        if isinstance(other, Permutation):
            pass
        elif isinstance(other, Cycles):
            other = other.to_permutation()
        else:
            other = Permutation(*other)
        N = max(self._max, other._max) + 1
        return Permutation(*[other[self[i]] for i in range(N)])

    def __pow__(self, n: int) -> Permutation:
        if n == 0:
            return Permutation(0)
        elif n > 0:
            n = n % self.order
            ret = Permutation(0)
            while n > 0:
                if n % 2 == 1:
                    ret *= self
                self *= self
                n //= 2
            return ret
        else:
            return self.inv()**(-n)

    def inv(self) -> Permutation:
        return Permutation(v for _, v in sorted([(v, i)
                                                 for i, v in enumerate(self)]))

    @functools.cached_property
    def order(self):
        return self.to_cycles().order

    def to_cycles(self) -> Cycles:
        cycles = []
        rest = list(self._perm)
        while len(rest) > 0:
            cycle = [rest.pop(0)]
            el = self._perm[cycle[-1]]
            while cycle[0] != el:
                cycle.append(el)
                rest.remove(el)
                el = self._perm[cycle[-1]]
            if len(cycle) > 1:
                cycles.append(tuple(cycle))
        return Cycles(*cycles)

    def to_matrix(self) -> np.ndarray:
        N = self._max + 1
        ret = np.zeros((N, N), dtype=np.int8)
        for i in range(N):
            ret[i, self[i]] = 1
        return ret

    def replace(self, expr) -> Permutation:
        """replaces each part in expr by its image under the permutation."""
        if isinstance(expr, (tuple, list)):
            return type(expr)(self.replace(e) for e in expr)
        else:
            return self[expr]


@functools.total_ordering
class Cycles():

    __slots__ = ('_cycles', '_support', '_max', '_min')

    def __init__(self, *cycles, _simplify: bool = True):
        self._support = set()
        if _simplify:
            ret = []
            for cycle in cycles:
                self._support.update(cycle)
                i = cycle.index(min(cycle))
                cycle = cycle[i:] + cycle[:i]
                ret.append(tuple(cycle))
            self._cycles = tuple(sorted(ret))
        else:
            self._cycles = tuple(cycles)
        self._support = tuple(sorted(self._support))
        if self._support:
            self._max = max(self._support)
            self._min = min(self._support)
        else:
            self._max = 0
            self._min = 0

    def __hash__(self):
        return hash(self._cycles)

    def __eq__(self, value: Cycles) -> bool:
        return self._cycles == value._cycles

    def __lt__(self, value: Cycles) -> bool:
        return self._cycles < value._cycles

    def __mul__(self, other: Cycles | Permutation) -> Cycles:
        """Returns the product of two cycles.

        The product of permutations a, b is understood to be the permutation
        resulting from applying a, then b.
        """
        return (self.to_permutation() * other).to_cycles()

    def __pow__(self, n: int) -> Cycles:
        if n == 0:
            return Cycles()
        elif n > 0:
            n = n % self.order
            ret = Cycles()
            while n > 0:
                if n % 2 == 1:
                    ret *= self
                self *= self
                n //= 2
            return ret
        else:
            return self.inv()**(-n)

    def inv(self):
        c = Cycles(*[(cycle[0], ) + tuple(reversed(cycle[1:]))
                     for cycle in self],
                   _simplify=False)
        c._max = self._max
        c._min = self._min
        c._support = self._support
        return c

    @functools.cached_property
    def order(self):
        return np.lcm.reduce([len(cycle) for cycle in self._cycles])

    @property
    def support(self):
        return self._support

    @property
    def max(self):
        return self._max

    @property
    def min(self):
        return self._min

    def __len__(self):
        return len(self._support)

    def __repr__(self):
        return f'Cycles{tuple(self._cycles)!r}'

    def to_permutation(self) -> Permutation:
        if len(self._support) == 0:
            return Permutation(0)
        ret = list(range(self._max + 1))
        for cycle in self._cycles:
            for i in range(len(cycle)):
                ret[cycle[i]] = cycle[(i + 1) % len(cycle)]
        return Permutation(*ret)

    def to_matrix(self) -> np.ndarray:
        return self.to_permutation().to_matrix()


class PermutationGroup():

    def __init__(self, generators: list[Cycles]):
        self.generators = generators
        self._elements = []

    @staticmethod
    def _generate(generators: list[Cycles]):
        gens = [Cycles()]
        elements = generators.copy()
        while True:
            new_elements = []
            for a, b in chain(product(gens, elements), product(elements, gens),
                              product(elements, elements)):
                c = a * b
                if c not in gens and c not in elements and c not in new_elements:
                    new_elements.append(c)
            gens.extend(elements)
            if len(new_elements) == 0:
                break
            elements = new_elements
        return sorted(gens)

    @property
    def elements(self):
        if self._elements == []:
            self._elements = self._generate(self.generators)
        return self._elements

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, i):
        return self.elements[i]


class SymmetricGroup(PermutationGroup):

    def __init__(self, N: int):
        if N < 2:
            super().__init__([])
        elif N == 2:
            super().__init__([Cycles((0, 1))])
        else:
            super().__init__([Cycles((0, 1)), Cycles(tuple(range(N)))])
        self.N = N

    def __len__(self):
        return np.math.factorial(self.N)


class CyclicGroup(PermutationGroup):

    def __init__(self, N: int):
        if N < 2:
            super().__init__([])
        else:
            super().__init__([Cycles(tuple(range(N)))])
        self.N = N

    def __len__(self):
        return self.N


class DihedralGroup(PermutationGroup):

    def __init__(self, N: int):
        if N < 2:
            generators = []
        elif N == 2:
            generators = [Cycles((0, 1))]
        else:
            generators = [
                Cycles(tuple(range(N))),
                Cycles(*[(i + N % 2, N - 1 - i) for i in range(N // 2)])
            ]
        super().__init__(generators)
        self.N = N

    def __len__(self):
        return 2 * self.N


class AbelianGroup(PermutationGroup):

    def __init__(self, *n: int):
        self.n = tuple(sorted(n))
        generators = []
        start = 0
        for ni in self.n:
            if ni >= 2:
                generators.append(Cycles(tuple(range(start, start + ni))))
                start += ni
        super().__init__(generators)

    def __len__(self):
        return np.multiply.reduce(self.n)


class AlternatingGroup(PermutationGroup):

    def __init__(self, N: int):
        if N <= 2:
            generators = []
        elif N == 3:
            generators = [Cycles((0, 1, 2))]
        else:
            generators = [
                Cycles((0, 1, 2)),
                Cycles(tuple(range(N))) if N %
                2 else Cycles(tuple(range(1, N)))
            ]
        super().__init__(generators)
        self.N = N

    def __len__(self):
        return np.math.factorial(self.N) // 2
