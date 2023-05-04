from itertools import permutations, product, chain
import numpy as np


def to_cycle_notation(a: tuple) -> tuple[tuple]:
    """
    Cauchy's two-line notation to cycle notation

    >>> to_cycle_notation((1, 2, 3, 0))
    ((0, 1, 2, 3),)
    >>> to_cycle_notation((3, 2, 1, 0))
    ((0, 3), (1, 2))
    >>> to_cycle_notation((1, 2, 3, 0, 5, 6, 7, 4))
    ((0, 1, 2, 3), (4, 5, 6, 7))
    """
    cycles = []
    rest = list(a)
    while len(rest) > 0:
        cycle = [rest.pop(0)]
        el = a[cycle[-1]]
        while cycle[0] != el:
            cycle.append(el)
            rest.remove(el)
            el = a[cycle[-1]]
        if len(cycle) > 1:
            i = cycle.index(min(cycle))
            cycles.append(tuple(cycle[i:] + cycle[:i]))
    return tuple(sorted(cycles))


def _from_cycle_notation(a: tuple[tuple], N: int | None = None) -> tuple:
    if N is None:
        N = max(max(c) for c in a) + 1
    ret = list(range(N))
    for cycle in a:
        for i in range(len(cycle)):
            ret[cycle[i]] = cycle[(i + 1) % len(cycle)]
    return tuple(ret)


def from_cycle_notation(a: tuple[tuple], N: int | None = None) -> tuple:
    """
    Cycle notation to Cauchy's two-line notation
    """
    ret = (0, )
    for x in a:
        ret = mul(_from_cycle_notation((x, )), ret)
    return ret


def to_matrix(a: tuple):
    ret = np.zeros((len(a), len(a)), dtype=int)
    for i in range(len(a)):
        ret[i, a[i]] = 1
    return ret


def inv_cycles(cycles: tuple[tuple]) -> tuple[tuple]:
    ret = []
    for cycle in reversed(cycles):
        cycle = tuple(reversed(cycle))
        i = cycle.index(min(cycle))
        ret.append(tuple(cycle[i:] + cycle[:i]))
    return tuple(ret)


def mul_cycles(cycles1: tuple[tuple], cycles2: tuple[tuple]) -> tuple[tuple]:
    ret = mul(from_cycle_notation(cycles1), from_cycle_notation(cycles2))
    return to_cycle_notation(ret)


def inv(a):
    return tuple(v for _, v in sorted([(v, i) for i, v in enumerate(a)]))


def mul(a, b):
    N = max(len(a), len(b))
    a = a + tuple(range(len(a), N))
    b = b + tuple(range(len(b), N))
    return tuple(a[b[i]] for i in range(N))


def generate(generators: list[tuple[tuple]]):
    gens = [()]
    elements = generators.copy()
    while True:
        new_elements = []
        for a, b in chain(product(gens, elements), product(elements, gens),
                          product(elements, elements)):
            c = mul_cycles(a, b)
            if c not in gens and c not in elements and c not in new_elements:
                new_elements.append(c)
        gens.extend(elements)
        if len(new_elements) == 0:
            break
        elements = new_elements
    return gens


class PermutationGroup():

    def __init__(self, generators: list[tuple[tuple]]):
        self.generators = generators
        self._gens = []

    @property
    def gens(self):
        if self._gens == []:
            self._gens = generate(self.generators)
        return self._gens

    def __len__(self):
        return len(self.gens)

    def __getitem__(self, i):
        return self.gens[i]


class SymmetricGroup(PermutationGroup):

    def __init__(self, N: int):
        if N < 2:
            super().__init__([])
        elif N == 2:
            super().__init__([((0, 1), )])
        else:
            super().__init__([((0, 1), ), (tuple(range(N)), )])
        self.N = N

    def __len__(self):
        return np.math.factorial(self.N)


class CyclicGroup(PermutationGroup):

    def __init__(self, N: int):
        if N < 2:
            super().__init__([])
        else:
            super().__init__([(tuple(range(N)), )])
        self.N = N

    def __len__(self):
        return self.N


class DihedralGroup(PermutationGroup):

    def __init__(self, N: int):
        if N < 2:
            generators = []
        elif N == 2:
            generators = [((0, 1), )]
        else:
            generators = [(tuple(range(N)), ),
                          tuple([(i, N - 1 - i) for i in range(N // 2)])]
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
                generators.append((tuple(range(start, start + ni)), ))
                start += ni
        super().__init__(generators)

    def __len__(self):
        return np.multiply.reduce(self.n)


class AlternatingGroup(PermutationGroup):

    def __init__(self, N: int):
        if N <= 2:
            generators = []
        elif N == 3:
            generators = [((0, 1, 2), )]
        else:
            generators = [((0, 1, 2), ), (tuple(range(N)), ) if N % 2 else
                          (tuple(range(1, N)), )]
        super().__init__(generators)
        self.N = N

    def __len__(self):
        return np.math.factorial(self.N) // 2
