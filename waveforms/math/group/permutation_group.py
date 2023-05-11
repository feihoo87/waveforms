from __future__ import annotations

import functools
import logging
import operator
import weakref
from itertools import chain, product

import numpy as np

logger = logging.getLogger(__name__)


class _NotContained(Exception):
    pass


@functools.total_ordering
class Cycles():

    __slots__ = ('_cycles', '_support', '_mapping', '_expr')

    def __init__(self, *cycles):
        self._mapping = {}
        self._expr: list[Cycles] = []
        if len(cycles) == 0:
            self._cycles = ()
            self._support = ()
            return

        support = set()
        ret = []
        for cycle in cycles:
            if len(cycle) <= 1:
                continue
            support.update(cycle)
            i = cycle.index(min(cycle))
            cycle = cycle[i:] + cycle[:i]
            ret.append(tuple(cycle))
            for i in range(len(cycle) - 1):
                self._mapping[cycle[i]] = cycle[i + 1]
            self._mapping[cycle[-1]] = cycle[0]
        self._cycles = tuple(sorted(ret))
        self._support = tuple(sorted(support))

    def __hash__(self):
        return hash(self._cycles)

    def is_identity(self):
        return len(self._cycles) == 0

    def __eq__(self, value: Cycles) -> bool:
        return self._cycles == value._cycles

    def __lt__(self, value: Cycles) -> bool:
        return self._cycles < value._cycles

    def __mul__(self, other: Cycles) -> Cycles:
        """Returns the product of two cycles.

        The product of permutations a, b is understood to be the permutation
        resulting from applying a, then b.
        """
        support = sorted(set(self.support + other.support), reverse=True)
        mapping = {
            a: b
            for a, b in zip(support, other.replace(self.replace(support)))
            if a != b
        }
        return Cycles._from_sorted_mapping(mapping)

    @staticmethod
    def _from_sorted_mapping(mapping: dict[int, int]) -> Cycles:
        c = Cycles()
        if not mapping:
            return c

        c._support = tuple(reversed(mapping.keys()))
        c._mapping = mapping.copy()

        cycles = []
        while len(mapping) > 0:
            k, el = mapping.popitem()
            cycle = [k]
            while k != el:
                cycle.append(el)
                el = mapping.pop(el)
            cycles.append(tuple(cycle))
        c._cycles = tuple(cycles)

        return c

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
        c = Cycles()
        if len(self._cycles) == 0:
            return c
        c._cycles = tuple([(cycle[0], ) + tuple(reversed(cycle[1:]))
                           for cycle in self._cycles])
        c._support = self._support
        c._mapping = {v: k for k, v in self._mapping.items()}
        return c

    @property
    def order(self):
        """Returns the order of the permutation.

        The order of a permutation is the least integer n such that
        p**n = e, where e is the identity permutation.
        """
        return np.lcm.reduce([len(cycle) for cycle in self._cycles])

    @property
    def support(self):
        """Returns the support of the permutation.

        The support of a permutation is the set of elements that are moved by
        the permutation.
        """
        return self._support

    @property
    def signature(self):
        """Returns the signature of the permutation.

        The signature of the permutation is (-1)^n, where n is the number of
        transpositions of pairs of elements that must be composed to build up
        the permutation. 
        """
        return 1 - 2 * ((len(self._support) - len(self._cycles)) % 2)

    def __len__(self):
        return len(self._support)

    def __repr__(self):
        return f'Cycles{tuple(self._cycles)!r}'

    def to_matrix(self) -> np.ndarray:
        """Returns the matrix representation of the permutation."""
        if self._support:
            return self(np.eye(max(self._support) + 1, dtype=np.int8))
        else:
            return np.eye(0, dtype=np.int8)

    def replace(self, expr):
        """replaces each part in expr by its image under the permutation."""
        if isinstance(expr, (tuple, list)):
            return type(expr)(self.replace(e) for e in expr)
        elif isinstance(expr, Cycles):
            return Cycles(*[self.replace(cycle) for cycle in expr._cycles])
        else:
            return self._mapping.get(expr, expr)

    def _replace(self, x: int) -> int:
        for cycle in self._cycles:
            if x in cycle:
                return cycle[(cycle.index(x) + 1) % len(cycle)]

    def __call__(self, expr: list | tuple | str | bytes | np.ndarray):
        return permute(expr, self)


class ExCycles(Cycles):

    def __mul__(self, other: Cycles) -> ExCycles:
        c = super().__mul__(other)
        ret = ExCycles()
        if c == Cycles():
            return ret
        ret._cycles = c._cycles
        ret._mapping = c._mapping
        ret._support = c._support
        if isinstance(other, ExCycles):
            ret._expr = self._merge(self._expr, other._expr)
        else:
            ret._expr = self._merge(self._expr, [other])
        return ret

    @staticmethod
    def _merge(*exprs):
        ret = []
        stack = []
        for g in chain.from_iterable(exprs):
            ret.append(g)
            if stack and g == stack[-1][0]:
                stack[-1][1] += 1
            else:
                stack.append([g, 1])

            if stack[-1][0].order == stack[-1][1]:
                _, n = stack.pop()
                for _ in range(n):
                    ret.pop()
        return ret

    def inv(self) -> ExCycles:
        c = super().inv()
        ret = ExCycles()
        ret._cycles = c._cycles
        ret._mapping = c._mapping
        ret._support = c._support
        ret._expr = []
        for c in reversed(self._expr):
            ret._expr.extend([c for _ in range(c.order - 1)])
        return ret


def permute(expr: list | tuple | str | bytes | np.ndarray, perm: Cycles):
    """replaces each part in expr by its image under the permutation."""
    ret = list(expr)
    for cycle in perm._cycles:
        i = cycle[0]
        for j in cycle[1:]:
            ret[i], ret[j] = ret[j], ret[i]
    if isinstance(expr, list):
        return ret
    elif isinstance(expr, tuple):
        return tuple(ret)
    elif isinstance(expr, str):
        return ''.join(ret)
    elif isinstance(expr, bytes):
        return b''.join(ret)
    elif isinstance(expr, np.ndarray):
        return np.array(ret)
    else:
        return ret


def _ne(a, b):
    if isinstance(a, np.ndarray):
        return not np.allclose(a, b)
    else:
        return a != b


def _encode(perm: list, codes: dict) -> list:
    """encode the permutation"""
    ret = []
    for x in perm:
        for k, v in codes.items():
            if _ne(x, v):
                continue
            ret.append(k)
            break
        codes.pop(k)
    return ret


def find_permutation(expr1: list, expr2: list) -> Cycles:
    """find the permutation that transform expr1 to expr2"""
    if len(expr1) != len(expr2):
        raise ValueError("expr1 and expr2 must have the same length")
    codes = {}
    support = []
    perm = []
    for i, (a, b) in enumerate(zip(expr1, expr2)):
        if type(a) != type(b) or _ne(a, b):
            perm.append(b)
            support.append(i)
            codes[i] = a
    if not support:
        return Cycles()
    mapping = {
        k: v
        for k, v in reversed(list(zip(support, _encode(perm, codes))))
        if k != v
    }
    return Cycles._from_sorted_mapping(mapping)


def random_permutation(n: int) -> Cycles:
    """return a random permutation of n elements"""
    cycles = []
    perm = np.random.permutation(n)
    rest = list(perm)
    while len(rest) > 0:
        cycle = [rest.pop(0)]
        el = perm[cycle[-1]]
        while cycle[0] != el:
            cycle.append(el)
            rest.remove(el)
            el = perm[cycle[-1]]
        if len(cycle) > 1:
            cycles.append(tuple(cycle))
    return Cycles(*cycles)


def schreier_tree_(alpha: int, orbit: set[int],
                   generators: list[Cycles]) -> dict[int, ExCycles]:
    """constructs the Schreier tree for the group generated by the generators."""
    # breadth first search to determine the orbit of alpha

    cosetRepresentative = {}

    # group element moving original alpha to actual alpha
    cosetRepresentative[alpha] = ExCycles()
    orbit.remove(alpha)
    if not orbit:
        return cosetRepresentative
    new_nodes = {alpha}

    while new_nodes:
        nodes = new_nodes.copy()
        new_nodes.clear()
        for alpha, g in product(nodes, generators):
            h = cosetRepresentative[alpha]
            # image of actual alpha under generator g
            ag = g.replace(alpha)
            if ag not in cosetRepresentative:
                cosetRepresentative[ag] = h * g
                if ag in orbit:
                    orbit.remove(ag)
                if not orbit:
                    return cosetRepresentative
                new_nodes.add(ag)

    return cosetRepresentative


def schreier_sims(alpha: int, generators: list[Cycles],
                  cosetRepresentative: dict[int, ExCycles]) -> list[ExCycles]:
    # schreier lemma loop to get the schreier generators
    new_generators = set()
    composition_table = set()
    rest_generators = set()

    def _add_new_generator(s, s_inv):
        composition_table.add(s)
        composition_table.add(s_inv)
        for generator in new_generators:
            composition_table.add(generator * s)
            composition_table.add(generator * s_inv)
        new_generators.add(s)

    for g in generators:
        if alpha in g.support:
            rest_generators.add(g)
        else:
            s = g
            s_inv = g.inv()
            _add_new_generator(s, s_inv)

    for (i, s), g in product(cosetRepresentative.items(), rest_generators):
        sd = cosetRepresentative[g.replace(i)]
        sg = (s * g) * sd.inv()
        if not sg.is_identity():
            sg_inv = sg.inv()
            if sg not in composition_table and sg_inv not in composition_table:
                _add_new_generator(sg, sg_inv)
    return list(new_generators)


def grow(cosetRepresentative: dict[int, ExCycles], generators: list[Cycles],
         generator: Cycles) -> dict[int, ExCycles]:
    new_nodes = set()
    nodes = set(cosetRepresentative.keys())

    if not nodes & set(generator.support):
        return cosetRepresentative

    rest = nodes.copy()
    for orbit in PermutationGroup(generators + [generator]).orbits():
        if set(orbit) & rest:
            rest.update(orbit)
    rest = rest - nodes

    def updateCosetRepresentative(ag, h, g, new_nodes, rest):
        if ag not in cosetRepresentative:
            cosetRepresentative[ag] = h * g
            new_nodes.add(ag)
            try:
                rest.remove(ag)
            except:
                pass
            if not rest:
                return True
        return False

    for alpha in nodes:
        h = cosetRepresentative[alpha]
        ag = generator.replace(alpha)
        finished = updateCosetRepresentative(ag, h, generator, new_nodes, rest)
        if finished:
            return cosetRepresentative

    while new_nodes:
        nodes = new_nodes.copy()
        new_nodes.clear()
        for alpha, g in product(nodes, generators + [generator]):
            h = cosetRepresentative[alpha]
            # image of actual alpha under generator g
            ag = g.replace(alpha)
            finished = updateCosetRepresentative(ag, h, g, new_nodes, rest)
            if finished:
                return cosetRepresentative

    return cosetRepresentative


def schreier_tree(alpha: int, generators: list[Cycles]) -> dict[int, ExCycles]:
    """constructs the Schreier tree for the group generated by the generators."""
    cosetRepresentative = {alpha: ExCycles()}
    for i, generator in enumerate(generators):
        cosetRepresentative = grow(cosetRepresentative, generators[:i],
                                   generator)
    return cosetRepresentative


class PermutationGroup():

    def __init__(self, generators: list[Cycles]):
        self.generators = generators
        self._elements = []
        self.fixed_points = ()
        self.stabilizer_chain: tuple[tuple[int], PermutationGroup,
                                     dict[int, ExCycles]] = None

    def __repr__(self) -> str:
        return f"PermutationGroup({self.generators})"

    @staticmethod
    def _generate(generators: list[Cycles]):
        gens = {Cycles()}
        elements = set(generators) | set(g.inv() for g in generators)
        while True:
            new_elements = set()
            for a, b in chain(product(gens, elements), product(elements, gens),
                              product(elements, elements)):
                c = a * b
                if c not in gens and c not in elements:
                    new_elements.add(c)
            gens.update(elements)
            if len(new_elements) == 0:
                break
            elements = new_elements
        return sorted(gens)

    @property
    def support(self):
        support = set()
        for g in self.generators:
            support.update(g.support)
        return sorted(support)

    @property
    def elements(self):
        if self._elements == []:
            self._elements = self._generate(self.generators)
        return self._elements

    def orbits(self, alpha: int | None = None):
        orbit_parts = []
        for g in self.generators:
            for cycle in g._cycles:
                for orbit in orbit_parts:
                    if set(cycle) & set(orbit):
                        orbit.update(cycle)
                        break
                else:
                    orbit_parts.append(set(cycle))
        orbits = []
        for x in orbit_parts:
            for y in orbits:
                if x & y:
                    y.update(x)
                    break
            else:
                orbits.append(x)
        if alpha is None:
            return orbits
        elif isinstance(alpha, int):
            for orbit in orbits:
                if alpha in orbit:
                    return orbit
            return [alpha]

    def make_stabilizer_chain(self, base: list[int] | None = None):
        if len(self.generators) == 0 or self.stabilizer_chain is not None:
            return

        generators = [*self.generators]
        orbits = self.orbits()

        if base is None:
            base = []
        for a in self.support:
            if a not in base:
                base.append(a)

        for i, alpha in enumerate(base):
            for orbit in orbits:
                if alpha in orbit:
                    break
            else:
                continue
            # get the coset representatives for G(alpha)
            cosetRepresentative = schreier_tree(alpha, generators)
            if len(cosetRepresentative) <= 1:
                continue
            break

        # schreier lemma loop to get the schreier generators
        schreierGenerators = schreier_sims(alpha, generators,
                                           cosetRepresentative)
        fixed_points = self.fixed_points + (alpha, )
        sub_group = PermutationGroup(schreierGenerators)
        logger.debug(
            f"{fixed_points} {len(schreierGenerators)=} {len(cosetRepresentative)=}"
        )
        sub_group.fixed_points = fixed_points
        sub_group.make_stabilizer_chain(base[i:])
        self.stabilizer_chain = (fixed_points, sub_group, cosetRepresentative)

    def __len__(self):
        self.make_stabilizer_chain()
        if not self.generators:
            return 1
        else:
            fixed_points, sub_group, cosetRepresentative = self.stabilizer_chain
            return len(cosetRepresentative) * len(sub_group)

    def __getitem__(self, i):
        return self.elements[i]

    def __contains__(self, perm: Cycles):
        if perm in self.generators:
            return True
        if self._elements:
            return perm in self._elements
        try:
            self.express(perm)
            return True
        except _NotContained:
            return False

    def express(self, perm: Cycles):
        if perm.is_identity():
            return ExCycles()
        self.make_stabilizer_chain()
        if not self.generators or self.stabilizer_chain is None:
            raise _NotContained
        fixed_points, sub_group, cosetRepresentative = self.stabilizer_chain
        if set(fixed_points) & set(perm.support):
            for x in cosetRepresentative.values():
                if not set((perm * x.inv()).support) & set(fixed_points):
                    return sub_group.express(perm * x.inv()) * x
            else:
                raise _NotContained
        else:
            return sub_group.express(perm)


class SymmetricGroup(PermutationGroup):

    def __init__(self, N: int):
        if N < 2:
            super().__init__([])
        elif N == 2:
            super().__init__([Cycles((0, 1))])
        else:
            super().__init__([Cycles((0, 1)), Cycles(tuple(range(N)))])
        self.N = N

    def __repr__(self) -> str:
        return f"SymmetricGroup({self.N})"

    def __len__(self):
        return np.math.factorial(self.N)

    def __contains__(self, perm: Cycles):
        return set(perm.support) <= set(range(self.N))


class CyclicGroup(PermutationGroup):

    def __init__(self, N: int):
        if N < 2:
            super().__init__([])
        else:
            super().__init__([Cycles(tuple(range(N)))])
        self.N = N

    def __repr__(self) -> str:
        return f"CyclicGroup({self.N})"

    def __len__(self):
        return max(self.N, 1)


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

    def __repr__(self) -> str:
        return f"DihedralGroup({self.N})"

    def __len__(self):
        if self.N == 1:
            return 1
        elif self.N == 2:
            return 2
        return max(2 * self.N, 1)


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

    def __repr__(self) -> str:
        return f"AbelianGroup{self.n}"

    def __len__(self):
        return max(np.multiply.reduce(self.n), 1)


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

    def __repr__(self) -> str:
        return f"AlternatingGroup({self.N})"

    def __len__(self):
        return max(np.math.factorial(self.N) // 2, 1)

    def __contains__(self, perm: Cycles):
        return perm in SymmetricGroup(self.N) and perm.signature == 1
