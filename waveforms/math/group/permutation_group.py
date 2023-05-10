from __future__ import annotations

import functools
from itertools import chain, product

import numpy as np


@functools.total_ordering
class Cycles():

    __slots__ = ('_cycles', '_support', '_mapping')

    def __init__(self, *cycles):
        self._mapping = {}
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


class PermutationGroup():

    def __init__(self, generators: list[Cycles]):
        self.generators = generators
        self._elements = []
        self._stabilizer_chain = None

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

    def orbits(self, element: int | None = None):
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
                if element is None or element in x:
                    orbits.append(x)
        return orbits

    def _stabilizer_chain(self):
        if self._stabilizer_chain is None:
            self._stabilizer_chain = schreier_sims(self)

    def __len__(self):
        self._stabilizer_chain()
        return np.multiply.reduce(self._stabilizer_chain[0])

    def __getitem__(self, i):
        return self.elements[i]


def schreier_tree(
        alpha: int,
        orbit: set[int],
        generators: list[Cycles],
        gen: Cycles = Cycles(),
        cosetRepresentative: dict[int, Cycles] | None = None
) -> dict[int, Cycles]:
    """constructs the Schreier tree for the group generated by the generators."""
    # depth first search to determine the orbit of alpha

    if cosetRepresentative is None:
        cosetRepresentative = {}

    # group element moving original alpha to actual alpha
    cosetRepresentative[alpha] = gen
    orbit.remove(alpha)
    if not orbit:
        return cosetRepresentative

    for g in generators:
        ag = g.replace(alpha)  # image of actual alpha under generator g
        if ag not in cosetRepresentative:
            cosetRepresentative = schreier_tree(ag, orbit, generators, gen * g,
                                                cosetRepresentative)
            if not orbit:
                break
    return cosetRepresentative


def schreier_tree(alpha: int, orbit: set[int],
                  generators: list[Cycles]) -> dict[int, Cycles]:
    """constructs the Schreier tree for the group generated by the generators."""
    # breadth first search to determine the orbit of alpha

    cosetRepresentative = {}

    # group element moving original alpha to actual alpha
    cosetRepresentative[alpha] = Cycles()
    orbit.remove(alpha)
    if not orbit:
        return cosetRepresentative
    new_nodes = {alpha}

    while new_nodes:
        nodes = new_nodes.copy()
        new_nodes.clear()
        for alpha in nodes:
            gen = cosetRepresentative[alpha]
            for g in generators:
                ag = g.replace(
                    alpha)  # image of actual alpha under generator g
                if ag not in cosetRepresentative:
                    cosetRepresentative[ag] = gen * g
                    orbit.remove(ag)
                    if not orbit:
                        return cosetRepresentative
                    new_nodes.add(ag)

    return cosetRepresentative


def schreier_sims(group: PermutationGroup):
    orders = []

    generators = [*group.generators]
    orbits = group.orbits()

    stabilizer_chain = []
    fixed_points = ()

    for alpha in group.support:
        for orbit in orbits:
            if alpha in orbit:
                break
        else:
            continue
        # get the coset representatives for G(alpha)
        cosetRepresentative = schreier_tree(alpha, orbit.copy(), generators)
        if len(cosetRepresentative) <= 1:
            continue
        # schreier lemma loop to get the schreier generators
        new_generators = set()
        composition_table = set()

        for i, s in cosetRepresentative.items():
            for g in generators:
                sd = cosetRepresentative[g.replace(i)]
                sg = s * g * sd.inv()
                if not sg.is_identity():
                    sg_inv = sg.inv()
                    if sg not in composition_table and sg_inv not in composition_table:
                        for generator in new_generators:
                            composition_table.add(generator * sg)
                            composition_table.add(generator * sg_inv)
                        new_generators.add(sg)

        orders.append(len(cosetRepresentative))
        sub_group = PermutationGroup(list(new_generators))
        stabilizer_chain.append((fixed_points, sub_group))
        fixed_points = fixed_points + (alpha, )
        generators = list(new_generators)
        orbits = PermutationGroup(generators).orbits()

    return orders, stabilizer_chain


class StabilizerChain():

    def __init__(self, stabPoint):
        self.stabPoint = stabPoint
        # An index into the base for the point stabilized by this group's subgroup.
        self.orbitTree = {}
        # A tree to keep track of the orbit in our group of the point stabilized by our subgroup.
        self.transversalSet = {}
        # A set of coset representatives of this group's subgroup.
        self.generators = set()
        # A set of permutations generating this group.
        self.subGroup = None
        # A pointer to this group's subgroup, or null to mean the trivial group.

    def is_member(self, g: Cycles) -> bool:
        return g.is_identity() or g in self.generators

    def extend(self, generator: list[Cycles], support: list[int]):

        # This is the major optimization of Schreier-Sims.  Weed out redundant Schreier generators.
        if (self.is_member(generator)):
            return

        # Our group just got bigger, but the stabilizer chain rooted at our subgroup is still the same.
        self.generators.add(generator)

        # Explore all new orbits we can reach with the addition of the new generator.
        # Note that if the tree was empty to begin with, the identity must be returned
        # in the set to satisfy a condition of Schreier's lemma.
        newTerritorySet = self.orbitTree.Grow(generator, support)

        # By the orbit-stabilizer theorem, the permutations in the returned set are
        # coset representatives of the cosets of our subgroup.
        for permutation in newTerritorySet:
            self.transversalSet.Add(permutation)

        # We now apply Schreier's lemma to find new generators for our subgroup.
        # Some iterations of this loop are redundant, but we ignore that for simplicity.
        for cosetRepresentative in self.transversalSet:

            for generator in self.generators:
                schreierGenerator = self.CalcSchreierGenerator(
                    cosetRepresentative, generator)
                if (schreierGenerator.is_identity()):
                    continue

                if self.subGroup is None:
                    self.subGroup = StabilizerChain(self.stabPoint + 1)

                self.subGroup.extend(schreierGenerator, support)


def MakeStabChain(generators: list[Cycles], support: list[int]):

    stab_chain = StabilizerChain(0)
    for generator in generators:
        stab_chain.extend(generator, support)
    return stab_chain


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