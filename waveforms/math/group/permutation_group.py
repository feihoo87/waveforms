from __future__ import annotations

import bisect
import copy
import functools
import logging
import math
import operator
import random
import weakref
from itertools import chain, combinations, product
from typing import Callable, TypeVar

import numpy as np

logger = logging.getLogger(__name__)


class _NotContained(Exception):
    pass


@functools.total_ordering
class Cycles():

    __slots__ = ('_cycles', '_support', '_mapping', '_expr', '_order')

    def __init__(self, *cycles):
        self._mapping = {}
        self._expr: list[Cycles] = []
        self._order = None
        if len(cycles) == 0:
            self._cycles = ()
            self._support = ()
            return

        if not isinstance(cycles[0], (list, tuple)):
            cycles = (cycles, )

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

    def __rmul__(self, other: Cycles) -> Cycles:
        return other.__mul__(self)

    @staticmethod
    def _from_sorted_mapping(mapping: dict[int, int]) -> Cycles:
        c = Cycles()
        if not mapping:
            return c

        c._support = tuple(reversed(mapping.keys()))
        c._mapping = mapping.copy()
        c._order = 1

        cycles = []
        while mapping:
            k, el = mapping.popitem()
            cycle = [k]
            while k != el:
                cycle.append(el)
                el = mapping.pop(el)
            cycles.append(tuple(cycle))
            c._order = math.lcm(c._order, len(cycle))
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
        c._order = self._order
        return c

    @property
    def order(self):
        """Returns the order of the permutation.

        The order of a permutation is the least integer n such that
        p**n = e, where e is the identity permutation.
        """
        if self._order is None:
            self._order = math.lcm(*[len(cycle) for cycle in self._cycles])
        return self._order

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
            return permute(np.eye(max(self._support) + 1, dtype=np.int8), self)
        else:
            return np.eye(0, dtype=np.int8)

    def replace(self, expr):
        """replaces each part in expr by its image under the permutation."""
        if isinstance(expr, (tuple, list)):
            return type(expr)(self.replace(e) for e in expr)
        elif isinstance(expr, Cycles):
            return Cycles(*[self.replace(cycle) for cycle in expr._cycles])
        else:
            return self._replace(expr)

    def _replace(self, x: int) -> int:
        return self._mapping.get(x, x)

    def __call__(self, *cycle):
        return self * Cycles(*cycle)

    def commutator(self, x: Cycles) -> Cycles:
        """Return the commutator of ``self`` and ``x``: ``self*x*self.inv()*x.inv()``

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Commutator
        """
        return self * x * self.inv() * x.inv()

    def simplify(self) -> Cycles:
        return self


class _ExCycles(Cycles):

    def __mul__(self, other: Cycles) -> _ExCycles:
        c = super().__mul__(other)
        ret = _ExCycles()
        if c == Cycles():
            return ret
        ret._cycles = c._cycles
        ret._mapping = c._mapping
        ret._support = c._support
        ret._order = c._order
        ret._expr = (self, other)
        return ret

    def __rmul__(self, other: Cycles) -> _ExCycles:
        c = super().__rmul__(other)
        ret = _ExCycles()
        if c == Cycles():
            return ret
        ret._cycles = c._cycles
        ret._mapping = c._mapping
        ret._support = c._support
        ret._order = c._order
        ret._expr = (other, self)
        return ret

    def simplify(self) -> Cycles:
        if isinstance(self._expr, tuple):
            self._simplify()
        return self

    def _simplify(self):
        if isinstance(self._expr[0], str):
            self._simplify_inv()
        else:
            self._simplify_mul()

    def _simplify_inv(self):
        self._expr[1].simplify()
        if isinstance(self._expr[1], _ExCycles):
            expr = self._expr[1]._expr
        else:
            expr = [[self._expr[1], 1]]
        ret = []
        for g, n in expr:
            if ret and ret[-1][0] == g:
                ret[-1][1] = (ret[-1][1] + n) % g.order
                if ret[-1][1] == 0:
                    ret.pop()
            else:
                ret.append([g, n])
        self._expr = ret

    def _simplify_mul(self):
        self._expr[0].simplify()
        self._expr[1].simplify()
        if isinstance(self._expr[0], _ExCycles):
            ret = self._expr[0]._expr.copy()
        else:
            ret = [[self._expr[0], 1]]
        if isinstance(self._expr[1], _ExCycles):
            expr = self._expr[1]._expr
        else:
            expr = [[self._expr[1], 1]]

        for g, n in expr:
            if ret and ret[-1][0] == g:
                ret[-1][1] = (ret[-1][1] + n) % g.order
                if ret[-1][1] == 0:
                    ret.pop()
            else:
                ret.append([g, n])
        self._expr = ret

    def inv(self) -> _ExCycles:
        c = super().inv()
        ret = _ExCycles()
        ret._cycles = c._cycles
        ret._mapping = c._mapping
        ret._support = c._support
        ret._order = c._order
        ret._expr = ('-', self)
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


class PermutationGroup():

    def __init__(self, generators: list[Cycles]):
        self.generators = generators
        self._elements = []
        self._support = None

        self._order = None
        self._orbits = None
        self._center = []
        self._is_abelian = None
        self._is_trivial = None

        # these attributes are assigned after running schreier_sims
        self._base = []
        self._strong_gens = []
        self._basic_orbits = []
        self._transversals: list[dict[int, Cycles]] = []

    def __repr__(self) -> str:
        return f"PermutationGroup({self.generators})"

    def is_trivial(self):
        """Test if the group is the trivial group.

        This is true if the group contains only the identity permutation.
        """
        if self._is_trivial is None:
            self._is_trivial = len(self.generators) == 0
        return self._is_trivial

    def is_abelian(self):
        """Test if the group is Abelian.
        """
        if self._is_abelian is not None:
            return self._is_abelian

        self._is_abelian = True
        for x, y in combinations(self.generators, 2):
            if not x * y == y * x:
                self._is_abelian = False
                break
        return True

    def is_subgroup(self, G: PermutationGroup):
        """Return ``True`` if all elements of ``self`` belong to ``G``."""
        if not isinstance(G, PermutationGroup):
            return False
        if self == G or self.is_trivial():
            return True
        if G.order() % self.order() != 0:
            return False
        return all(g in G for g in self.generators)

    def generate(self, method: str = "schreier_sims"):
        if method == "schreier_sims":
            yield from self.generate_schreier_sims()
        elif method == "dimino":
            yield from self.generate_dimino(self.generators)

    @staticmethod
    def generate_dimino(generators: list[Cycles]):
        """Yield group elements using Dimino's algorithm."""
        e = Cycles()
        yield e
        gens = {e}
        elements = set(generators) | set(g.inv() for g in generators)
        while True:
            new_elements = set()
            for a, b in chain(product(gens, elements), product(elements, gens),
                              product(elements, elements)):
                c = a * b
                if c not in gens and c not in elements and c not in new_elements:
                    new_elements.add(c)
                    yield c
            gens.update(elements)
            if len(new_elements) == 0:
                break
            elements = new_elements

    def generate_schreier_sims(self):
        """Yield group elements using the Schreier-Sims representation
        in coset_rank order
        """
        if self.is_trivial():
            yield Cycles()
            return

        self.schreier_sims()
        for gens in product(
                *
            [list(coset.values()) for coset in reversed(self._transversals)]):
            yield functools.reduce(operator.mul, gens)

    @property
    def support(self):
        if self._support is None:
            support = set()
            for g in self.generators:
                support.update(g.support)
            self._support = sorted(support)
        return self._support

    @property
    def elements(self):
        if self._elements == []:
            for g in self.generate():
                bisect.insort(self._elements, g)
        return self._elements

    def random(self, N=1):
        """Return a random element of the group.

        If N > 1, return a list of N random elements.
        """
        self.schreier_sims()
        transversals = self._transversals
        orbits = self._basic_orbits
        ret = []
        for _ in range(N):
            g = Cycles()
            for orbit, coset in zip(orbits, transversals):
                g *= coset[random.choice(orbit)]
            ret.append(g)
        if N == 1:
            return ret[0]
        return ret

    @property
    def base(self):
        """Return a base from the Schreier-Sims algorithm."""
        if self._base == []:
            self.schreier_sims()
        return self._base

    def orbit(
        self,
        alpha: TypeVar['T'],
        action: Callable[[TypeVar['T'], Cycles], TypeVar['T']] | None = None
    ) -> list[TypeVar['T']]:
        """finds the orbit under the group action given by a function `action`
        """
        if isinstance(alpha, int) and action is None:
            for orbit in self.orbits():
                if alpha in orbit:
                    return orbit
            else:
                return [alpha]
        elif isinstance(alpha, Cycles) and action is None:
            action = lambda x, y: y * x
        elif action is None:
            action = permute
        orbit = [alpha]
        for beta in orbit:
            for g in self.generators:
                beta = action(beta, g)
                if beta not in orbit:
                    orbit.append(beta)
        return orbit

    def orbits(self):
        if self._orbits is None:
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
            self._orbits = orbits
        return self._orbits

    def schreier_sims(self, base: list[int] | None = None):
        """Schreier-Sims algorithm.

        Explanation
        ===========

        It computes the generators of the chain of stabilizers
        `G > G_{b_1} > .. > G_{b1,..,b_r} > 1`
        in which `G_{b_1,..,b_i}` stabilizes `b_1,..,b_i`,
        and the corresponding ``s`` cosets.
        An element of the group can be written as the product
        `h_1*..*h_s`.

        We use the incremental Schreier-Sims algorithm.
        """
        if self._transversals and (base is None or base == self._base):
            return

        base, strong_gens = schreier_sims_incremental(self.generators,
                                                      base=base)
        self._base = base
        self._strong_gens = strong_gens
        if not base:
            self._transversals = []
            self._basic_orbits = []
            return

        strong_gens_distr = _distribute_gens_by_base(base, strong_gens)

        # Compute basic orbits and transversals from a base and strong generating set.
        transversals = []
        basic_orbits = []
        for alpha, gens in zip(base, strong_gens_distr):
            transversal = _orbit_transversal(gens, alpha)
            basic_orbits.append(list(transversal.keys()))
            transversals.append(transversal)

        self._transversals = transversals
        self._basic_orbits = [sorted(x) for x in basic_orbits]

    def order(self):
        if self._order is None:
            if self.is_trivial():
                self._order = 1
            else:
                self.schreier_sims()
                self._order = math.prod(len(x) for x in self._transversals)
        return self._order

    def index(self, H: PermutationGroup):
        """
        Returns the index of a permutation group.

        Examples
        ========

        >>> a = Permutation(1,2,3)
        >>> b =Permutation(3)
        >>> G = PermutationGroup([a])
        >>> H = PermutationGroup([b])
        >>> G.index(H)
        3

        """
        if H.is_subgroup(self):
            return self.order() // H.order()

    def __len__(self):
        return self.order()

    def __getitem__(self, i):
        return self.elements[i]

    def __contains__(self, perm: Cycles):
        if perm in self.generators or perm.is_identity():
            return True
        if self._elements:
            return perm in self._elements
        try:
            perm = self.coset_factor(perm)
            return True
        except _NotContained:
            return False

    def __eq__(self, other) -> bool:
        """Return ``True`` if PermutationGroup generated by elements in the
        group are same i.e they represent the same PermutationGroup.
        """
        if not isinstance(other, PermutationGroup):
            raise TypeError(
                f"'==' not supported between instances of '{type(self)}' and '{type(other)}'"
            )

        set_self_gens = set(self.generators)
        set_other_gens = set(other.generators)

        # before reaching the general case there are also certain
        # optimisation and obvious cases requiring less or no actual
        # computation.
        if set_self_gens == set_other_gens:
            return True

        # in the most general case it will check that each generator of
        # one group belongs to the other PermutationGroup and vice-versa
        for gen1 in set_self_gens:
            if gen1 not in other:
                return False
        for gen2 in set_other_gens:
            if gen2 not in self:
                return False
        return True

    def __lt__(self, other) -> bool:
        if isinstance(other, PermutationGroup):
            return self.is_subgroup(other) and self.order() < other.order()
        else:
            raise TypeError(
                f"'<' not supported between instances of '{type(self)}' and '{type(other)}'"
            )

    def __le__(self, other) -> bool:
        if isinstance(other, PermutationGroup):
            return self.is_subgroup(other)
        else:
            raise TypeError(
                f"'<=' not supported between instances of '{type(self)}' and '{type(other)}'"
            )

    def __mul__(self, other: Cycles):
        if other in self:
            return self
        return Coset(self, other, left=False)

    def __rmul__(self, other: Cycles):
        if other in self:
            return self
        return Coset(self, other, left=True)

    def coset_factor(self, g: Cycles, index=False):
        """Return ``G``'s (self's) coset factorization of ``g``

        Explanation
        ===========

        If ``g`` is an element of ``G`` then it can be written as the product
        of permutations drawn from the Schreier-Sims coset decomposition,

        The permutations returned in ``f`` are those for which
        the product gives ``g``: ``g = f[n]*...f[1]*f[0]`` where ``n = len(B)``
        and ``B = G.base``. f[i] is one of the permutations in
        ``self._basic_orbits[i]``.
        """
        self.schreier_sims()
        factors = []
        for alpha, coset, orbit in zip(self._base, self._transversals,
                                       self._basic_orbits):
            beta = g._replace(alpha)
            if beta == alpha:
                if index:
                    factors.append(0)
                continue
            if beta not in coset:
                raise _NotContained
            u = coset[beta]
            if index:
                factors.append(orbit.index(beta))
            else:
                factors.append(u)
            g = g * u.inv()
            if g.is_identity():
                break
        if not g.is_identity():
            raise _NotContained
        return factors

    def coset_rank(self, g):
        """rank using Schreier-Sims representation.

        Explanation
        ===========

        The coset rank of ``g`` is the ordering number in which
        it appears in the lexicographic listing according to the
        coset decomposition

        The ordering is the same as in G.generate(method='coset').
        If ``g`` does not belong to the group it returns None.
        """
        try:
            index = self.coset_factor(g, index=True)
            index = index + [0] * (len(self._transversals) - len(index))
        except _NotContained:
            raise IndexError(f"Permutation {g} not contained in group.")
        rank = 0
        b = 1
        for i, coset in zip(index, self._transversals):
            rank += b * i
            b = b * len(coset)
        return rank

    def coset_unrank(self, rank):
        """unrank using Schreier-Sims representation

        coset_unrank is the inverse operation of coset_rank
        if 0 <= rank < order; otherwise it returns None.

        """
        if rank < 0 or rank >= self.order():
            return None
        transversals = self._transversals
        orbits = self._basic_orbits
        ret = Cycles()
        for orbit, coset in zip(orbits, transversals):
            rank, c = divmod(rank, len(coset))
            ret = coset[orbit[c]] * ret
        return ret

    def express(self, perm: Cycles):
        if perm.is_identity():
            return Cycles()
        self.schreier_sims()
        return functools.reduce(operator.mul, self.coset_factor(perm)[::-1])

    def stabilizer_chain(self) -> list[tuple[tuple[int], PermutationGroup]]:
        r"""
        Return a chain of stabilizers relative to a base and strong generating
        set.

        Explanation
        ===========

        The ``i``-th basic stabilizer `G^{(i)}` relative to a base
        `(b_1, b_2, \dots, b_k)` is `G_{b_1, b_2, \dots, b_{i-1}}`.
        """
        self.schreier_sims()
        strong_gens = self._strong_gens
        base = self._base
        if not base:  # e.g. if self is trivial
            return []
        strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
        basic_stabilizers = []
        for i, gens in enumerate(strong_gens_distr):
            basic_stabilizers.append((tuple(base[:i]), PermutationGroup(gens)))
        basic_stabilizers.append((tuple(base), PermutationGroup([])))
        return basic_stabilizers

    def stabilizer(self, alpha) -> PermutationGroup:
        """Return the stabilizer subgroup of ``alpha``."""
        orb = [alpha]
        table = {alpha: Cycles()}
        table_inv = {alpha: Cycles()}
        used = {}
        used[alpha] = True
        stab_gens = []
        for b in orb:
            for gen in self.generators:
                temp = gen[b]
                if temp not in used:
                    gen_temp = table[b] * gen
                    orb.append(temp)
                    table[temp] = gen_temp
                    table_inv[temp] = gen_temp.inv()
                    used[temp] = True
                else:
                    schreier_gen = table[b] * gen * table_inv[temp]
                    if schreier_gen not in stab_gens:
                        stab_gens.append(schreier_gen)
        return PermutationGroup(stab_gens)

    def centralizer(self, H: PermutationGroup) -> PermutationGroup:
        """Return the centralizer of ``H`` in ``self``."""
        raise NotImplementedError

    def normalizer(self, H: PermutationGroup) -> PermutationGroup:
        """Return the normalizer of ``H`` in ``self``."""
        raise NotImplementedError

    def center(self) -> PermutationGroup:
        """Return the center of group."""
        return self.centralizer(self)


class Coset():

    def __init__(self, H: PermutationGroup, g: Cycles, left: bool = True):
        self._left = left
        self._norm = True
        self.H = H
        self.g = g
        for gen in self.H.generators:
            if gen * self.g not in self.g * self.H:
                self._norm = False
                break

    def is_left_coset(self):
        return self._left

    def is_right_coset(self):
        return not self._left

    def __contains__(self, perm: Cycles):
        if self._left:
            return self.g * perm in self.H
        else:
            return perm * self.g in self.H

    def generate(self):
        if self._left:
            for perm in self.H.generate():
                yield self.g * perm
        else:
            for perm in self.H.generate():
                yield perm * self.g

    def __mul__(self, other: PermutationGroup | Coset) -> Coset:
        if isinstance(other, PermutationGroup) and other == self.H:
            return self
        elif isinstance(other, Coset) and other.H == self.H:
            return Coset(self.H, self.g * other.g, self._left)
        else:
            raise TypeError(f"Cannot multiply {self} by {other}")


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


def _strip(h: Cycles, base, orbits, transversals, j):
    """
    """
    base_len = len(base)
    for i in range(j + 1, base_len):
        beta = h._replace(base[i])
        if beta == base[i]:
            continue
        if beta not in orbits[i]:
            return h, i + 1
        u = transversals[i][beta]
        if h == u:
            return None, base_len + 1
        h = h * u.inv()
    return h, base_len + 1


def _orbit_transversal(
    generators: list[Cycles],
    alpha: int,
    Identity: Cycles = _ExCycles(),
) -> tuple[list[tuple[int, Cycles]], dict[int, list[int]]]:
    r"""Computes a transversal for the orbit of ``alpha`` as a set.

    Explanation
    ===========

    generators   generators of the group ``G``

    For a permutation group ``G``, a transversal for the orbit
    `Orb = \{g(\alpha) | g \in G\}` is a set
    `\{g_\beta | g_\beta(\alpha) = \beta\}` for `\beta \in Orb`.
    Note that there may be more than one possible transversal.
    """
    tr = [(alpha, Identity)]
    db = {alpha}
    for x, px in tr:
        for i, gen in enumerate(generators):
            temp = gen._replace(x)
            if temp not in db:
                db.add(temp)
                tr.append((temp, px * gen))

    return dict(tr)


def _distribute_gens_by_base(base: list,
                             gens: list[Cycles]) -> list[list[Cycles]]:
    r"""
    Distribute the group elements ``gens`` by membership in basic stabilizers.

    Explanation
    ===========

    Notice that for a base `(b_1, b_2, \dots, b_k)`, the basic stabilizers
    are defined as `G^{(i)} = G_{b_1, \dots, b_{i-1}}` for
    `i \in\{1, 2, \dots, k\}`.

    Parameters
    ==========

    base : a sequence of points in `\{0, 1, \dots, n-1\}`
    gens : a list of elements of a permutation group of degree `n`.

    Returns
    =======
    list
        List of length `k`, where `k` is the length of *base*. The `i`-th entry
        contains those elements in *gens* which fix the first `i` elements of
        *base* (so that the `0`-th entry is equal to *gens* itself). If no
        element fixes the first `i` elements of *base*, the `i`-th element is
        set to a list containing the identity element.
    """
    base_len = len(base)
    stabs = [[] for _ in range(base_len)]
    max_stab_index = 0
    for gen in gens:
        j = 0
        while j < base_len - 1 and gen._replace(base[j]) == base[j]:
            j += 1
        if j > max_stab_index:
            max_stab_index = j
        for k in range(j + 1):
            stabs[k].append(gen)
    for i in range(max_stab_index + 1, base_len):
        stabs[i].append(Cycles())
    return stabs


def schreier_sims_incremental(
    gens: list[Cycles],
    base: list[int] | None = None
) -> tuple[list[int], list[Cycles], dict[int, list[int]]]:
    """Extend a sequence of points and generating set to a base and strong
    generating set.

    Parameters
    ==========
    gens
        The generating set to be extended to a strong generating set
        relative to the base obtained.

    base
        The sequence of points to be extended to a base. Optional
        parameter with default value ``[]``.

    Returns
    =======

    (base, strong_gens)
        ``base`` is the base obtained, and ``strong_gens`` is the strong
        generating set relative to it. The original parameters ``base``,
        ``gens`` remain unchanged.
    """
    if base is None:
        base = []
    else:
        base = base.copy()
    support = set()
    for g in gens:
        support.update(g.support)
    # handle the trivial group
    if len(gens) == 1 and gens[0].is_identity():
        return base, gens, {gens[0]: [gens[0]]}
    # remove the identity as a generator
    gens = [x for x in gens if not x.is_identity()]
    # make sure no generator fixes all base points
    for gen in gens:
        if all(x == gen._replace(x) for x in base):
            for new in support:
                if gen._replace(new) != new:
                    break
            else:
                assert None  # can this ever happen?
            base.append(new)
    #logger.debug("Schreier-Sims: base = %s, gens = %s", _base, _gens)
    # distribute generators according to basic stabilizers
    strong_gens_distr = _distribute_gens_by_base(base, gens)
    new_strong_gens = []
    # initialize the basic stabilizers, basic orbits and basic transversals
    orbs = {}
    transversals = {}
    for i, alpha in enumerate(base):
        transversals[i] = _orbit_transversal(strong_gens_distr[i], alpha)
        orbs[i] = list(transversals[i].keys())
    # main loop: amend the stabilizer chain until we have generators
    # for all stabilizers
    base_len = len(base)
    i = base_len - 1
    while i >= 0:
        # this flag is used to continue with the main loop from inside
        # a nested loop
        continue_i = False
        # test the generators for being a strong generating set
        db = {}
        for beta, u_beta in list(transversals[i].items()):
            for j, gen in enumerate(strong_gens_distr[i]):
                gb = gen._replace(beta)
                u1 = transversals[i][gb]
                g1 = u_beta * gen
                if g1 != u1:
                    # test if the schreier generator is in the i+1-th
                    # would-be basic stabilizer
                    new_strong_generator_found = False
                    try:
                        u1_inv = db[gb]
                    except KeyError:
                        u1_inv = db[gb] = u1.inv()
                    schreier_gen = g1 * u1_inv
                    h, j = _strip(schreier_gen, base, orbs, transversals, i)
                    if j <= base_len:
                        # new strong generator h at level j
                        new_strong_generator_found = True
                    elif h is not None:
                        # h fixes all base points
                        new_strong_generator_found = True
                        for moved in support:
                            if h._replace(moved) != moved:
                                break
                        base.append(moved)
                        base_len += 1
                        strong_gens_distr.append([])
                    if new_strong_generator_found:
                        # if a new strong generator is found, update the
                        # data structures and start over
                        new_strong_gens.append(h)
                        for l in range(i + 1, j):
                            strong_gens_distr[l].append(h)
                            transversals[l] =\
                            _orbit_transversal(strong_gens_distr[l],
                                base[l])
                            orbs[l] = list(transversals[l].keys())
                        i = j - 1
                        # continue main loop using the flag
                        continue_i = True
                if continue_i is True:
                    break
            if continue_i is True:
                break
        logger.debug(
            "Schreier-Sims: i = %s, continue_i = %s, len(transversals[i]) = %s",
            i, continue_i, len(transversals[i]))
        if continue_i is True:
            continue
        i -= 1

    return (base, gens + new_strong_gens)
