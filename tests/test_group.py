import functools
import operator

from waveforms.math.group import *


def test_init():
    G = PermutationGroup([Cycles((1, 2, 3)), Cycles((4, 5))])
    assert G.generators == [Cycles((1, 2, 3)), Cycles((4, 5))]


def test_order():
    G = PermutationGroup([Cycles((1, 2, 3)), Cycles((4, 5))])
    assert G.order() == 6

    G = PermutationGroup([
        Cycles((1, 2, 3, 4), (8, 17, 14, 11), (7, 20, 13, 10)),
        Cycles((5, 6, 7, 8), (1, 9, 21, 17), (2, 10, 22, 18)),
        Cycles((9, 10, 11, 12), (1, 13, 23, 5), (4, 16, 22, 8)),
        Cycles((13, 14, 15, 16), (3, 19, 23, 11), (4, 20, 24, 12)),
        Cycles((17, 18, 19, 20), (2, 6, 24, 14), (3, 7, 21, 15)),
        Cycles((21, 22, 23, 24), (5, 12, 15, 18), (6, 9, 16, 19))
    ])
    assert G.order() == 88179840


def test_orbit():
    G = SymmetricGroup(5)
    assert set(G.orbit("aaaaa")) == {"aaaaa"}
    assert set(
        G.orbit("aaaab")) == {"aaaab", "baaaa", "abaaa", "aabaa", "aaaba"}
    assert set(G.orbit("aaabb")) == {
        'aaabb', 'aabab', 'baaab', 'babaa', 'baaba', 'abbaa', 'bbaaa', 'aabba',
        'ababa', 'abaab'
    }


def test_contains():
    rot1 = Cycles((1, 3, 8, 6), (2, 5, 7, 4), (9, 48, 15, 12),
                  (10, 47, 16, 13), (11, 46, 17, 14))

    rot2 = Cycles((6, 15, 35, 26), (7, 22, 34, 19), (8, 30, 33, 11),
                  (12, 14, 29, 27), (13, 21, 28, 20))

    rot3 = Cycles((1, 12, 33, 41), (4, 20, 36, 44), (6, 27, 38, 46),
                  (9, 11, 26, 24), (10, 19, 25, 18))

    rot4 = Cycles((1, 24, 40, 17), (2, 18, 39, 23), (3, 9, 38, 32),
                  (41, 43, 48, 46), (42, 45, 47, 44))

    rot5 = Cycles((3, 43, 35, 14), (5, 45, 37, 21), (8, 48, 40, 29),
                  (15, 17, 32, 30), (16, 23, 31, 22))

    rot6 = Cycles((24, 27, 30, 43), (25, 28, 31, 42), (26, 29, 32, 41),
                  (33, 35, 40, 38), (34, 37, 39, 36))

    RubikGroup = PermutationGroup([rot1, rot2, rot3, rot4, rot5, rot6])

    assert RubikGroup.order() == 43252003274489856000

    assert Cycles(1, 9, 46) not in RubikGroup
    assert Cycles((1, 3), (9, 48), (17, 46)) not in RubikGroup
    assert Cycles(2, 47) not in RubikGroup
    assert Cycles((2, 47), (31, 37)) in RubikGroup
    assert Cycles((1, 9, 46), (3, 48, 17), (8, 15, 14)) in RubikGroup
    superflip = Cycles((2, 47), (4, 10), (7, 13), (5, 16), (20, 19), (21, 22),
                       (28, 34), (18, 44), (25, 36), (45, 23), (42, 39),
                       (31, 37))
    assert superflip in RubikGroup


def test_rank():
    G = SymmetricGroup(5)
    for i, g in enumerate(G.generate_schreier_sims()):
        assert G.coset_unrank(i) == g
        assert i == G.coset_rank(g)
    assert i + 1 == 120


def test_random():
    G = SymmetricGroup(10)
    for i in range(10):
        g = G.random()
        assert g in G


def test_subgroup():
    G = PermutationGroup([Cycles((1, 2, 3)), Cycles((4, 5))])
    assert G <= SymmetricGroup(6)
    assert G < SymmetricGroup(6)
    assert G <= G
    assert G == G
    assert G != SymmetricGroup(6)
    assert not G < SymmetricGroup(5)


def test_simplify():
    group = SymmetricGroup(5)

    lst1 = []
    for i in range(100):
        lst1.append(group.random())

    perm = functools.reduce(operator.mul, lst1)
    perm2 = perm.inv()

    assert (perm * perm2).is_identity()

    perm.simplify()

    lst2 = []
    for g, n in perm._expr:
        for _ in range(n):
            lst2.append(g)

    assert functools.reduce(operator.mul, lst2) == perm

    perm2 = group.express(perm2)
    perm2.simplify()
    for g, n in perm2._expr:
        for _ in range(n):
            lst2.append(g)
    assert functools.reduce(operator.mul, lst2) == Cycles()
