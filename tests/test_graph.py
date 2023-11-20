from waveforms.math.graph import *


def graph_nodes(edges):
    nodes = set()
    for u, v, *_ in edges:
        nodes.add(u)
        nodes.add(v)
    return nodes


def assert_minimum_spanning_tree(edges, result):
    ret = minimum_spanning_tree(edges)
    assert graph_nodes(ret) == graph_nodes(edges)
    assert set(ret) == set(result)


def test_minimum_spanning_tree():
    assert minimum_spanning_tree([]) == []
    assert minimum_spanning_tree([(1, 1)]) == [(1, 1)]

    edges = [(1, 2, 3), (2, 3, 1), (3, 4, 4), (1, 4, 2)]
    assert set(minimum_spanning_tree(edges)) == set([(1, 2), (1, 4), (2, 3)])

    edges = [(1, 2, 1), (3, 4, 1)]
    assert set(minimum_spanning_tree(edges)) == set([(1, 2), (3, 4)])

    edges = [(1, 2, 1), (2, 3, 1), (1, 3, 1)]
    assert set(minimum_spanning_tree(edges)) in [
        set([(1, 2), (2, 3)]),
        set([(1, 3), (2, 3)]),
        set([(1, 2), (1, 3)])
    ]


def test_minimum_spanning_tree2():
    edges = [(1, 2, 4), (1, 3, 1), (2, 3, 3), (2, 4, 2), (3, 4, 5), (3, 5, 6),
             (4, 5, 7)]

    assert set(minimum_spanning_tree(edges)) == set([(1, 3), (2, 4), (2, 3),
                                                     (3, 5)])


def test_minimum_spanning_tree3():
    edges = [(1, 2, 7), (1, 3, 9), (1, 6, 14), (2, 3, 10), (2, 4, 15),
             (3, 4, 11), (3, 6, 2), (4, 5, 6), (5, 6, 9)]

    assert set(minimum_spanning_tree(edges)) == set([(3, 6), (4, 5), (1, 2),
                                                     (1, 3), (5, 6)])


def test_minimum_spanning_tree4():
    edges = [(1, 2, 1), (1, 2, 2), (2, 3, 1), (2, 3, 2), (3, 4, 1), (3, 4, 3)]
    assert set(minimum_spanning_tree(edges)) == set([(1, 2), (2, 3), (3, 4)])


def test_minimum_spanning_tree5():
    edges = [(1, 2, 1), (2, 3, 2), (4, 5, 3), (5, 6, 4)]
    assert set(minimum_spanning_tree(edges)) == set([(1, 2), (2, 3), (4, 5),
                                                     (5, 6)])
