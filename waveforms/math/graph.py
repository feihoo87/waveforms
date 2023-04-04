from typing import Hashable


class _Graph():

    def __init__(self):
        self._edges = []
        self._nodes = set()

    def add_edge(self, a, b):
        self._edges.append((a, b))
        self._nodes.add(a)
        self._nodes.add(b)

    def __or__(self, other):
        self._edges.extend(other._edges)
        self._nodes.update(other._nodes)
        return self

    def __contains__(self, item):
        return item in self._nodes


def graphs(
    edges: list[tuple[Hashable, Hashable]]
) -> list[list[tuple[Hashable, Hashable]]]:
    """
    graphs

    Args:
        edges: list of edges

    Returns:
        list of graphs
    """
    graphs: list[_Graph] = []
    for a, b in edges:
        extended = []
        for i, g in enumerate(graphs):
            if a in g and b in g:
                g.add_edge(a, b)
                break
            elif a in g or b in g:
                g.add_edge(a, b)
                extended.append(i)
        else:
            if len(extended) == 0:
                g = _Graph()
                g.add_edge(a, b)
                graphs.append(g)
        if len(extended) == 2:
            graphs[extended[0]] = graphs[extended[0]] | graphs[extended[1]]
            del graphs[extended[1]]

    return [[tuple(e) for e in g._edges] for g in graphs]


def minimum_spanning_tree(
    edges: list[tuple[Hashable, Hashable]]
    | list[tuple[Hashable, Hashable, float]]
) -> list[tuple[Hashable, Hashable]]:
    """
    minimum spanning tree

    Args:
        edges: list of edges

    Returns:
        list of edges in minimum spanning tree
    """
    if not edges:
        return []

    if isinstance(edges[0], tuple) and len(edges[0]) == 2:
        edges = [(u, v, 1) for u, v in edges]

    nodes = set()
    for u, v, _ in edges:
        nodes.add(u)
        nodes.add(v)

    nodes = list(nodes)
    nodes_map = {n: i for i, n in enumerate(nodes)}

    edges = [(nodes_map[u], nodes_map[v], w) for u, v, w in edges]
    edges.sort(key=lambda e: e[2])

    tree = []
    parent = list(range(len(nodes)))
    rank = [0] * len(nodes)

    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    def union(u, v):
        u = find(u)
        v = find(v)
        if u == v:
            return
        if rank[u] < rank[v]:
            u, v = v, u
        parent[v] = u
        if rank[u] == rank[v]:
            rank[u] += 1

    for u, v, w in edges:
        if find(u) != find(v):
            union(u, v)
            tree.append((nodes[u], nodes[v]))

    return tree
