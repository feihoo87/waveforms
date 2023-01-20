from typing import Hashable


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
    groups = []
    ret = []

    for a, b in edges:

        group_a_index, group_b_index = None, None
        for i, group in enumerate(groups):
            if a in group and b in group:
                break
            elif a in group:
                group_a_index = i
            elif b in group:
                group_b_index = i
            else:
                continue
            if group_a_index is not None and group_b_index is not None:
                group_a_index, group_b_index = sorted(
                    [group_a_index, group_b_index])
                group_b = groups.pop(group_b_index)
                group_a = groups.pop(group_a_index)
                groups.append(group_a | group_b)
                x = ret.pop(group_b_index)
                y = ret.pop(group_a_index)
                ret.append([*x, *y, (a, b)])
                break
        else:
            if group_a_index is None and group_b_index is None:
                groups.append({a, b})
                ret.append([(a, b)])
            elif group_a_index is None:
                groups[group_b_index].add(a)
                ret.append([(a, b)])
            elif group_b_index is None:
                groups[group_a_index].add(b)
                ret.append([(a, b)])

    return ret


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
