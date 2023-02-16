import numpy as np

from .fibheap import FibHeap, FibNode


def _generate_standrad_information(key, bayes_matrices):
    ret = []
    for i, k in enumerate(key):
        tmp = sorted([(p, j) for j, p in enumerate(bayes_matrices[i][k])],
                     key=lambda x: abs(x[0]),
                     reverse=True)
        ret.append((tuple(tmp), i))
    return tuple(ret)


def _generate_population(std_info, population):
    state, ret = [0] * len(std_info), population
    for i in range(len(std_info)):
        state[std_info[i][-1]] = std_info[i][0][0][1]
        ret *= std_info[i][0][0][0]
    return ret, tuple(state)


def _add_std_info_to_heap(state, population, heap, heap_nodes, max_n, min_p):
    if state in heap_nodes.keys():
        new_key = heap_nodes[state].key[0] * heap_nodes[state].key[
            -1] + population
        heap.change(heap_nodes[state],
                    (abs(new_key), state, (-1 if new_key < 0 else 1)))
    elif heap.key_number < max_n:
        heap_nodes[state] = FibNode(
            (abs(population), state, (-1 if population < 0 else 1)))
        heap.insert(heap_nodes[state])
    else:
        if abs(population) < max(min_p, heap.min_node.key[0]):
            return False
        trash = heap.extract_min()
        heap_nodes.pop(trash.key[1])
        heap_nodes[state] = FibNode(
            (abs(population), state, (-1 if population < 0 else 1)))
        heap.insert(heap_nodes[state])
    return True


def _update_count(population, std_info, heap, heap_nodes, max_n, min_p):
    queue = FibHeap()

    p, state = _generate_population(std_info, population)
    queue.insert(FibNode((-abs(p), state, std_info, 1 if p > 0 else -1)))

    for _ in range(max_n):
        node = queue.extract_min()
        if node is None or not _add_std_info_to_heap(
                node.key[1], -node.key[0] * node.key[-1], heap, heap_nodes,
                max_n, min_p):
            break

        tmp_info = sorted(list(node.key[2]),
                          reverse=True,
                          key=lambda x:
                          (len(x[0]), abs(x[0][1][0] / x[0][0][0])
                           if len(x[0]) > 1 else 0))

        for i in range(len(std_info)):
            if len(tmp_info[i][0]) == 1:
                break
            new_info = tuple([
                *tmp_info[:i], (tmp_info[i][0][1:], tmp_info[i][1]),
                *tmp_info[i + 1:]
            ])
            p, state = _generate_population(new_info, population)
            if heap.key_number < max_n or p > min(min_p, heap.min_node.key[0]):
                queue.insert(
                    FibNode((-abs(p), state, new_info, (-1 if p < 0 else 1))))
            elif abs(p) < max(min_p, heap.min_node.key[0]):
                break
    heap.consolidate()


def bayesian_correction_automatic_trimming(state,
                                           correction_matrices,
                                           size_lim=None,
                                           eps=None):
    from .fit.readout import count_state

    counts = count_state(state)
    shots = len(state)

    if size_lim is None:
        size_lim = shots * 4
    if eps is None:
        eps = 0.5 / size_lim

    heap_nodes = {}
    heap = FibHeap()

    for key, value in sorted(counts.items(), key=lambda x: x[1],
                             reverse=False):
        std_info = _generate_standrad_information(key, correction_matrices)
        _update_count(value / shots, std_info, heap, heap_nodes, size_lim, eps)

    result = {}
    while heap.key_number:
        node = heap.extract_min()
        if node is None:
            break
        result[node.key[1]] = node.key[0] * node.key[-1]
    return result


def bayesian_correction_in_subspace(state, correction_matrices, subspace):
    """Apply a correction matrix to a state.

    Args:
        state (np.array, dtype=int): The state to be corrected.
        correction_matrices (np.array): A list of correction matrices.
        subspace (np.array, dtype=int): The basis of subspace.

    Returns:
        np.array: The corrected state.

    Examples:
        >>> state = np.random.randint(2, size = (101, 1024, 4))
        >>> PgPe = np.array([[0.1, 0.8], [0.03, 0.91], [0.02, 0.87], [0.05, 0.9]])
        >>> correction_matrices = np.array(
            [np.array([[Pe, Pe - 1], [-Pg, 1 - Pg]]) / (Pe - Pg) for Pg, Pe in PgPe])
        >>> subspace = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0]])
        >>> result = bayesian_correction(state, correction_matrices, subspace)
        >>> result.shape
        (101, 5)
    """
    num_qubits = state.shape[-1]
    site_index = np.arange(num_qubits)

    shape = tuple([*state.shape[:-1], len(subspace)])
    state = state.reshape(-1, num_qubits)

    if len(subspace) < len(state):
        ret = []
        for target_state in subspace:
            A = np.prod(correction_matrices[site_index, target_state, state],
                        axis=-1)
            ret.append(A)
        ret = np.array(ret).T.reshape(shape)
    else:
        ret = []
        for bit_string in state:
            A = np.prod(correction_matrices[site_index, subspace, bit_string],
                        axis=-1)
            ret.append(A)
        ret = np.array(ret).reshape(shape)
    ret = ret.mean(axis=-2)
    return ret


def bayesian_correction(state,
                        correction_matrices,
                        *,
                        subspace=None,
                        size_lim=1024,
                        eps=1e-6):
    """Apply a correction matrix to a state.

    Args:
        state (np.array, dtype=int): The state to be corrected.
        correction_matrices (np.array): A list of correction matrices.
        subspace (np.array, dtype=int): The basis of subspace.
        size_lim (int): The maximum size of the heap.
        eps (float): The minimum probability of the state.

    Returns:
        np.array: The corrected state.

    Examples:
        >>> state = np.random.randint(2, size = (101, 1024, 4))
        >>> PgPe = np.array([[0.1, 0.8], [0.03, 0.91], [0.02, 0.87], [0.05, 0.9]])
        >>> correction_matrices = np.array(
            [np.array([[Pe, Pe - 1], [-Pg, 1 - Pg]]) / (Pe - Pg) for Pg, Pe in PgPe])
        >>> result = bayesian_correction(state, correction_matrices)
        >>> result.shape
        (101, 1024)
    """
    if subspace is None:
        state = np.array(state)
        if state.ndim == 2:
            return bayesian_correction_automatic_trimming(
                state, correction_matrices, size_lim, eps)
        shape = state.shape[:-2]
        state = state.reshape(-1, state.shape[-2], state.shape[-1])
        ret = []
        for s in state:
            ret.append(
                bayesian_correction_automatic_trimming(s, correction_matrices,
                                                       size_lim, eps))
        return np.array(ret).reshape(shape)
    else:
        return bayesian_correction_in_subspace(state, correction_matrices,
                                               subspace)
