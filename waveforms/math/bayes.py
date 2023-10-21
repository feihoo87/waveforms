import itertools
import math

import numpy as np
from scipy.linalg import logm

from .fibheap import FibHeap, FibNode
from .paulis import string_to_matrices


def input_state_set(n):
    """
    Input states for CTMP calibration.

    Args:
        n: number of qubits

    Returns:
        a (M x n) array, each row is a state
        M is the smallest integer such that
        comb(M, M // 2) >= n
    """
    r = 2
    while True:
        if math.comb(2 * r, r) >= n:
            break
        r += 1
    ret = np.zeros((2 * r, n), dtype=np.int8)

    for i, x in enumerate(itertools.combinations(range(2 * r), r=r)):
        if i >= n:
            break
        ret[x, i] = 1
    return ret


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

    See also:
        https://doi.org/10.1103/PhysRevA.103.042605

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


def get_error_rates(matrices, N):
    rates1 = {}
    rates2 = {}

    for i, j in itertools.combinations(range(N), r=2):
        if (i, j) in matrices:
            G = logm(matrices[(i, j)])
        elif (j, i) in matrices:
            G = logm(matrices[(j, i)])
            G = np.array([[G[0, 0], G[0, 2], G[0, 1], G[0, 3]],
                          [G[2, 0], G[2, 2], G[2, 1], G[2, 3]],
                          [G[1, 0], G[1, 2], G[1, 1], G[1, 3]],
                          [G[3, 0], G[3, 2], G[3, 1], G[3, 3]]])
        else:
            continue
        G = np.clip(G, 0, np.inf)
        rates2[(i, j)] = np.diag(G[::-1, :])
        if i not in rates1:
            rates1[i] = np.array([0.0, 0.0])
        rates1[i] += np.array([G[2, 0] + G[3, 1], G[0, 2] + G[1, 3]
                               ]) / (2 * (N - 1))

    gamma = np.max(list(rates1.values()), axis=-1).sum() + np.max(
        list(rates2.values()), axis=-1).sum()
    return gamma, rates1, rates2


def _sample_x(x, gamma, rates1, rates2=None):
    a = 0
    seed = np.random.random()
    for q1, r in rates1.items():
        a += r[x[q1]] / gamma
        if a > seed:
            x = x.copy()
            x[q1] ^= 1
            return x
    if rates2 is not None:
        for (q1, q2), r in rates2.items():
            a += r[2 * x[q1] + x[q2]] / gamma
            if a > seed:
                x = x.copy()
                x[q1] ^= 1
                x[q2] ^= 1
                return x
    return x


def exception(state,
              e_ops: np.ndarray | list[str],
              correction_matrices: np.ndarray | None = None,
              gamma=None,
              rates1=None,
              rates2=None):
    """Calculate the exceptions of the operators.

    Consider a simple case when A is a tensor product of 2 x 2 stochastic matrices.
    and observable operator has a tensor product form.

    Args:
        state (np.array, dtype=int): The bit string of the state. The shape
            should be (..., shots, num_qubits).
        e_ops (np.array): A list of operators. Each operator should be diagonal.
        correction_matrices (np.array): A list of correction matrices.

    See also:
        https://doi.org/10.1103/PhysRevA.103.042605

    Returns:
        np.array: The exceptions of the operators.

    Examples:
        >>> state = np.random.randint(2, size = (101, 1024, 4))
        >>> errors = [[0.05, 0.1], [0.02, 0.03], [0.01, 0.08], [0.02, 0.03]]
        >>> correction_matrices = np.array([
            np.linalg.inv(np.array([[1 - eps, eta], [eps, 1 - eta]]))
            for eps, eta in errors
        ])
        >>> ops = ['0III', '0II1', 'ZIZZ']
        >>> result = exception(state, ops, correction_matrices)
        >>> result.shape
        (101, 3)
    """

    *datashape, shots, num_qubits = state.shape
    site_index = np.arange(num_qubits)

    if e_ops and isinstance(e_ops[0], str):
        e_ops = [string_to_matrices(s, diag=True) for s in e_ops]
    e_ops = np.asarray(e_ops)

    *n_ops, num_qubits_, _, _ = e_ops.shape
    assert num_qubits == num_qubits_

    if gamma is not None and rates1 is not None:
        if rates2 is None:
            pass
        else:
            alpha = np.random.poisson(gamma, (*datashape, shots))
            state = state.reshape(-1, num_qubits)
            for i, (n, s) in enumerate(zip(alpha.reshape(-1), state)):
                for _ in range(n):
                    s = _sample_x(s, gamma, rates1, rates2)
                state[i] = s
            state = state.reshape(*datashape, shots, num_qubits)
            sign = (-1)**(alpha & 1)
            return np.moveaxis(
                (sign * e_ops[..., site_index, state].prod(axis=-1)).mean(axis=-1),
                0, -1)

    if correction_matrices is None:
        M = e_ops
    else:
        correction_matrices = np.asarray(correction_matrices)
        num_qubits_, _, _ = correction_matrices.shape
        assert num_qubits == num_qubits_
        M = e_ops @ correction_matrices

    return M[..., site_index, :,
             state.reshape(-1, num_qubits).astype(np.int8)].sum(axis=-1).prod(
                 axis=1).reshape(*datashape, shots,
                                 *n_ops).mean(axis=len(datashape))


def measure(op: str):
    """
    Measure the operator.

    Covnert the operator to the form of 'I', 'Z' by
    adding '-X/2' and 'Y/2' gates.
    return the covnerted operator and the circuit.

    Args:
        op: the operator to be measured
        the operator is a string of 'I', 'X', 'Y', 'Z'

    Returns:
        operator, circuit
        the operator is a string of 'I', 'Z'
        the circuit is a list of (gate, qubit_index)

    Examples:
        >>> measure('X')
        ('Z', [('-Y/2', 0)])
        >>> measure('Y')
        ('Z', [('X/2', 0)])
        >>> measure('Z')
        ('Z', [])
        >>> measure('I')
        ('I', [])
    """
    e_op = []
    circ = []

    sign = 1
    op = op.strip()
    if op.startswith('+'):
        op = op[1:]
    if op.startswith('-'):
        sign = -1
        op = op[1:]
    if op.startswith('i'):
        op = op[1:]
        sign *= 1j
    if sign == -1:
        e_op.append('-')
    elif sign == 1j:
        e_op.append('i')
    elif sign == -1j:
        e_op.append('-i')

    for i, c in enumerate(op):
        if c == 'X':
            e_op.append('Z')
            circ.append(('-Y/2', i))
        elif c == 'Y':
            e_op.append('Z')
            circ.append(('X/2', i))
        elif c in ['I', 'Z', '0', '1']:
            e_op.append(c)
        else:
            raise ValueError(f"Unknown operator {c}")
    return ''.join(e_op), circ
