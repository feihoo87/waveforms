import itertools
import random

from waveforms.math.bayes import *

num_qubits = 6
shots = 4096

input_states = np.array(
    [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0], [1, 0, 0, 0, 1, 1],
     [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0]],
    dtype=np.int8)

np.random.seed(0)
output_states = [input_states for _ in range(shots - 100)]
output_states.extend([
    np.random.randint(2, size=(input_states.shape[0], num_qubits))
    for _ in range(100)
])
output_states = np.moveaxis(np.array(output_states), 0, -2)


def test_input_state_set():
    input_states = input_state_set(num_qubits)
    assert input_states.shape[1] == num_qubits
    for i, j in itertools.combinations(range(num_qubits), 2):
        state_set = set()
        for x in input_states[:, [i, j]]:
            state_set.add(tuple(x))
        assert len(state_set) == 4
        assert (0, 0) in state_set
        assert (0, 1) in state_set
        assert (1, 0) in state_set
        assert (1, 1) in state_set


def test_extract_matrices():
    matrices = extract_matrices(input_states, output_states)
    assert len(matrices) == math.comb(num_qubits, 2)
    for i, j in itertools.combinations(range(num_qubits), 2):
        assert matrices[(i, j)].shape == (2**2, 2**2)
        assert np.allclose(matrices[(i, j)], np.eye(2**2), atol=0.05)


def test_get_error_rates():
    matrices = extract_matrices(input_states, output_states)
    gamma, rates1, rates2 = get_error_rates(matrices, num_qubits)
    assert gamma >= 0
    assert len(rates1) == num_qubits
    assert len(rates2) == math.comb(num_qubits, 2)
    assert np.allclose(list(rates1.values()), 0, atol=0.05)
    assert np.allclose(list(rates2.values()), 0, atol=0.05)


def test_exception_shape():
    shots = 1024
    for datashape in [(), (13, ), (7, 11)]:
        states = np.random.randint(2, size=(*datashape, shots, num_qubits))
        e_ops = [
            ''.join(random.choices("IZ10", k=num_qubits)) for _ in range(5)
        ]

        correction_matrices = [np.eye(2) for _ in range(num_qubits)]

        ret = exception(states,
                        e_ops=e_ops,
                        correction_matrices=correction_matrices)

        assert ret.shape == (*datashape, len(e_ops))

        matrices = extract_matrices(input_states, output_states)
        gamma, rates1, rates2 = get_error_rates(matrices, num_qubits)

        ret = exception(states,
                        e_ops=e_ops,
                        gamma=gamma,
                        rates1=rates1,
                        rates2=rates2)

        assert ret.shape == (*datashape, len(e_ops))


def test_measure():
    assert measure('X') == ('Z', [('-Y/2', 0)])
    assert measure('Y') == ('Z', [('X/2', 0)])
    assert measure('Z') == ('Z', [])
    assert measure('I') == ('I', [])
    assert measure('0') == ('0', [])
    assert measure('1') == ('1', [])
    assert measure('IX') == ('IZ', [('-Y/2', 1)])
    assert measure('XI') == ('ZI', [('-Y/2', 0)])
    assert measure('XX') == ('ZZ', [('-Y/2', 0), ('-Y/2', 1)])
    assert measure('-XY') == ('-ZZ', [('-Y/2', 0), ('X/2', 1)])
