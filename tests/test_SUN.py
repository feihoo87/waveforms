import numpy as np

from waveforms.math.group import SU


def test_SU():
    for N in range(2, 6):
        for i in range(N**2):
            mat = SU(N)[i]
            assert mat.shape == (N, N)
            if i == 0:
                assert np.allclose(mat, np.eye(N))
            else:
                assert abs(np.trace(mat)) < 1e-9
            assert np.allclose(mat, mat.T.conj())
