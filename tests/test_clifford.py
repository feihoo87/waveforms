from waveforms.quantum.clifford import cliffordOrder


def test_cliffordOrder():
    assert cliffordOrder(0) == 1
    assert cliffordOrder(1) == 24
    assert cliffordOrder(2) == 11520
    assert cliffordOrder(3) == 92897280
