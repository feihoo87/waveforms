from waveforms.math.group.permutation_group import Cycles, permute


def test_init():
    c = Cycles((1, 2, 3), (4, 5))
    assert c._cycles == ((1, 2, 3), (4, 5))
    assert c._support == (1, 2, 3, 4, 5)
    c2 = Cycles(1, 2, 3)(4, 5)
    assert c == c2


def test_permute():
    c = Cycles((1, 2, 3), (4, 5))
    assert c.replace(1) == 2
    assert c.replace(2) == 3
    assert c.replace(3) == 1
    assert c.replace(4) == 5
    assert c.replace(5) == 4

    assert permute("abcdef", c) == "adbcfe"
    assert permute("abcdef", c * c) == "acdbef"


def test_inv():
    c = Cycles((1, 2, 3), (4, 5))
    assert c.inv() == Cycles((1, 3, 2), (4, 5))
