from waveforms.math.interval import Interval


def test_empty():
    assert Interval().empty()
    assert not Interval("[1, 2]").empty()


def test_full():
    assert not Interval().full()
    assert Interval("(-inf, inf)").full()


def test_contains():
    assert 1.5 in Interval("[1, 2]")
    assert 1 not in Interval("(1, 2)")
    assert 2 not in Interval("[1, 2)")
    assert 1 not in Interval()


def test_subset():
    assert Interval("[1, 2]").is_subset_of(Interval("[1, 3]"))
    assert not Interval("[1, 3]").is_subset_of(Interval("[1, 2]"))
    assert Interval("[1, 2] U [3, 4]").is_subset_of(Interval("[0, 5]"))
    assert not Interval("[0, 5]").is_subset_of(Interval("[1, 2] U [3, 4]"))


def test_inverse():
    assert ~Interval("[1, 2]") == Interval("(-inf, 1) U (2, inf)")
    assert ~Interval("[1, 2) U [3, 4]") == Interval(
        "(-inf, 1) U [2, 3) U (4, inf)")


def test_and():
    assert (Interval("[1, 2]") & Interval("[2, 3]")) == Interval("[2, 2]")
    assert (Interval("[1, 2)") & Interval("[2, 3]")) == Interval()
    assert (Interval("[1, 3]") & Interval("[2, 4]")) == Interval("[2, 3]")


def test_or():
    assert (Interval("[1, 2]") | Interval("[2, 3]")) == Interval("[1, 3]")
    assert (Interval("[1, 2)") | Interval("[2, 3]")) == Interval("[1, 3]")
    assert (Interval("[1, 3]") | Interval("[2, 4]")) == Interval("[1, 4]")
