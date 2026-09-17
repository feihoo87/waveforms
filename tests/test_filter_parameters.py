"""Clock-independent predistortion parameters and first-sample prehistory."""

from collections import defaultdict
import copy
import pickle

import numpy as np
import pytest
from scipy.signal import sosfilt

import waveforms as wf
import waveforms.waveform as impl
from waveforms._waveform import quantize_samples


KINDS = ("real", "stack", "complex", "complex_stack")
STAGES = {50e-9: .12, 200e-9: -.04, 1e-6: .08}


def _wave(kind, constant=False):
    real = wf.const(.375) if constant else .125 + .25 * wf.cos(2 * np.pi * 7e6)
    imag = wf.const(-.125) if constant else -.125 + .125 * wf.sin(2 * np.pi * 11e6)
    if "stack" in kind:
        real = wf.WaveVStack([.5 * real, .5 * real])
        imag = wf.WaveVStack([.25 * imag, .75 * imag])
    if kind == "complex":
        real = wf.ComplexWaveform(real, imag)
    elif kind == "complex_stack":
        real = wf.ComplexWaveVStack(real, imag)
    real.start, real.stop, real.sample_rate = 0., 256e-9, 1_000_000_000
    return real


def _reference(raw, stages, rate):
    amp, tau = wf.exp_decay_filter_from_cascade(
        [(amp, tau) for tau, amp in sorted(stages.items()) if amp != 0])
    sos = wf.exp_decay_filter(amp, tau, rate, inv=True, output="sos")
    return sosfilt(sos, raw - raw[0]) + raw[0]


@pytest.mark.parametrize("kind", KINDS)
def test_default_mapping_assignment_and_independent_copies(kind):
    wave = _wave(kind)
    other = _wave(kind)
    assert isinstance(wave.filters, defaultdict)
    assert wave.filters.default_factory is float
    assert wave.filters[50e-9] == 0
    wave.filters[50e-9] += .12
    assert other.filters == {}
    params = defaultdict(lambda: 99, STAGES)
    wave.filters = params
    params[50e-9] = .9
    assert wave.filters == STAGES
    assert wave.filters.default_factory is float
    for duplicate in (copy.copy(wave), copy.deepcopy(wave),
                      pickle.loads(pickle.dumps(wave))):
        assert isinstance(duplicate.filters, defaultdict)
        assert duplicate.filters == STAGES
        assert duplicate.filters is not wave.filters
        duplicate.filters[50e-9] += .05
        assert wave.filters == STAGES
    wave.filters = None
    assert isinstance(wave.filters, defaultdict)
    assert wave.filters == {}


@pytest.mark.parametrize("kind", ["stack", "complex_stack"])
def test_delay_calibration_preserves_but_does_not_alias_filters(kind):
    wave = _wave(kind)
    wave.filters = STAGES
    restored = pickle.loads(pickle.dumps(wave))
    shifted = restored >> 2e-9
    assert shifted.filters == STAGES
    raw = shifted.sample(filters={})
    np.testing.assert_allclose(shifted.sample(),
                               _reference(raw, STAGES, shifted.sample_rate),
                               rtol=3e-12, atol=3e-13)
    shifted.filters[50e-9] += .01
    assert restored.filters == STAGES


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("rate", [500_000_000, 1_200_000_000, 2_400_000_000,
                                   7_000_000_000, 1_234_567_890.5])
def test_rate_override_inverse_filter_and_chunked_first_sample(kind, rate):
    wave = _wave(kind)
    wave.filters = STAGES
    raw = wave.sample(rate, filters={})
    expected = _reference(raw, STAGES, rate)
    actual = wave.sample(rate)
    assert actual[0] == raw[0]
    assert not np.allclose(actual, raw)
    # Native real/imag recurrences and SciPy complex arithmetic round slightly
    # differently for poles this close to one (small accumulated roundoff).
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=2e-12)
    for chunk_size in (1, 31, 256, len(raw) + 1):
        target = np.empty_like(expected)
        chunks = list(wave.sample(rate, chunk_size=chunk_size, out=target))
        np.testing.assert_allclose(np.concatenate(chunks), expected,
                                   rtol=3e-12, atol=2e-12)
        np.testing.assert_allclose(target, expected, rtol=3e-12, atol=2e-12)
    # Changing the device clock must not modify the serialized calibration.
    assert wave.sample_rate == 1_000_000_000
    assert wave.filters == STAGES


@pytest.mark.parametrize("kind", KINDS)
def test_constant_prehistory_has_no_startup_transient_after_mapping(kind):
    wave = _wave(kind, constant=True)
    wave.filters = STAGES
    mapping = wf.NonlinearMap.from_samples(
        [-1., 1.], [-.5, 1.5], method="linear", table_size=2)
    wave.nonlinear = (mapping, mapping) if "complex" in kind else mapping
    expected = .875 + .375j if "complex" in kind else .875
    np.testing.assert_array_equal(wave.sample(), np.full(256, expected))
    chunks = np.concatenate(list(wave.sample(chunk_size=7)))
    np.testing.assert_array_equal(chunks, np.full(256, expected))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", [np.int16, np.int32])
def test_initial_level_precedes_clipping_and_quantization(kind, dtype):
    wave = _wave(kind)
    wave.filters = STAGES
    mapping = wf.NonlinearMap.from_samples(
        [-1., 1.], [-.75, 1.25], method="linear", table_size=2)
    wave.nonlinear = (mapping, mapping) if "complex" in kind else mapping
    raw = wave.sample(filters={})
    filtered = _reference(raw, STAGES, wave.sample_rate)
    wave.min, wave.max = -.1, .2
    bits = np.dtype(dtype).itemsize * 8
    if "complex" in kind:
        expected = tuple(quantize_samples(np.clip(v, -.1, .2), bits)
                         for v in (filtered.real, filtered.imag))
        actual = wave.sample_iq(dtype=dtype)
        for component, reference in zip(actual, expected):
            np.testing.assert_array_equal(component, reference)
        chunks = list(wave.sample_iq(dtype=dtype, chunk_size=1))
        for index in (0, 1):
            np.testing.assert_array_equal(
                np.concatenate([chunk[index] for chunk in chunks]), expected[index])
    else:
        expected = quantize_samples(np.clip(filtered, -.1, .2), bits)
        np.testing.assert_array_equal(wave.sample(dtype=dtype), expected)
        np.testing.assert_array_equal(
            np.concatenate(list(wave.sample(dtype=dtype, chunk_size=1))), expected)


@pytest.mark.parametrize("kind", KINDS)
def test_empty_and_single_sample(kind):
    wave = _wave(kind)
    wave.filters = STAGES
    wave.stop = wave.start
    assert wave.sample().size == 0
    assert list(wave.sample(chunk_size=1)) == []
    wave.stop = 1e-9
    expected = wave.sample(filters={})
    assert len(expected) == 1
    np.testing.assert_array_equal(wave.sample(), expected)
    np.testing.assert_array_equal(np.concatenate(list(wave.sample(chunk_size=1))), expected)


def test_cache_tracks_values_and_rate_but_not_initial_or_filter_state():
    impl._exp_decay_sos.cache_clear()
    wave = _wave("real")
    wave.filters = STAGES
    before = pickle.dumps(wave)
    initial_result = wave.sample()
    assert impl._exp_decay_sos.cache_info().misses == 1
    assert pickle.dumps(wave) == before
    np.testing.assert_array_equal(wave.sample(), initial_result)
    wave.filters = dict(reversed(list(STAGES.items())))
    wave.filters[2e-6]  # Reading a missing tau must not invalidate the cache.
    np.testing.assert_array_equal(wave.sample(), initial_result)
    assert impl._exp_decay_sos.cache_info().misses == 1
    # A different signal shares coefficients, not its first point or state.
    other = _wave("complex")
    other.filters = STAGES
    other.sample()
    assert impl._exp_decay_sos.cache_info().misses == 1
    wave.filters[50e-9] += .1
    assert not np.allclose(wave.sample(), initial_result)
    assert impl._exp_decay_sos.cache_info().misses == 2
    wave.sample(2_400_000_000)
    assert impl._exp_decay_sos.cache_info().misses == 3


def test_explicit_mapping_override_and_snapshot_during_stream():
    wave = _wave("real")
    wave.filters = STAGES
    raw = wave.sample(filters={})
    override = {75e-9: .2}
    expected = _reference(raw, override, wave.sample_rate)
    stream = wave.sample(filters=override, chunk_size=31)
    chunks = [next(stream)]
    override[75e-9] = .4
    wave.filters[50e-9] = .5
    chunks.extend(stream)
    np.testing.assert_allclose(np.concatenate(chunks), expected,
                               rtol=3e-12, atol=3e-13)
    np.testing.assert_array_equal(wave.sample(filters={75e-9: 0}), raw)
    np.testing.assert_array_equal(wave.sample(filters={}), raw)


@pytest.mark.parametrize("params,message", [
    ({0.: .1}, "positive"), ({-1.: .1}, "positive"),
    ({np.inf: .1}, "finite"), ({np.nan: .1}, "finite"),
    ({1e-6: np.nan}, "finite"), ({1e-6: np.inf}, "finite"),
])
def test_invalid_active_parameters(params, message):
    wave = _wave("real")
    wave.filters = params
    with pytest.raises(ValueError, match=message):
        wave.sample()
    with pytest.raises(ValueError, match=message):
        list(wave.sample(chunk_size=7))


def test_old_sos_tuple_is_not_a_parameter_mapping():
    wave = _wave("real")
    old = (np.array([[1., 0., 0., 1., 0., 0.]]), 0.)
    with pytest.raises(TypeError, match="mapping"):
        wave.filters = old
    with pytest.raises(TypeError, match="mapping"):
        wave.sample(filters=old)
