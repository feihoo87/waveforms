"""Output limits belong after calibration/filtering, before DAC conversion."""

import pickle

import numpy as np
import pytest
from scipy.signal import sosfilt

import waveforms as wf
from waveforms._waveform import quantize_samples


def _configure(wave, rate=128):
    wave.start = 0.0
    wave.stop = 1.0
    wave.sample_rate = rate
    return wave


def _signal(kind):
    real = 0.45 + 0.35 * wf.cos(6 * np.pi)
    imag = -0.3 + 0.2 * wf.sin(4 * np.pi)
    if kind == "real":
        return real
    if kind == "complex":
        # Component limits must not truncate a later calibration/filter input.
        real.min, real.max = -0.01, 0.01
        imag.min, imag.max = -0.01, 0.01
        return wf.ComplexWaveform(real, imag)
    real_children = [0.6 * real, 0.4 * real]
    imag_children = [0.3 * imag, 0.7 * imag]
    for child in real_children + imag_children:
        child.min, child.max = -0.01, 0.01
    real_stack = wf.WaveVStack(real_children)
    if kind == "stack":
        return real_stack
    imag_stack = wf.WaveVStack(imag_children)
    real_stack.min, real_stack.max = -0.02, 0.02
    imag_stack.min, imag_stack.max = -0.02, 0.02
    return wf.ComplexWaveVStack(real_stack, imag_stack)


def _clip(values, lower, upper):
    if np.iscomplexobj(values):
        return (np.clip(values.real, lower, upper)
                + 1j * np.clip(values.imag, lower, upper))
    return np.clip(values, lower, upper)


@pytest.mark.parametrize("kind", ["real", "stack", "complex", "complex_stack"])
@pytest.mark.parametrize("rate", [128, 127.5])
@pytest.mark.parametrize("gain", [0.25, 2.0])
def test_output_limits_follow_mapping_and_filter_with_chunk_state(kind, rate, gain):
    wave = _configure(_signal(kind), rate)
    wave.min, wave.max = -0.22, 0.27
    # Stateful filter exercises overshoot and continuing with the unclipped zf.
    sos = np.array([[gain * 0.5, 0., 0., 1., -0.5, 0.]])
    initial = 0.03
    wave.filters = (sos, initial)
    mapping = wf.NonlinearMap.from_samples(
        [-3., 3.], [-6., 6.], method="linear", table_size=2)
    wave.nonlinear = (mapping, mapping) if "complex" in kind else mapping

    positions = np.arange(0., 1., 1 / rate)
    raw = 0.45 + 0.35 * np.cos(6 * np.pi * positions)
    if "complex" in kind:
        raw = raw + 1j * (-0.3 + 0.2 * np.sin(4 * np.pi * positions))
    filtered = sosfilt(sos, 2 * raw - initial) + initial
    expected = _clip(filtered, wave.min, wave.max)

    np.testing.assert_allclose(wave.sample(), expected, rtol=2e-13, atol=2e-14)
    chunks = list(wave.sample(chunk_size=13))
    np.testing.assert_allclose(np.concatenate(chunks), expected,
                               rtol=2e-13, atol=2e-14)
    target = np.empty_like(expected)
    assert wave.sample(out=target) is target
    np.testing.assert_allclose(target, expected, rtol=2e-13, atol=2e-14)
    restored = pickle.loads(pickle.dumps(wave))
    np.testing.assert_array_equal(restored.sample(), wave.sample())

    if "complex" in kind:
        for dtype, bits in ((np.int16, 16), (np.int32, 32)):
            targets = tuple(np.empty(len(expected), dtype=dtype) for _ in range(2))
            actual = wave.sample_iq(dtype=dtype, full_scale=0.7, out=targets)
            for result, output, component in zip(actual, targets,
                                                  (expected.real, expected.imag)):
                assert result is output
                np.testing.assert_array_equal(
                    result, quantize_samples(component, bits, 0.7))
            iq_chunks = list(wave.sample_iq(
                dtype=dtype, full_scale=0.7, chunk_size=13, out=targets))
            for index, component in enumerate((expected.real, expected.imag)):
                np.testing.assert_array_equal(
                    np.concatenate([chunk[index] for chunk in iq_chunks]),
                    quantize_samples(component, bits, 0.7))
    else:
        for dtype, bits in ((np.int16, 16), (np.int32, 32)):
            expected_integer = quantize_samples(expected, bits, 0.7)
            target = np.empty(len(expected), dtype=dtype)
            assert wave.sample(out=target, full_scale=0.7) is target
            np.testing.assert_array_equal(target, expected_integer)
            chunks = wave.sample(out=target, chunk_size=13, full_scale=0.7)
            np.testing.assert_array_equal(np.concatenate(list(chunks)), expected_integer)


@pytest.mark.parametrize("gain,expected", [(2., .2), (.25, .1)])
def test_filter_gain_never_sees_prematurely_limited_input(gain, expected):
    wave = _configure(wf.const(.4))
    wave.min, wave.max = -.2, .2
    wave.filters = (np.array([[gain, 0., 0., 1., 0., 0.]]), 0.)
    np.testing.assert_allclose(wave.sample(), expected)
    np.testing.assert_allclose(np.concatenate(list(wave.sample(chunk_size=7))), expected)


@pytest.mark.parametrize("kind", ["real", "stack", "complex", "complex_stack"])
def test_limits_cannot_hide_nonlinear_domain_errors(kind):
    wave = _configure(_signal(kind))
    wave.min, wave.max = -.1, .1
    mapping = wf.NonlinearMap.from_samples([-.2, .2], [-.2, .2], method="linear")
    wave.nonlinear = (mapping, mapping) if "complex" in kind else mapping
    with pytest.raises(ValueError, match="outside its domain"):
        wave.sample()
    with pytest.raises(ValueError, match="outside its domain"):
        list(wave.sample(chunk_size=7))


@pytest.mark.parametrize("rate", [2_400_000_000, 7_000_000_000])
@pytest.mark.parametrize("spacing", [2e-9, 40e-9])
@pytest.mark.parametrize("many_scales", [False, True])
@pytest.mark.parametrize("limits", [(-.18, .26), (.08, .26), (-.3, -.1)])
def test_stack_native_limits_after_overlap_and_before_quantization(
        rate, spacing, many_scales, limits):
    pulse = wf.gaussian(20e-9)
    pulse.min, pulse.max = -.001, .001
    scales = np.linspace(-.9, .9, 80) if many_scales else np.resize([-.6, .4, .8], 80)
    stack = wf.WaveVStack.from_events(
        [pulse], np.zeros(80, dtype=np.uint32),
        np.arange(80, dtype=np.int64) * wf.time_to_tick(spacing), scales)
    stack.offset = .03
    stack.start, stack.stop = -30e-9, 80 * spacing + 30e-9
    stack.sample_rate = rate
    raw = stack.sample()
    plan = stack._sample_plan_cache[1]
    stack.min, stack.max = limits
    expected = np.clip(raw, *limits)
    np.testing.assert_array_equal(stack.sample(), expected)
    # Limits are execution parameters: changing them must not rebuild the plan.
    assert stack._sample_plan_cache[1] is plan
    np.testing.assert_allclose(np.concatenate(list(stack.sample(chunk_size=137))),
                               expected, rtol=2e-13, atol=2e-14)
    for dtype, bits in ((np.int16, 16), (np.int32, 32)):
        expected_integer = quantize_samples(expected, bits, .7)
        out = np.empty(len(raw), dtype=dtype)
        assert stack.sample(dtype=dtype, out=out, full_scale=.7) is out
        np.testing.assert_array_equal(out, expected_integer)
    stack.min, stack.max = -np.inf, np.inf
    np.testing.assert_array_equal(stack.sample(), raw)


@pytest.mark.parametrize("kind", ["stack", "complex_stack"])
def test_stack_owns_limits_and_preserves_them_in_pickle_and_shift(kind):
    stack = _configure(_signal(kind))
    assert stack.min == -np.inf and stack.max == np.inf
    raw = stack.sample()
    stack.min, stack.max = -.17, .23
    expected = _clip(raw, stack.min, stack.max)
    np.testing.assert_array_equal(stack.sample(), expected)
    shifted = stack >> 0.0
    assert (shifted.min, shifted.max) == (-.17, .23)
    np.testing.assert_array_equal(shifted.sample(), expected)
    restored = pickle.loads(pickle.dumps(stack))
    assert (restored.min, restored.max) == (-.17, .23)
    np.testing.assert_array_equal(restored.sample(), expected)
    # Legacy states omit limits, and the oldest also omit the nonlinear map.
    state = stack.__getstate__()[:-2]
    for legacy in (state, state[:-2] + state[-1:]):
        restored = type(stack).__new__(type(stack))
        restored.__setstate__(legacy)
        assert restored.min == -np.inf and restored.max == np.inf
        np.testing.assert_array_equal(restored.sample(), raw)


@pytest.mark.parametrize("kind", ["stack", "complex_stack"])
def test_stack_direct_evaluation_limits_only_its_own_contribution(kind):
    stack = _signal(kind)
    positions = np.array([.013, .129, .44])
    raw = stack(positions)
    stack.min, stack.max = -.17, .23
    expected = _clip(raw, stack.min, stack.max)
    output = np.full_like(raw, 3)
    assert stack(positions, out=output, accumulate=True) is output
    np.testing.assert_allclose(output, 3 + expected)
    np.testing.assert_allclose(stack(float(positions[0])), expected[0])


def test_empty_stack_limits_apply_to_idle_and_float_output_buffers():
    stack = _configure(wf.WaveVStack())
    stack.min, stack.max = .13, .27
    for dtype in (np.float32, np.float64, np.int16, np.int32):
        target = np.empty(128, dtype=dtype)
        result = stack.sample(dtype=dtype, out=target)
        assert result is target
        if np.issubdtype(dtype, np.integer):
            expected = quantize_samples(np.full(128, .13), np.iinfo(dtype).bits)
        else:
            expected = np.full(128, .13, dtype=dtype)
        np.testing.assert_array_equal(result, expected)
    stack.stop = stack.start
    assert stack.sample(dtype=np.int16).size == 0


@pytest.mark.parametrize("kind", ["real", "stack", "complex", "complex_stack"])
@pytest.mark.parametrize("bounds", [
    (1., -1.), (np.nan, 1.), (-1., np.nan), (np.inf, np.inf), (-np.inf, -np.inf),
])
def test_invalid_output_limits_are_rejected(kind, bounds):
    wave = _configure(_signal(kind))
    wave.min, wave.max = bounds
    with pytest.raises(ValueError, match="min and max"):
        wave.sample()


@pytest.mark.parametrize("rate", [2_400_000_000, 7_000_000_000])
@pytest.mark.parametrize("bits", [16, 32])
def test_integer_limit_rounding_at_half_codes_and_beyond_full_scale(rate, bits):
    wave = wf.WaveVStack([wf.square(20e-9), -wf.square(20e-9) >> 40e-9])
    wave.start, wave.stop, wave.sample_rate = -30e-9, 70e-9, rate
    raw = wave.sample()
    full_scale = .7
    code = full_scale / (2 ** (bits - 1))
    dtype = np.int16 if bits == 16 else np.int32
    for lower, upper in ((-.5 * code, 2.5 * code), (.8, 1.1), (-1.1, -.8),
                         (-np.inf, .2), (-.2, np.inf), (.1, .1)):
        wave.min, wave.max = lower, upper
        np.testing.assert_array_equal(
            wave.sample(dtype=dtype, full_scale=full_scale),
            quantize_samples(np.clip(raw, lower, upper), bits, full_scale))


def test_many_scale_native_limits_span_scratch_blocks_and_partial_placements():
    wave = wf.WaveVStack.from_events(
        [wf.gaussian(300e-9)], np.zeros(80, dtype=np.uint32),
        np.arange(80, dtype=np.int64) * wf.time_to_tick(600e-9),
        np.linspace(-.9, .9, 80))
    wave.start, wave.stop = 47e-9, 79 * 600e-9 + 73e-9
    wave.sample_rate = 2_400_000_000
    raw = wave.sample()
    wave.min, wave.max = -.17, .23
    for dtype, bits in ((np.int16, 16), (np.int32, 32)):
        np.testing.assert_array_equal(
            wave.sample(dtype=dtype, full_scale=.7),
            quantize_samples(np.clip(raw, -.17, .23), bits, .7))
