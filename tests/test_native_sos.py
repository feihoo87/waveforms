"""Native SOS processing agrees with SciPy and keeps output limits out of state."""

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

import waveforms as wf
from waveforms._waveform import quantize_samples, sosfilt_samples


def _clip(values, lower, upper):
    if np.iscomplexobj(values):
        return (np.clip(values.real, lower, upper)
                + 1j * np.clip(values.imag, lower, upper))
    return np.clip(values, lower, upper)


@pytest.mark.parametrize("sections", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("complex_signal", [False, True])
def test_scipy_equivalence_and_streaming(sections, complex_signal):
    rng = np.random.default_rng(832)
    sos = butter(2 * sections, .2, output="sos")
    values = rng.normal(size=1031)
    zi = rng.normal(size=(sections, 2)) * .001
    initial = .07
    if complex_signal:
        values = values + 1j * rng.normal(size=len(values))
        zi = zi + 1j * rng.normal(size=zi.shape) * .001
        initial += .04j
    original = values.copy()
    initial_state = zi.copy()
    filtered, expected_state = sosfilt(sos, values - initial, zi=zi)
    expected = _clip(filtered + initial, -.17, .23)
    result, state = sosfilt_samples(values, sos, initial, zi,
                                    lower=-.17, upper=.23)
    np.testing.assert_allclose(result, expected, rtol=3e-13, atol=3e-14)
    np.testing.assert_allclose(state, expected_state, rtol=3e-13, atol=3e-14)
    np.testing.assert_array_equal(zi, initial_state)
    np.testing.assert_array_equal(values, original)

    # Boundaries straddle the internal block size and include a one-sample chunk.
    output = np.empty_like(values)
    state = zi
    start = 0
    for stop in (1, 254, 511, 512, 1024, len(values)):
        part, state = sosfilt_samples(values[start:stop], sos, initial, state,
                                     lower=-.17, upper=.23,
                                     out=output[start:stop])
        assert np.shares_memory(part, output)
        start = stop
    np.testing.assert_array_equal(output, result)
    np.testing.assert_allclose(state, expected_state, rtol=3e-13, atol=3e-14)


@pytest.mark.parametrize("bits", [16, 32])
@pytest.mark.parametrize("sections", [1, 4, 8])
def test_direct_quantization_matches_scipy(bits, sections):
    rng = np.random.default_rng(337)
    values = rng.uniform(-2, 2, 5003)
    sos = butter(sections * 2, .25, output="sos")
    expected = quantize_samples(
        np.clip(sosfilt(sos, values - .1) + .1, -.25, .35), bits, .7)
    output = np.empty_like(expected)
    result, _ = sosfilt_samples(values, sos, .1, bits=bits, full_scale=.7,
                                lower=-.25, upper=.35, out=output)
    assert result is output
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("complex_signal", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_aliasing_and_readonly_inputs(complex_signal, overlap):
    rng = np.random.default_rng(4)
    storage = rng.normal(size=601)
    if complex_signal:
        storage = storage + 1j * rng.normal(size=len(storage))
    values = storage[:-1]
    output = storage[1:] if overlap else values
    sos = butter(6, .3, output="sos")
    sos.flags.writeable = False
    # SciPy 1.13 requires a writable coefficient buffer.
    expected = sosfilt(sos.copy(), values.copy())
    result, _ = sosfilt_samples(values, sos, out=output)
    assert result is output
    np.testing.assert_allclose(result, expected, rtol=3e-13, atol=3e-14)
    values.flags.writeable = False
    expected = sosfilt(sos.copy(), values.copy())
    result, _ = sosfilt_samples(values, sos)
    np.testing.assert_allclose(result, expected, rtol=3e-13, atol=3e-14)


def test_coefficients_may_overlap_output():
    sos = butter(8, .3, output="sos")
    values = np.linspace(-1, 1, sos.size)
    expected = sosfilt(sos.copy(), values)
    result, _ = sosfilt_samples(values, sos, out=sos.reshape(-1))
    np.testing.assert_allclose(result, expected, rtol=3e-13, atol=3e-14)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_empty_input_preserves_state(dtype):
    sos = butter(4, .3, output="sos")
    zi = np.ones((2, 2), dtype=dtype)
    result, state = sosfilt_samples(np.empty(0, dtype=dtype), sos, zi=zi)
    assert result.shape == (0,)
    assert result.dtype == dtype
    np.testing.assert_array_equal(state, zi)


@pytest.mark.parametrize("sos", [np.ones((2, 5)), np.ones((1, 2, 6)),
                                  [[1, 0, 0, 2, 0, 0]]])
def test_invalid_coefficients(sos):
    with pytest.raises(ValueError, match="sos"):
        sosfilt_samples(np.ones(3), sos)


def test_invalid_state_and_output():
    sos = butter(4, .3, output="sos")
    values = np.ones(10)
    with pytest.raises(ValueError, match="zi"):
        sosfilt_samples(values, sos, zi=np.zeros((1, 2)))
    with pytest.raises(ValueError, match="shape"):
        sosfilt_samples(values, sos, out=np.empty(9))
    with pytest.raises(TypeError, match="dtype"):
        sosfilt_samples(values, sos, bits=16, out=np.empty(10))
    with pytest.raises(ValueError, match="contiguous"):
        sosfilt_samples(values, sos, bits=16, out=np.empty(20, dtype=np.int16)[::2])
    with pytest.raises(TypeError, match="real signal"):
        sosfilt_samples(values + 1j, sos, bits=16)
    with pytest.raises(ValueError, match="full_scale"):
        sosfilt_samples(values, sos, bits=16, full_scale=0)
    with pytest.raises(ValueError, match="min and max"):
        sosfilt_samples(values, sos, lower=np.nan)
    with pytest.raises(ValueError, match="non-finite"):
        sosfilt_samples(np.array([np.nan]), sos, bits=16)


def _wave():
    wave = .4 + .5 * wf.cos(6 * np.pi)
    wave.start, wave.stop, wave.sample_rate = 0., 1., 1024
    wave.min, wave.max = -.15, .27
    return wave


@pytest.mark.parametrize("coefficient_dtype", [np.complex128, np.longdouble])
def test_scipy_fallback(coefficient_dtype):
    wave = _wave()
    sos = butter(4, .3, output="sos").astype(coefficient_dtype)
    if coefficient_dtype == np.complex128:
        sos[0, 0] += .1j
    wave.filters = sos, .03
    raw = .4 + .5 * np.cos(6 * np.pi * np.arange(1024) / 1024)
    expected = _clip(sosfilt(sos, raw - .03) + .03, wave.min, wave.max)
    np.testing.assert_allclose(wave.sample(), expected, rtol=3e-13, atol=3e-14)
    np.testing.assert_allclose(np.concatenate(list(wave.sample(chunk_size=73))),
                               expected, rtol=3e-13, atol=3e-14)


def test_pipeline_output_casting_and_single_section_vector():
    wave = _wave()
    wave.filters = butter(2, .3, output="sos")[0], .03
    expected = wave.sample()
    for dtype in (None, np.float32, np.float64):
        for stride in (1, 2):
            output = np.empty(1024 * stride)[::stride]
            result = wave.sample(dtype=dtype, out=output)
            assert result is output
            np.testing.assert_array_equal(result, expected.astype(dtype or np.float64))
    expected_i16 = quantize_samples(expected, 16)
    np.testing.assert_array_equal(wave.sample(dtype=np.int16), expected_i16)
    np.testing.assert_allclose(np.concatenate(list(wave.sample(chunk_size=31))),
                               expected, rtol=3e-13, atol=3e-14)
    with pytest.raises(TypeError, match="dtype"):
        wave.sample(dtype=np.int16, out=np.empty(1024))
    with pytest.raises(ValueError, match="contiguous"):
        wave.sample(out=np.empty(2048, dtype=np.int16)[::2])
