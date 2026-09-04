import pickle

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, sosfilt, tf2sos

import waveforms as wf
from waveforms._waveform import quantize_samples


def test_linear_map_binary_roundtrip_clip_and_output():
    x = np.linspace(-2.0, 2.0, 9)
    y = 0.5 + 1.75 * x
    mapping = wf.NonlinearMap.from_samples(
        x, y, method="linear", table_size=9,
    )
    points = np.linspace(-2.0, 2.0, 1001)
    expected = 0.5 + 1.75 * points
    assert np.allclose(mapping(points), expected, rtol=0, atol=2e-15)

    target = np.empty_like(points)
    assert mapping(points, out=target) is target
    assert np.array_equal(target, mapping(points))
    assert mapping(0.25) == pytest.approx(0.9375)
    assert mapping.method == "linear"
    assert mapping.dtype == np.dtype(np.float64)
    assert mapping.extrapolate == "error"
    assert mapping.point_count == 9
    assert mapping.domain == (-2.0, 2.0)

    data = mapping.to_bytes()
    assert data[:4] == b"NLM1"
    restored = wf.NonlinearMap.from_bytes(data)
    assert restored.to_bytes() is data
    assert restored == mapping
    assert hash(restored) == hash(mapping)
    assert pickle.loads(pickle.dumps(mapping)) == mapping
    assert np.array_equal(
        mapping._apply_quantized(points, 16),
        quantize_samples(mapping(points), 16),
    )

    malformed = bytearray(data)
    malformed[9] = 1
    with pytest.raises(ValueError, match="NLM1"):
        wf.NonlinearMap.from_bytes(malformed)
    with pytest.raises(ValueError, match="NLM1"):
        wf.NonlinearMap.from_bytes(data[:-1])

    with pytest.raises(ValueError, match="outside its domain"):
        mapping(np.array([-2.1, 0.0]))
    clipped = wf.NonlinearMap.from_samples(
        x, y, method="linear", table_size=9, extrapolate="clip",
    )
    assert np.array_equal(clipped([-3.0, 3.0]), [y[0], y[-1]])


def test_monotone_cubic_accuracy_storage_and_centering():
    x = np.array([4.0, 4.15, 4.4, 4.9, 5.6, 6.0])
    y = np.array([0.31, 0.27, 0.18, 0.02, -0.11, -0.16])
    reference = 4.9
    mapping = wf.NonlinearMap.from_samples(
        x, y, table_size=1025, reference=reference,
    )
    relative = np.linspace(mapping.domain[0], mapping.domain[1], 10001)
    reference_curve = PchipInterpolator(x, y)
    expected = reference_curve(reference + relative) - reference_curve(reference)
    assert np.max(np.abs(mapping(relative) - expected)) < 2e-7
    assert mapping(0.0) == pytest.approx(0.0, abs=2e-15)
    assert mapping.method == "monotone_cubic"
    assert mapping.reference_input == reference
    assert mapping.reference_output == pytest.approx(
        reference_curve(reference), abs=2e-7)
    assert np.all(np.diff(mapping(relative)) <= 0)

    compact = wf.NonlinearMap.from_samples(
        x, y, table_size=1025, reference=reference, dtype=np.float32,
    )
    assert compact.dtype == np.dtype(np.float32)
    assert compact(0.0) == pytest.approx(0.0, abs=2e-7)
    assert len(compact.to_bytes()) == 48 + 16 * (compact.point_count - 1)
    assert len(mapping.to_bytes()) == 48 + 32 * (mapping.point_count - 1)
    assert np.max(np.abs(compact(relative) - expected)) < 3e-7


@pytest.mark.parametrize("x,y,message", [
    ([0, 0, 1], [0, 1, 2], "increase strictly"),
    ([0, 1, 2], [0, 2, 1], "one inverse branch"),
    ([0, 1], [0, np.nan], "finite"),
])
def test_map_validation(x, y, message):
    with pytest.raises(ValueError, match=message):
        wf.NonlinearMap.from_samples(x, y)
    with pytest.raises(ValueError, match="reference"):
        wf.NonlinearMap.from_samples([0, 1], [0, 1], reference=2)
    with pytest.raises(TypeError, match="float32 or float64"):
        wf.NonlinearMap.from_samples([0, 1], [0, 1], dtype=np.int16)
    with pytest.raises(ValueError, match="method"):
        wf.NonlinearMap.from_samples([0, 1], [0, 1], method="bezier")
    with pytest.raises(ValueError, match="NLM1"):
        wf.NonlinearMap.from_bytes(b"not a map")


def test_error_controlled_table_compilation():
    x = np.array([0.0, 0.13, 0.41, 1.0])
    y = np.array([0.0, 0.2, 0.75, 1.0])
    mapping = wf.NonlinearMap.from_samples(
        x, y, method="linear", table_size=5,
        max_error=1e-3, max_table_size=4097,
    )
    assert mapping.point_count > 5
    points = np.linspace(0.0, 1.0, 100_001)
    assert np.max(np.abs(mapping(points) - np.interp(points, x, y))) < 1e-3

    with pytest.raises(ValueError, match="requires more than"):
        wf.NonlinearMap.from_samples(
            x, y, method="linear", table_size=5,
            max_error=1e-6, max_table_size=9,
        )
    with pytest.raises(ValueError, match="max_error"):
        wf.NonlinearMap.from_samples(x, y, max_error=0)


def test_waveform_sampling_order_chunking_quantization_and_pickle():
    sample_rate = 1024
    mapping = wf.NonlinearMap.from_samples(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        [0.0, 0.0625, 0.25, 0.5625, 1.0],
        table_size=257,
    )
    b, a = butter(3, 40.0, "lowpass", fs=sample_rate)
    sos = tf2sos(b, a)
    waveform = wf.t()
    waveform.start = 0.0
    waveform.stop = 1.0
    waveform.sample_rate = sample_rate
    waveform.nonlinear = mapping
    waveform.filters = (sos, 0.0)

    raw = np.arange(sample_rate, dtype=np.float64) / sample_rate
    expected = sosfilt(sos, mapping(raw))
    actual = waveform.sample()
    assert np.allclose(actual, expected, rtol=3e-14, atol=3e-14)
    chunks = np.concatenate(list(waveform.sample(chunk_size=73)))
    assert np.allclose(chunks, expected, rtol=3e-14, atol=3e-14)

    expected_int16 = quantize_samples(expected, 16)
    output = np.empty(sample_rate, dtype=np.int16)
    assert waveform.sample(dtype=np.int16, out=output) is output
    assert np.array_equal(output, expected_int16)

    restored = pickle.loads(pickle.dumps(waveform))
    assert restored.nonlinear == mapping
    assert np.array_equal(restored.sample(), actual)


def test_stack_mapping_happens_after_event_accumulation():
    pulse = 0.4 * wf.square(1.0)
    stack = wf.WaveVStack([pulse, pulse])
    stack.start = -0.25
    stack.stop = 0.25
    stack.sample_rate = 1000
    stack.nonlinear = wf.NonlinearMap.from_samples(
        [0.0, 0.4, 0.8], [0.0, 0.16, 0.64], table_size=257,
    )
    samples = stack.sample()
    assert np.allclose(samples, 0.64, atol=2e-14)
    assert not np.allclose(samples, 0.32)


def test_complex_waveform_uses_explicit_component_maps():
    real_map = wf.NonlinearMap.from_samples(
        [0.0, 0.5, 1.0], [0.0, 0.25, 1.0], table_size=257,
    )
    imag_map = wf.NonlinearMap.from_samples(
        [0.0, 0.5, 1.0], [0.0, 1.0, 2.0],
        method="linear", table_size=257,
    )
    waveform = wf.ComplexWaveform(0.5, 0.25)
    waveform.start = 0.0
    waveform.stop = 1.0
    waveform.sample_rate = 16
    waveform.nonlinear = (real_map, imag_map)
    expected = real_map(0.5) + 1j * imag_map(0.25)
    assert np.allclose(waveform.sample(), expected)
    i, q = waveform.sample_iq(dtype=np.int16)
    assert np.array_equal(i, quantize_samples(
        np.full(16, expected.real), 16))
    assert np.array_equal(q, quantize_samples(
        np.full(16, expected.imag), 16))

    waveform.nonlinear = real_map
    with pytest.raises(TypeError, match="complex waveforms require"):
        waveform.sample()


@pytest.mark.parametrize("method", ["linear", "monotone_cubic"])
@pytest.mark.parametrize("storage", [np.float32, np.float64])
def test_simd_batches_match_scalar_edges_aliasing_and_quantization(
        method, storage):
    mapping = wf.NonlinearMap.from_samples(
        [-1.0, -0.6, -0.1, 0.35, 1.0],
        [-0.8, -0.5, 0.05, 0.4, 0.9],
        method=method, table_size=257, dtype=storage, extrapolate="clip",
    )
    source = np.linspace(-1.2, 1.2, 65)
    source[0] = -1.0
    source[-1] = 1.0
    for count in (1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65):
        values = source[:count].copy()
        expected = np.array([mapping(float(value)) for value in values])
        actual = mapping(values)
        assert np.allclose(actual, expected, rtol=2e-15, atol=2e-15)

        in_place = values.copy()
        assert mapping(in_place, out=in_place) is in_place
        assert np.allclose(in_place, expected, rtol=2e-15, atol=2e-15)

        assert np.array_equal(
            mapping._apply_quantized(values, 16),
            quantize_samples(expected, 16),
        )
        assert np.array_equal(
            mapping._apply_quantized(values, 32),
            quantize_samples(expected, 32),
        )


def test_simd_batches_preserve_error_extrapolation_and_nonfinite_checks():
    mapping = wf.NonlinearMap.from_samples(
        [-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5], table_size=257,
    )
    for bad_value in (-1.01, 1.01, np.nan, np.inf, -np.inf):
        values = np.linspace(-0.9, 0.9, 32)
        values[19] = bad_value
        with pytest.raises(ValueError, match="outside its domain"):
            mapping(values)


def test_simd_quantized_pipeline_spans_multiple_cache_blocks():
    mapping = wf.NonlinearMap.from_samples(
        [-1.0, -0.25, 0.3, 1.0], [-0.9, -0.2, 0.4, 0.95],
        table_size=1025,
    )
    values = 0.99 * np.sin(np.linspace(-70.0, 70.0, 12_345))
    mapped = mapping(values)
    assert np.array_equal(
        mapping._apply_quantized(values, 16),
        quantize_samples(mapped, 16),
    )
    assert np.array_equal(
        mapping._apply_quantized(values, 32),
        quantize_samples(mapped, 32),
    )
