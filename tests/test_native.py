import pickle

import numpy as np
import pytest

import waveforms as wf


RATE = 2_400_000_000


def _pulse(native=False):
    gaussian = wf.native_gaussian if native else wf.gaussian
    cosine = wf.native_cos if native else wf.cos
    sine = wf.native_sin if native else wf.sin
    square = wf.native_square if native else wf.square
    return (
        0.8 * gaussian(20e-9) * cosine(2 * np.pi * 100e6)
        + 0.15 * gaussian(12e-9) * sine(2 * np.pi * 180e6, 0.2)
        + 0.05 * square(6e-9)
    )


def _stack(count, native=False, spacing=40e-9):
    pulse = (0.8 * (wf.native_gaussian(20e-9)
                    if native else wf.gaussian(20e-9))
             * (wf.native_cos(2 * np.pi * 100e6)
                if native else wf.cos(2 * np.pi * 100e6)))
    stack_type = wf.NativeWaveVStack if native else wf.WaveVStack
    stack = stack_type([pulse >> (index * spacing) for index in range(count)])
    stack.start = -20e-9
    stack.stop = count * spacing
    stack.sample_rate = RATE
    return stack


def test_native_wave_block_roundtrip_pickle_and_numerics():
    expected = _pulse()
    actual = _pulse(native=True)
    expected.start = actual.start = -20e-9
    expected.stop = actual.stop = 20e-9
    expected.sample_rate = actual.sample_rate = RATE
    actual.label = "native-pulse"

    data = actual.to_bytes()
    restored = wf.NativeWaveform.from_bytes(data)
    x = np.linspace(-20e-9, 20e-9, 200_000, endpoint=False)

    assert data[:4] == b"WNF4"
    assert actual.to_bytes() is data
    assert restored.to_bytes() is data
    assert restored == wf.NativeWaveform.from_bytes(data)
    assert np.allclose(actual(x), expected(x), rtol=2e-15, atol=2e-15)
    assert np.array_equal(restored(x), actual(x))
    assert len(data) < len(expected.to_bytes())

    unpickled = pickle.loads(pickle.dumps(actual, protocol=5))
    assert unpickled == actual
    assert unpickled.start == actual.start
    assert unpickled.stop == actual.stop
    assert unpickled.sample_rate == actual.sample_rate
    assert unpickled.label == actual.label


@pytest.mark.parametrize("dtype", [np.float64, np.int16, np.int32])
def test_native_device_grid_sampling_matches_packed_backend(dtype):
    expected = _pulse()
    actual = _pulse(native=True)
    expected.start = actual.start = -20e-9
    expected.stop = actual.stop = 20e-9
    expected.sample_rate = actual.sample_rate = RATE

    reference = expected.sample(dtype=dtype)
    result = actual.sample(dtype=dtype)
    if dtype == np.float64:
        assert np.allclose(result, reference, rtol=2e-15, atol=2e-15)
    else:
        assert np.array_equal(result, reference)

    target = np.empty_like(result)
    returned = actual.sample(dtype=dtype, out=target)
    assert returned is target
    assert np.array_equal(target, result)


def test_native_complex_block_roundtrip_and_pickle():
    pulse = _pulse(native=True)
    complex_wave = pulse + 0.5j * (pulse >> 2e-9)
    complex_wave.start = -20e-9
    complex_wave.stop = 20e-9
    complex_wave.sample_rate = RATE
    complex_wave.label = "iq"
    x = np.linspace(-20e-9, 20e-9, 8192, endpoint=False)
    expected = pulse(x) + 0.5j * (pulse >> 2e-9)(x)

    data = complex_wave.to_bytes()
    restored = wf.NativeComplexWaveform.from_bytes(data)
    assert data[:4] == b"WNC4"
    assert np.allclose(restored(x), expected, rtol=2e-15, atol=2e-15)
    unpickled = pickle.loads(pickle.dumps(complex_wave, protocol=5))
    assert np.allclose(unpickled(x), expected, rtol=2e-15, atol=2e-15)
    assert unpickled.start == complex_wave.start
    assert unpickled.stop == complex_wave.stop
    assert unpickled.sample_rate == complex_wave.sample_rate
    assert unpickled.label == complex_wave.label


def test_native_stack_sampling_lazy_roundtrip_shift_offset_and_pickle():
    expected = _stack(100)
    actual = _stack(100, native=True)

    assert np.allclose(actual.sample(), expected.sample(),
                       rtol=2e-15, atol=2e-15)
    assert np.array_equal(actual.sample(dtype=np.int16),
                          expected.sample(dtype=np.int16))

    data = actual.to_bytes()
    restored = wf.NativeWaveVStack.from_bytes(data)
    restored.start = actual.start
    restored.stop = actual.stop
    restored.sample_rate = actual.sample_rate
    assert data[:4] == b"WNS4"
    assert restored.to_bytes() is data
    assert np.array_equal(restored.sample(), actual.sample())

    transformed = (actual >> 5e-9) + 0.125
    transformed_data = transformed.to_bytes()
    transformed_restored = wf.NativeWaveVStack.from_bytes(transformed_data)
    x = np.linspace(-20e-9, 4e-6, 10_000, endpoint=False)
    assert transformed_data[:4] == b"WNS4"
    assert np.allclose(transformed_restored(x), transformed(x),
                       rtol=2e-15, atol=2e-15)

    actual.label = "events"
    unpickled = pickle.loads(pickle.dumps(actual, protocol=5))
    assert unpickled == actual
    assert unpickled.sample_rate == actual.sample_rate
    assert unpickled.label == actual.label


def test_native_overlapping_stack_quantization_and_empty_stack():
    expected = _stack(50, spacing=10e-9)
    actual = _stack(50, native=True, spacing=10e-9)
    assert np.array_equal(actual.sample(dtype=np.int16),
                          expected.sample(dtype=np.int16))
    assert np.array_equal(actual.sample(dtype=np.int32),
                          expected.sample(dtype=np.int32))

    empty = wf.NativeWaveVStack([])
    empty.start = -1e-9
    empty.stop = 1e-9
    empty.sample_rate = RATE
    assert empty.to_bytes() == b"WNS4\x01\x00\x00\x00" + b"\x00" * 8
    assert not np.any(empty.sample())
    assert not np.any(wf.NativeWaveVStack.from_bytes(empty.to_bytes())(
        np.linspace(-1e-9, 1e-9, 17)
    ))


def test_native_blocks_reject_malformed_layouts():
    wave_data = bytearray(_pulse(native=True).to_bytes())
    with pytest.raises(ValueError):
        wf.NativeWaveform.from_bytes(wave_data[:20])
    wave_data[0] = 0
    with pytest.raises(ValueError):
        wf.NativeWaveform.from_bytes(wave_data)

    wave_data = bytearray(_pulse(native=True).to_bytes())
    wave_data[24] = 255
    with pytest.raises(ValueError):
        wf.NativeWaveform.from_bytes(wave_data)

    stack_data = bytearray(_stack(2, native=True).to_bytes())
    with pytest.raises(ValueError):
        wf.NativeWaveVStack.from_bytes(stack_data[:-1])
    event_offset = len(stack_data) - 40
    stack_data[event_offset:event_offset + 4] = (99).to_bytes(4, "little")
    malformed = wf.NativeWaveVStack.from_bytes(stack_data)
    with pytest.raises(ValueError):
        _ = malformed._core.event_count


def test_native_format_is_language_neutral_and_self_describing():
    description = wf.native_format_description()
    assert "little-endian" in description
    assert "120 GHz" in description
    assert "ABI 1" in description
