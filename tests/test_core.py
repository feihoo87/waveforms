import pickle
from pathlib import Path

import numpy as np
import pytest

import waveforms as wf
import waveforms._waveform as core
from waveforms._waveform import quantize_samples


RATE = 2_400_000_000


def _pulse():
    return (
        0.8 * wf.gaussian(20e-9) * wf.cos(2 * np.pi * 100e6)
        + 0.15 * wf.gaussian(12e-9) * wf.sin(2 * np.pi * 180e6, 0.2)
        + 0.05 * wf.square(6e-9)
    )


def _stack(count, spacing=40e-9):
    pulse = 0.8 * wf.gaussian(20e-9) * wf.cos(2 * np.pi * 100e6)
    stack = wf.WaveVStack([
        pulse >> (index * spacing) for index in range(count)
    ])
    stack.start = -20e-9
    stack.stop = count * spacing
    stack.sample_rate = RATE
    return stack


def test_unified_type_hierarchy_and_removed_legacy_api():
    real = wf.gaussian(20e-9)
    complex_wave = real + 0.5j * real
    real_stack = wf.WaveVStack([real])
    complex_stack = wf.ComplexWaveVStack([complex_wave])

    assert isinstance(real, wf.RealWaveform)
    assert isinstance(real_stack, wf.RealWaveVStack)
    assert isinstance(real_stack, wf.WaveVStack)
    assert isinstance(complex_stack, wf.WaveVStack)
    assert all(isinstance(value, wf.Waveform) for value in (
        real, complex_wave, real_stack, complex_stack
    ))
    assert not any(name.startswith("native_") for name in dir(wf))
    assert not any(name.startswith("Native") for name in dir(wf))
    assert not hasattr(core, "PackedWaveform")
    assert not hasattr(core, "PackedStack")


def test_canonical_real_waveform_equality_and_hash():
    a = wf.gaussian(12e-9)
    b = wf.cos(2 * np.pi * 100e6)

    assert a + b == b + a
    assert hash(a + b) == hash(b + a)
    assert a + a == 2 * a
    assert a * a == a ** 2
    assert (a + b).simplify().to_bytes() == (b + a).simplify().to_bytes()
    assert wf.WaveVStack([a, b]) == wf.WaveVStack([b, a])


def test_c_symbolic_frequency_filter():
    envelope = wf.gaussian(20e-9)
    low = envelope * wf.cos(2 * np.pi * 100e6)
    high = 0.25 * envelope * wf.sin(2 * np.pi * 300e6)
    baseband = 0.1 * envelope
    signal = low + high + baseband
    x = np.linspace(-20e-9, 20e-9, 4097)

    assert np.allclose(
        signal.filter(2 * np.pi * 200e6, 2 * np.pi * 400e6)(x),
        high(x),
    )
    assert np.allclose(
        signal.filter(0, 2 * np.pi * 200e6)(x),
        (low + baseband)(x),
    )


def test_c_symbolic_filter_reduces_trigonometric_products_and_powers():
    x = np.linspace(-1.0, 1.0, 4097)
    envelope = wf.gaussian(2.0)
    first_frequency = 9.0
    second_frequency = 7.0
    first_phase = 0.31
    second_phase = -0.27
    cutoff = 5.0

    first_cos = wf.cos(first_frequency, first_phase)
    first_sin = wf.sin(first_frequency, first_phase)
    second_cos = wf.cos(second_frequency, second_phase)
    second_sin = wf.sin(second_frequency, second_phase)
    difference = first_frequency - second_frequency
    phase_difference = first_phase - second_phase

    assert np.allclose(
        (2 * envelope * first_cos * second_cos).filter(high=cutoff)(x),
        (envelope * wf.cos(difference, phase_difference))(x),
        atol=2e-11,
    )
    assert np.allclose(
        (2 * envelope * first_cos * second_sin).filter(high=cutoff)(x),
        (-envelope * wf.sin(difference, phase_difference))(x),
        atol=2e-11,
    )
    assert np.allclose(
        (2 * envelope * first_sin * second_sin).filter(high=cutoff)(x),
        (envelope * wf.cos(difference, phase_difference))(x),
        atol=2e-11,
    )

    powered = wf.cos(first_frequency, first_phase) ** 2
    assert np.allclose(powered.filter(high=cutoff)(x), 0.5, atol=2e-11)
    assert np.allclose(
        powered.filter(2 * first_frequency, np.inf)(x),
        (0.5 * wf.cos(2 * first_frequency, 2 * first_phase))(x),
        atol=2e-11,
    )

    retained = (1.5e-15 * first_cos).filter(eps=1e-15)
    discarded = (0.5e-15 * first_cos).filter(eps=1e-15)
    assert np.max(np.abs(retained(x))) > 1e-15
    assert np.array_equal(discarded(x), np.zeros_like(x))


def test_down_conversion_filter_matches_historic_symbolic_behavior():
    x = np.linspace(-100.0, 100.0, 10001)
    envelope = wf.gaussian(100.0)
    radio_frequency = 92.0451
    local_frequency = 92.0
    phase = 0.32
    rf, _ = wf.mixing(
        envelope, freq=radio_frequency, phase=phase, DRAGScaling=0.0,
    )

    i = (2 * rf * wf.cos(-2 * np.pi * local_frequency)).filter(
        high=2 * np.pi * local_frequency,
    )
    q = (2 * rf * wf.sin(-2 * np.pi * local_frequency)).filter(
        high=2 * np.pi * local_frequency,
    )
    difference = 2 * np.pi * (radio_frequency - local_frequency)

    assert np.allclose(
        i(x), (envelope * wf.cos(difference, -phase))(x), atol=2e-9,
    )
    assert np.allclose(
        q(x), (envelope * wf.sin(difference, -phase))(x), atol=2e-9,
    )


def test_wave_block_roundtrip_pickle_and_numerics():
    actual = _pulse()
    actual.start = -20e-9
    actual.stop = 20e-9
    actual.sample_rate = RATE
    actual.label = "pulse"
    data = actual.to_bytes()
    restored = wf.Waveform.from_bytes(data)
    x = np.linspace(-20e-9, 20e-9, 200_000, endpoint=False)

    assert data[:4] == b"WNF4"
    assert actual.to_bytes() is data
    assert restored.to_bytes() is data
    assert restored == wf.Waveform.from_bytes(data)
    assert wf.RealWaveform.from_bytes(data) == restored
    assert np.array_equal(restored(x), actual(x))

    unpickled = pickle.loads(pickle.dumps(actual, protocol=5))
    assert unpickled == actual
    assert unpickled.start == actual.start
    assert unpickled.stop == actual.stop
    assert unpickled.sample_rate == actual.sample_rate
    assert unpickled.label == actual.label


@pytest.mark.parametrize("dtype", [np.float64, np.int16, np.int32])
def test_device_grid_sampling_and_output_buffer(dtype):
    actual = _pulse()
    actual.start = -20e-9
    actual.stop = 20e-9
    actual.sample_rate = RATE
    result = actual.sample(dtype=dtype)
    target = np.empty_like(result)
    assert actual.sample(dtype=dtype, out=target) is target
    assert np.array_equal(target, result)


def test_complex_block_roundtrip_and_pickle():
    pulse = _pulse()
    complex_wave = pulse + 0.5j * (pulse >> 2e-9)
    complex_wave.start = -20e-9
    complex_wave.stop = 20e-9
    complex_wave.sample_rate = RATE
    complex_wave.label = "iq"
    x = np.linspace(-20e-9, 20e-9, 8192, endpoint=False)
    expected = pulse(x) + 0.5j * (pulse >> 2e-9)(x)

    data = complex_wave.to_bytes()
    restored = wf.Waveform.from_bytes(data)
    assert data[:4] == b"CWF1"
    assert isinstance(restored, wf.ComplexWaveform)
    assert np.allclose(restored(x), expected, rtol=2e-15, atol=2e-15)
    unpickled = pickle.loads(pickle.dumps(complex_wave, protocol=5))
    assert np.allclose(unpickled(x), expected, rtol=2e-15, atol=2e-15)
    assert unpickled.label == complex_wave.label


def test_stack_sampling_roundtrip_shift_offset_and_pickle():
    actual = _stack(100)
    data = actual.to_bytes()
    restored = wf.WaveVStack.from_bytes(data)
    restored.start = actual.start
    restored.stop = actual.stop
    restored.sample_rate = actual.sample_rate

    assert data[:4] == b"WNS4"
    assert restored.to_bytes() is data
    assert isinstance(wf.RealWaveVStack.from_bytes(data),
                      wf.RealWaveVStack)
    assert np.array_equal(restored.sample(), actual.sample())
    assert np.array_equal(restored.sample(dtype=np.int16),
                          actual.sample(dtype=np.int16))

    transformed = (actual >> 5e-9) + 0.125
    transformed_restored = wf.WaveVStack.from_bytes(transformed.to_bytes())
    x = np.linspace(-20e-9, 4e-6, 10_000, endpoint=False)
    assert np.allclose(transformed_restored(x), transformed(x),
                       rtol=2e-15, atol=2e-15)

    actual.label = "events"
    unpickled = pickle.loads(pickle.dumps(actual, protocol=5))
    assert unpickled == actual
    assert unpickled.sample_rate == actual.sample_rate
    assert unpickled.label == actual.label


@pytest.mark.parametrize("dtype,bits", [
    (np.float64, 0),
    (np.int16, 16),
    (np.int32, 32),
])
def test_stack_sample_plan_matches_direct_sampler(dtype, bits):
    pulse = wf.gaussian(20e-9) * wf.cos(2 * np.pi * 100e6)
    scales = (0.25, 0.5, 1.0)
    actual = wf.WaveVStack([
        scales[index % len(scales)] * pulse >> (index * 40e-9)
        for index in range(75)
    ])
    actual.start = -20e-9
    actual.stop = 75 * 40e-9
    actual.sample_rate = RATE
    actual.offset = 0.071

    first = actual.sample(dtype=dtype)
    cache_key, sample_plan = actual._sample_plan_cache
    target = np.empty_like(first)
    result = actual.sample(dtype=dtype, out=target)
    direct = actual._core.sample(
        cache_key[0], cache_key[1], cache_key[2], cache_key[3],
        cache_key[4], actual.offset, bits, 1.0,
    )

    assert result is target
    assert sample_plan.count == len(result)
    assert sample_plan.group_count == 1
    assert sample_plan.non_overlapping
    assert np.array_equal(result, direct)
    assert actual._sample_plan_cache[1] is sample_plan


def test_overlapping_stack_quantization_and_empty_stack():
    actual = _stack(50, spacing=10e-9)
    for dtype, bits in ((np.int16, 16), (np.int32, 32)):
        expected = quantize_samples(actual.sample(), bits)
        assert np.array_equal(actual.sample(dtype=dtype), expected)

    empty = wf.WaveVStack([])
    empty.start = -1e-9
    empty.stop = 1e-9
    empty.sample_rate = RATE
    assert empty.to_bytes() == b"WNS4\x02\x00\x00\x00" + b"\x00" * 8
    assert not np.any(empty.sample())
    assert not np.any(wf.WaveVStack.from_bytes(empty.to_bytes())(
        np.linspace(-1e-9, 1e-9, 17)
    ))


def test_blocks_reject_malformed_layouts():
    wave_data = bytearray(_pulse().to_bytes())
    with pytest.raises(ValueError):
        wf.Waveform.from_bytes(wave_data[:20])
    wave_data[0] = 0
    with pytest.raises(ValueError):
        wf.Waveform.from_bytes(wave_data)

    wave_data = bytearray(_pulse().to_bytes())
    wave_data[24] = 255
    with pytest.raises(ValueError):
        wf.Waveform.from_bytes(wave_data)

    stack_data = bytearray(_stack(2).to_bytes())
    with pytest.raises(ValueError):
        wf.WaveVStack.from_bytes(stack_data[:-1])
    event_offset = len(stack_data) - 40
    stack_data[event_offset:event_offset + 4] = (99).to_bytes(4, "little")
    malformed = wf.WaveVStack.from_bytes(stack_data)
    with pytest.raises(ValueError):
        _ = malformed._core.event_count


def test_format_is_language_neutral_and_uses_the_global_clock():
    data = _pulse().to_bytes()
    assert data[:4] == b"WNF4"
    assert int.from_bytes(data[4:6], "little") == 2
    assert wf.get_time_resolution() == 1 / 120_000_000_000
    package_dir = Path(wf.__file__).resolve().parent
    assert (package_dir / "_cwaveform.c").is_file()
    assert (package_dir / "_cwaveform.h").is_file()
