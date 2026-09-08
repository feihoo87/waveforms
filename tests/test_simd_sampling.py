"""Numerical boundaries for native SIMD quantization and table loads."""

import pickle

import numpy as np
import pytest

import waveforms as wf
from waveforms._waveform import quantize_samples


def reference_quantize(values, bits, full_scale):
    limit = float(1 << (bits - 1))
    scaled = np.clip(values * (limit / full_scale), -limit, limit - 1)
    integral = np.trunc(scaled)
    rounded = integral + np.copysign(
        (np.abs(scaled - integral) >= .5).astype(float), scaled)
    return rounded.astype(np.int16 if bits == 16 else np.int32)


@pytest.mark.parametrize('bits', [16, 32])
@pytest.mark.parametrize('full_scale', [1., .7, 2., 1e-100, 1e100])
def test_quantization_half_lsb_neighbors_tails_and_saturation(bits, full_scale):
    limit = float(1 << (bits - 1))
    halves = np.array([.5, 1.5, 2.5, 127.5, limit - 1.5])
    steps = np.concatenate([np.nextafter(halves, 0), halves,
                            np.nextafter(halves, np.inf)])
    values = np.concatenate([steps, -steps, [-2 * limit, 2 * limit, 0., -0.]])
    values *= full_scale / limit
    # Rotate the half-LSB neighbors through every SIMD lane and the tail.
    for count in (0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65, 257):
        for shift in range(8):
            source = np.resize(np.roll(values, shift), count)
            expected = reference_quantize(source, bits, full_scale)
            padded = np.full(count + 2, 123, dtype=expected.dtype)
            output = padded[1:-1]  # Not 16/32/64-byte aligned.
            assert quantize_samples(source, bits, full_scale, out=output) is output
            np.testing.assert_array_equal(output, expected)
            assert padded[0] == padded[-1] == 123


@pytest.mark.parametrize('bits', [16, 32])
def test_nonfinite_quantization_in_each_vector_lane_and_tail(bits):
    for count in (16, 33, 257):
        for lane in range(count):
            source = np.zeros(count)
            source[lane] = [np.nan, np.inf, -np.inf][lane % 3]
            with pytest.raises(ValueError):
                quantize_samples(source, bits)


@pytest.mark.parametrize('storage', [np.float32, np.float64])
@pytest.mark.parametrize('method', ['linear', 'monotone_cubic'])
def test_table_loads_random_intervals_endpoints_and_aliasing(storage, method):
    rng = np.random.default_rng(7428)
    for table_size in (2, 3, 17, 4097):
        mapping = wf.NonlinearMap.from_samples(
            [-1., -.73, -.11, .29, 1.], [-.9, -.79, -.25, .42, .96],
            method=method, dtype=storage, table_size=table_size,
            extrapolate='clip')
        for count in (7, 8, 9, 15, 16, 17, 31, 32, 33, 257):
            source = rng.uniform(-1.2, 1.2, count)
            source[::4] = -1.
            source[1::4] = 1.
            expected = np.array([mapping(float(value)) for value in source])
            padded = np.full(count + 2, 123.)
            output = padded[1:-1]
            mapping(source, out=output)
            np.testing.assert_allclose(output, expected, rtol=2e-15, atol=2e-15)
            assert padded[0] == padded[-1] == 123.
            mapping(source, out=source)
            np.testing.assert_array_equal(source, output)


@pytest.mark.parametrize('bits', [16, 32])
@pytest.mark.parametrize('varied', [False, True])
@pytest.mark.parametrize('limited', [False, True])
@pytest.mark.parametrize('offset', [0., .07])
def test_stack_template_quantization_blocks_and_clipped_placements(
        bits, varied, limited, offset):
    events = 101
    pulse = .9 * wf.gaussian(300e-9)
    scales = (np.linspace(-.9, .9, events) if varied else
              np.resize([.5, -.75, 1.], events))
    stack = wf.WaveVStack.from_events(
        [pulse], np.zeros(events, dtype=np.uint32),
        np.arange(events, dtype=np.int64) * wf.time_to_tick(600e-9), scales)
    stack.start, stack.stop = 0., (events - 1) * 600e-9
    stack.sample_rate = 2_400_000_000
    stack.offset = offset
    if limited:
        stack.min, stack.max = -.3, .35
    restored = pickle.loads(pickle.dumps(stack)) >> (7 / 120e9)
    expected = reference_quantize(restored.sample(), bits, .7)
    np.testing.assert_array_equal(restored.sample(dtype=expected.dtype, full_scale=.7), expected)
    output = np.empty_like(expected)
    assert restored.sample(out=output, full_scale=.7) is output
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize('block', [None, -200e6, 80e6])
@pytest.mark.parametrize('width', [12e-9, 20e-9, 1000e-9])
def test_drag_vector_evaluation_and_cold_template_plan(block, width):
    templates = [.4 * wf.drag(5e9, width, delta=-200e6, block_freq=block,
                             phase=index * .371, t0=3e-9)
                 for index in range(9)]
    pulse = templates[3] >> (7 / 120e9)
    positions = np.linspace(-width, 2 * width, 1031)
    scalar = np.array([pulse(float(position)) for position in positions])
    np.testing.assert_allclose(pulse(positions), scalar, rtol=2e-14, atol=2e-14)
    # Tick residues exercise different sampling phases and clipping at both
    # ends; the old event evaluator provides an independent scalar reference.
    delays = (np.arange(31, dtype=np.int64) * wf.time_to_tick(3 * width)
              + np.arange(31) % 50)
    stack = wf.WaveVStack.from_events(
        templates, np.arange(31, dtype=np.uint32) % len(templates), delays,
        np.resize([.5, -.75, 1.], 31))
    stack.start, stack.stop = 0., 90 * width
    stack.sample_rate = 2_400_000_000
    restored = pickle.loads(pickle.dumps(stack)) >> (7 / 120e9)
    block_before_sampling = restored.to_bytes()
    count = len(restored.sample())
    scalar = restored._core.sample(0, count, 50, 1, 7, 0.)
    np.testing.assert_allclose(restored.sample(), scalar, rtol=2e-14, atol=2e-14)
    for bits in (16, 32):
        expected = reference_quantize(scalar, bits, 1.)
        np.testing.assert_array_equal(restored.sample(dtype=expected.dtype), expected)
    assert restored.to_bytes() == block_before_sampling


def test_drag_plan_falls_back_for_unsupported_companion_nodes():
    pulse = wf.drag(5e9, 1e-6, delta=-200e6, block_freq=80e6) * wf.mollifier(1e-6)
    stack = wf.WaveVStack.from_events([pulse], [0], [0], [.7])
    stack.start, stack.stop, stack.sample_rate = 0., 1e-6, 2_400_000_000
    expected = stack._core.sample(0, 2400, 50, 1, 0, 0.)
    np.testing.assert_array_equal(stack.sample(), expected)
