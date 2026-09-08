"""Regression coverage for direct native loading and output placement."""

import ctypes
from concurrent.futures import ThreadPoolExecutor
import importlib
import pickle
import struct

import numpy as np
import pytest

import waveforms as wf
import waveforms._waveform as core
from waveforms._waveform import quantize_samples


@pytest.fixture(scope='module')
def native():
    library = ctypes.CDLL(core.__file__)
    library.cwaveform_stack_from_bytes.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    library.cwaveform_stack_from_bytes.restype = ctypes.c_void_p
    library.cwaveform_stack_release.argtypes = [ctypes.c_void_p]
    library.cwaveform_stack_release.restype = None
    library.cwaveform_stack_bytes.argtypes = [ctypes.c_void_p,
                                            ctypes.POINTER(ctypes.c_size_t)]
    library.cwaveform_stack_bytes.restype = ctypes.c_void_p
    library.cwaveform_stack_hash.argtypes = [ctypes.c_void_p]
    library.cwaveform_stack_hash.restype = ctypes.c_uint64
    return library


def _stack(delays, *, scales=None):
    delays = np.asarray(delays, dtype=np.int64)
    if scales is None:
        scales = np.resize([.5, -.75, 1.0], len(delays))
    stack = wf.WaveVStack.from_events(
        [.8 * wf.square(8 / 120e9), -.3 * wf.square(4 / 120e9)],
        np.arange(len(delays), dtype=np.uint32) % 2, delays, scales)
    stack.start, stack.stop, stack.sample_rate = 0, 120 / 120e9, 120_000_000_000
    stack.offset = .07
    return stack


def test_native_decode_owns_exact_bytes_and_preserves_hash(native):
    original = _stack([-9, 0, 19, 33, 61, 118, 130])
    data = original._core.to_bytes()
    source = ctypes.create_string_buffer(data)
    handle = native.cwaveform_stack_from_bytes(source, len(data))
    assert handle
    try:
        # Mutating the transport buffer must not change the native handle.
        ctypes.memset(source, 0, len(data))
        size = ctypes.c_size_t()
        pointer = native.cwaveform_stack_bytes(handle, ctypes.byref(size))
        assert size.value == len(data)
        assert ctypes.string_at(pointer, size.value) == data
        assert native.cwaveform_stack_hash(handle) == original._core.hash64
    finally:
        native.cwaveform_stack_release(handle)


def test_lazy_native_hash_matches_format_and_is_safe_for_parallel_readers(native):
    stack = _stack(np.arange(10000) * 13)
    data = stack._core.to_bytes()
    expected = 1469598103934665603
    for byte in data:
        expected = ((expected ^ byte) * 1099511628211) & ((1 << 64) - 1)
    source = ctypes.create_string_buffer(data)
    handle = native.cwaveform_stack_from_bytes(source, len(data))
    assert handle
    try:
        with ThreadPoolExecutor(max_workers=8) as workers:
            hashes = list(workers.map(
                lambda _: native.cwaveform_stack_hash(handle), range(64)))
        assert hashes == [expected] * len(hashes)
        assert stack._core.hash64 == expected
    finally:
        native.cwaveform_stack_release(handle)


def test_native_decode_rejects_invalid_blocks_and_releases_partial_loads(native):
    data = _stack([0, 16, 32])._core.to_bytes()
    malformed = [data[:index] for index in range(len(data))]
    event_offset = len(data) - 3 * 20
    for offset, fmt, value in (
        (0, '4s', b'BAD!'), (4, 'H', 65535), (6, 'H', 1),
        (8, 'I', 2**32 - 1), (12, 'I', 2**32 - 1),
        (16, 'I', 2**32 - 1),
        (24 + 24, 'B', 255),  # First template's first node.
        (event_offset, 'I', 2),
        (event_offset + 12 * 3, 'd', np.nan),
        (event_offset + 12 * 3 + 8, 'd', np.inf),
        (event_offset + 12 * 3 + 16, 'd', -np.inf),
    ):
        candidate = bytearray(data)
        struct.pack_into('<' + fmt, candidate, offset, value)
        malformed.append(bytes(candidate))
    for candidate in malformed:
        source = ctypes.create_string_buffer(candidate)
        handle = native.cwaveform_stack_from_bytes(source, len(candidate))
        try:
            assert not handle
        finally:
            if handle:
                native.cwaveform_stack_release(handle)


@pytest.mark.parametrize('delays', [[], [-30, -20], [140, 150], [0, 119],
                                    [4, 10, 16, 22], [8, 30, 70],
                                    [8, 30, 31], [80, 30, 8]])
@pytest.mark.parametrize('bits', [0, 16, 32])
@pytest.mark.parametrize('bounds', [(-np.inf, np.inf), (.1, .2)])
def test_gap_initialization_edges_overlap_and_out_of_order(delays, bits, bounds):
    stack = _stack(delays)
    stack.min, stack.max = bounds
    restored = pickle.loads(pickle.dumps(stack)) >> (1 / 120e9)
    expected = restored._core.sample(0, 120, 1, 1, 1, stack.offset)
    np.clip(expected, *bounds, out=expected)
    if bits:
        expected = quantize_samples(expected, bits, .7)
    output = np.full_like(expected, 17)
    assert restored.sample(out=output, full_scale=.7) is output
    if bits:
        np.testing.assert_array_equal(output, expected)
    else:
        # The plan groups by template; overlapping additions can differ from
        # the event-order evaluator by a rounding bit even before this change.
        np.testing.assert_allclose(output, expected, rtol=1e-15, atol=1e-15)
    first = output.copy()
    output.fill(23)
    restored.sample(out=output, full_scale=.7)
    np.testing.assert_array_equal(output, first)
    reference = stack >> (1 / 120e9)
    assert restored.begin == reference.begin
    assert restored.end == reference.end
    assert restored._core.lower_tick == stack._core.lower_tick
    assert restored._core.upper_tick == stack._core.upper_tick


def test_gap_initialization_skips_zero_scales_and_empty_templates():
    stack = _stack([8, 30, 70], scales=[0, 1, 0])
    expected = stack._core.sample(0, 120, 1, 1, 0, stack.offset)
    np.testing.assert_array_equal(stack.sample(), expected)
    empty = wf.WaveVStack.from_events([wf.zero()], [0], [0], [1.0])
    empty.start, empty.stop, empty.sample_rate = 0, 1e-9, 10_000_000_000
    np.testing.assert_array_equal(empty.sample(), np.zeros(10))


@pytest.mark.parametrize('bits', [0, 16, 32])
def test_many_scales_and_sparse_output(bits):
    stack = _stack(np.arange(200) * 13, scales=np.linspace(-.8, .8, 200))
    stack.stop = 2700 / 120e9
    stack.min, stack.max = -.2, .25
    expected = stack._core.sample(0, 2700, 1, 1, 0, stack.offset)
    np.clip(expected, -.2, .25, out=expected)
    if bits:
        expected = quantize_samples(expected, bits, .7)
    output = np.full_like(expected, 17)
    stack.sample(out=output, full_scale=.7)
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize('templates_count', [1, 8, 9, 16, 33, 129])
def test_plan_lookup_growth_repeated_keys_and_multiple_tick_phases(templates_count):
    rng = np.random.default_rng(417)
    templates = [wf.square((100 + index) / 120e9)
                 for index in range(templates_count)]
    count = 2 * templates_count + 100
    ids = np.arange(count, dtype=np.uint32) % templates_count
    delays = np.arange(count, dtype=np.int64) * 500 + rng.integers(0, 50, count)
    scales = np.resize([.125, -.25, .5], count)
    stack = wf.WaveVStack.from_events(templates, ids, delays, scales)
    stack.start, stack.stop, stack.sample_rate = 0, count * 500 / 120e9, 2_400_000_000
    restored = pickle.loads(pickle.dumps(stack)) >> (7 / 120e9)
    samples = count * 10
    expected = restored._core.sample(0, samples, 50, 1, 7, 0)
    np.testing.assert_array_equal(restored.sample(), expected)
    plan = restored._sample_plan_cache[1]
    assert plan.group_count == len(set(zip(ids, (-(delays + 7)) % 50)))
    np.testing.assert_array_equal(restored.sample(), expected)
    # Rebuilding with a different calibration changes phases, not the format.
    restored.shift = 11 / 120e9
    expected = restored._core.sample(0, samples, 50, 1, 11, 0)
    np.testing.assert_array_equal(restored.sample(), expected)


@pytest.mark.parametrize('stack', [False, True])
@pytest.mark.parametrize('rate', [2_400_000_000, 1_234_567_891])
def test_float_output_is_written_directly(stack, rate, monkeypatch):
    module = importlib.import_module('waveforms.waveform')
    wave = wf.cos(2 * np.pi * 30e6)
    if stack:
        wave = wf.WaveVStack([wave * wf.square(2e-6)])
    wave.start, wave.stop, wave.sample_rate = 0, 1e-6, rate
    expected = wave.sample()
    output = np.empty_like(expected)
    finish = module._finish_samples

    def check_direct(values, dtype, full_scale, out, *args):
        assert values is output and out is output
        return finish(values, dtype, full_scale, out, *args)

    monkeypatch.setattr(module, '_finish_samples', check_direct)
    assert wave.sample(out=output) is output
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize('stack', [False, True])
@pytest.mark.parametrize('dtype', [None, np.float32, np.float64, np.complex128])
@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'unaligned', 'big_endian'])
def test_output_casting_and_layout_fallbacks(stack, dtype, layout):
    wave = .8 * wf.cos(2 * np.pi * 30e6)
    if stack:
        wave = wf.WaveVStack([wave * wf.square(2e-6)])
    wave.start, wave.stop, wave.sample_rate = 0, 1e-6, 2_400_000_000
    expected = wave.sample(dtype=dtype)
    output_dtype = np.complex128 if dtype == np.complex128 else np.float64
    if layout == 'strided':
        output = np.empty(2 * len(expected), dtype=output_dtype)[::2]
    elif layout == 'unaligned':
        output = np.ndarray(expected.shape, dtype=output_dtype,
                            buffer=bytearray(expected.size * np.dtype(output_dtype).itemsize + 1),
                            offset=1)
    else:
        if layout == 'big_endian':
            output_dtype = np.dtype(output_dtype).newbyteorder('>')
        output = np.empty(expected.shape, dtype=output_dtype)
    assert wave.sample(dtype=dtype, out=output) is output
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize('stack', [False, True])
def test_float_output_validation(stack):
    wave = _stack([10, 30]) if stack else wf.cos(1)
    wave.start, wave.stop, wave.sample_rate = 0, 1e-6, 1_000_000_000
    for output in (np.empty(999), np.empty((1, 1000))):
        with pytest.raises(ValueError, match='shape'):
            wave.sample(out=output)
    output = np.empty(1000)
    output.flags.writeable = False
    with pytest.raises(ValueError, match='writ'):
        wave.sample(out=output)
