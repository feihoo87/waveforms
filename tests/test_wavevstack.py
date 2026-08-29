import pickle

import numpy as np
from scipy.signal import butter, lfilter, lfiltic, tf2sos

from waveforms import (
    Waveform, WaveVStack, cos, gaussian, pi, poly, sin, step,
    time_to_tick, zero,
)


def _waves():
    return [cos(1), sin(2), gaussian(3),
            poly([1, -1 / 2, 1 / 6, -1 / 12])]


def test_wavevstack_sampling_algebra_and_shifts():
    x = np.linspace(-10, 10, 1001)
    waves = _waves()
    expected = zero()
    for wav in waves:
        expected += wav
    stack = WaveVStack(waves)

    assert stack.simplify() == expected
    assert np.allclose(stack(x), expected(x))
    assert np.allclose((stack + sin(2))(x), (expected + sin(2))(x))
    assert np.allclose((stack - sin(2))(x), (expected - sin(2))(x))
    assert np.allclose((stack * sin(2) + 3)(x),
                       (expected * sin(2) + 3)(x))
    assert np.allclose((stack / 2)(x), (expected / 2)(x))
    assert np.allclose((stack >> 0.6)(x), (expected >> 0.6)(x))
    assert np.allclose((stack << 1.4)(x), (expected << 1.4)(x))


def test_wavevstack_filtering_and_chunked_sampling():
    sample_rate = 1000
    b, a = butter(3, 4.0, "lowpass", fs=sample_rate)
    zi = lfiltic(b, a, [0])
    x = np.linspace(-1, 1, 2000, endpoint=False)

    stack = WaveVStack([step(0) << 0.5, -step(0)])
    stack.sample_rate = sample_rate
    stack.start = -1
    stack.stop = 1
    stack.filters = (tf2sos(b, a), 0)
    expected = lfilter(
        b, a, np.heaviside(x + 0.5, 1) - np.heaviside(x, 1), zi=zi
    )[0]

    assert np.allclose(stack.sample(), expected, atol=1e-6)
    chunks = np.concatenate(list(stack.sample(chunk_size=137)))
    assert np.allclose(chunks, expected, atol=1e-6)


def test_wavevstack_binary_template_sharing_and_pickle():
    template = gaussian(20e-9) * cos(2 * pi * 5e9)
    stack = WaveVStack([template >> (index * 80e-9)
                        for index in range(10_000)])
    assert len(stack.to_bytes()) < 300_000

    restored = WaveVStack.from_bytes(stack.to_bytes())
    assert restored == stack
    assert pickle.loads(pickle.dumps(stack)) == stack


def test_wavevstack_content_deduplication_and_constant_time_support():
    base = gaussian(20e-9) * cos(2 * pi * 5e9)
    waves = [
        Waveform.from_bytes(base.to_bytes()) >> (index * 80e-9)
        for index in range(100)
    ]
    stack = WaveVStack([zero(), *waves])

    assert len(stack) == stack.event_count == 100
    assert stack.template_count == 1
    assert np.isclose(stack.begin, base.begin)
    assert np.isclose(stack.end, base.end + 99 * 80e-9)

    restored = WaveVStack.from_bytes(stack.to_bytes())
    shifted = restored >> 2e-9
    assert restored.template_count == 1
    assert restored.event_count == 100
    assert np.isclose(shifted.begin, stack.begin + 2e-9)
    assert np.isclose(shifted.end, stack.end + 2e-9)

    empty = WaveVStack()
    assert empty.begin == -np.inf
    assert empty.end == np.inf


def test_wavevstack_compiler_event_columns():
    template = gaussian(20e-9) * cos(2 * pi * 5e9)
    delays = np.arange(100, dtype=np.int64) * time_to_tick(80e-9)
    stack = WaveVStack.from_events(
        (template,), np.zeros(100, dtype=np.uint32), delays,
        np.full(100, 0.25),
    )
    expected = WaveVStack([
        0.25 * template >> (index * 80e-9) for index in range(100)
    ])

    assert stack.template_count == 1
    assert stack.event_count == 100
    assert stack.to_bytes() == expected.to_bytes()
    positions = np.linspace(-20e-9, 8e-6, 1000)
    assert np.allclose(stack(positions), expected(positions))
