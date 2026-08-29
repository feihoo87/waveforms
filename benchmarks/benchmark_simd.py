"""Benchmark the loops targeted by native SIMD and batched-math kernels."""

from __future__ import annotations

import argparse
import gc
import json
import time

import numpy as np

import waveforms as wf
from waveforms._waveform import quantize_samples


def best_time(function, repeats):
    function()
    timings = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        function()
        timings.append(time.perf_counter() - start)
    return min(timings)


def benchmark(count, rate, repeats):
    duration = count / rate
    start = -duration / 2
    stop = start + duration
    width = duration * 1.8
    pulse = 0.8 * wf.gaussian(width) * wf.cos(2 * np.pi * 100e6)
    powered = pulse**2
    positions = np.linspace(start, stop, count, endpoint=False)
    values = 0.999 * np.sin(np.linspace(-100, 100, count))

    pulse.start = start
    pulse.stop = stop
    pulse.sample_rate = rate

    cases = {
        "evaluate_supported": lambda: pulse(positions),
        "evaluate_power_fallback": lambda: powered(positions),
        "sample_float64": lambda: pulse.sample(dtype=np.float64),
        "sample_int16": lambda: pulse.sample(dtype=np.int16),
        "sample_int32": lambda: pulse.sample(dtype=np.int32),
        "quantize_int16": lambda: quantize_samples(values, 16),
        "quantize_int32": lambda: quantize_samples(values, 32),
    }
    outputs = {name: function() for name, function in cases.items()}
    if not np.array_equal(
        outputs["sample_int16"], quantize_samples(outputs["sample_float64"], 16)
    ):
        raise AssertionError("int16 direct sampling differs from float quantization")
    if not np.array_equal(
        outputs["sample_int32"], quantize_samples(outputs["sample_float64"], 32)
    ):
        raise AssertionError("int32 direct sampling differs from float quantization")

    return {
        "count": count,
        "rate": rate,
        "milliseconds": {
            name: 1e3 * best_time(function, repeats)
            for name, function in cases.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1_000_000)
    parser.add_argument("--rate", type=int, default=2_400_000_000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.count, args.rate, args.repeats), indent=2))


if __name__ == "__main__":
    main()
