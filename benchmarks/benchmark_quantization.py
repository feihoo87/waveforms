"""Measure integer sampling fast paths and their float-plus-quantize baseline."""

from __future__ import annotations

import argparse
import gc
import json
import time

import numpy as np

import waveforms as wf
from waveforms._waveform import quantize_samples


def best_time(function, repeats=7):
    function()
    timings = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        function()
        timings.append(time.perf_counter() - start)
    return min(timings)


def make_stack(event_count, varying_scales=False):
    pulse = 0.8 * wf.gaussian(20e-9) * wf.cos(2 * np.pi * 100e6)
    waves = []
    for index in range(event_count):
        scale = 0.2 + index / event_count if varying_scales else 1.0
        waves.append(scale * (pulse >> (index * 40e-9)))
    stack = wf.WaveVStack(waves) + 0.13
    stack.start = -20e-9
    stack.stop = event_count * 40e-9
    return stack


def benchmark_stack(stack, rate, repeats):
    def cold_float():
        stack._sample_plan_cache = None
        return stack.sample(rate)

    def cold_int16():
        stack._sample_plan_cache = None
        return stack.sample(rate, dtype=np.int16)

    def cold_float_then_quantize():
        stack._sample_plan_cache = None
        return quantize_samples(stack.sample(rate), 16)

    stack._sample_plan_cache = None
    float_samples = stack.sample(rate)
    expected = quantize_samples(float_samples, 16)
    actual = stack.sample(rate, dtype=np.int16)
    if not np.array_equal(actual, expected):
        raise AssertionError("int16 fast path differs from float quantization")

    return {
        "samples": len(actual),
        "cold_float_ms": 1e3 * best_time(cold_float, repeats),
        "cold_int16_ms": 1e3 * best_time(cold_int16, repeats),
        "cold_float_then_quantize_ms": 1e3 * best_time(
            cold_float_then_quantize, repeats
        ),
        "warm_float_ms": 1e3 * best_time(lambda: stack.sample(rate), repeats),
        "warm_int16_ms": 1e3 * best_time(
            lambda: stack.sample(rate, dtype=np.int16), repeats
        ),
        "warm_float_then_quantize_ms": 1e3 * best_time(
            lambda: quantize_samples(stack.sample(rate), 16), repeats
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=int, default=2_400_000_000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("events", type=int, nargs="*", default=[100, 1000, 10000])
    args = parser.parse_args()

    results = {}
    for count in args.events:
        results[f"same_scale_{count}"] = benchmark_stack(
            make_stack(count), args.rate, args.repeats
        )
        results[f"varying_scale_{count}"] = benchmark_stack(
            make_stack(count, varying_scales=True), args.rate, args.repeats
        )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
