"""Compare device-grid/template sampling with the generic stack evaluator."""

from __future__ import annotations

import argparse
import gc
import time

import numpy as np

import waveforms as wf
from waveforms._waveform import (
    quantize_samples,
    sample_clock,
    sample_grid,
    time_to_tick,
)


def best_time(function, repeats):
    measurements = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        function()
        measurements.append(time.perf_counter() - start)
    return min(measurements)


def benchmark(event_count, sample_rate, repeats):
    pulse = 0.8 * wf.gaussian(20e-9) * wf.cos(2 * np.pi * 100e6)
    stack = wf.WaveVStack(
        pulse >> (index * 40e-9) for index in range(event_count)
    )
    stack.start = -20e-9
    stack.stop = event_count * 40e-9

    step_numerator, step_denominator = sample_clock(sample_rate)
    start_tick = time_to_tick(stack.start)
    stop_tick = time_to_tick(stack.stop)
    count = (
        (stop_tick - start_tick) * step_denominator
        + step_numerator - 1
    ) // step_numerator
    grid = sample_grid(
        start_tick, count, step_numerator, step_denominator
    )

    fast = best_time(
        lambda: stack.sample(sample_rate, dtype=np.int16), repeats
    )
    generic = best_time(
        lambda: quantize_samples(stack(grid), 16), repeats
    )
    print(
        f"{event_count:>7} events  {len(grid):>9} samples  "
        f"fast={fast * 1e3:>8.3f} ms  "
        f"generic={generic * 1e3:>8.3f} ms  "
        f"speedup={generic / fast:>5.2f}x"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=int, default=2_400_000_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("events", type=int, nargs="*", default=[100, 1000, 10000])
    args = parser.parse_args()
    for event_count in args.events:
        benchmark(event_count, args.rate, args.repeats)


if __name__ == "__main__":
    main()
