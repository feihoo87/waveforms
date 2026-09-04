"""Benchmark native post-sampling nonlinear transfer maps."""

from __future__ import annotations

import argparse
import gc
import json
import time

import numpy as np
from scipy.interpolate import PchipInterpolator

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


def benchmark(count, rate, table_size, repeats):
    calibration_x = np.linspace(-1.0, 1.0, 65)
    calibration_y = np.tanh(1.7 * calibration_x)
    table_x = np.linspace(-1.0, 1.0, table_size)
    table_y = np.interp(table_x, calibration_x, calibration_y)
    scipy_curve = PchipInterpolator(calibration_x, calibration_y)
    linear = wf.NonlinearMap.from_samples(
        calibration_x, calibration_y, method="linear",
        table_size=table_size,
    )
    cubic = wf.NonlinearMap.from_samples(
        calibration_x, calibration_y, method="monotone_cubic",
        table_size=table_size,
    )
    values = 0.99 * np.sin(np.linspace(-100.0, 100.0, count))

    duration = count / rate
    waveform = 0.99 * wf.sin(2 * np.pi * 100e6)
    waveform.start = -duration / 2
    waveform.stop = waveform.start + duration
    waveform.sample_rate = rate

    cases = {
        "native_linear_apply": lambda: linear(values),
        "numpy_interp": lambda: np.interp(values, table_x, table_y),
        "native_cubic_apply": lambda: cubic(values),
        "native_linear_int16": lambda: linear._apply_quantized(values, 16),
        "native_cubic_int16": lambda: cubic._apply_quantized(values, 16),
        "scipy_pchip": lambda: scipy_curve(values),
        "waveform_sample": lambda: waveform.sample(),
        "waveform_sample_linear": lambda: waveform.sample(nonlinear=linear),
        "waveform_sample_cubic": lambda: waveform.sample(nonlinear=cubic),
        "waveform_sample_cubic_int16": lambda: waveform.sample(
            nonlinear=cubic, dtype=np.int16),
    }
    outputs = {name: function() for name, function in cases.items()}
    if np.max(np.abs(outputs["native_linear_apply"]
                     - outputs["numpy_interp"])) > 5e-15:
        raise AssertionError("native linear map differs from numpy.interp")
    if np.max(np.abs(outputs["native_cubic_apply"]
                     - outputs["scipy_pchip"])) > 2e-6:
        raise AssertionError("native cubic map exceeds reference error")
    if not np.array_equal(
        outputs["native_linear_int16"],
        quantize_samples(outputs["native_linear_apply"], 16),
    ):
        raise AssertionError("native linear int16 map differs from quantization")
    if not np.array_equal(
        outputs["native_cubic_int16"],
        quantize_samples(outputs["native_cubic_apply"], 16),
    ):
        raise AssertionError("native cubic int16 map differs from quantization")

    return {
        "count": count,
        "rate": rate,
        "table_size": table_size,
        "serialized_bytes": {
            "linear": len(linear.to_bytes()),
            "cubic": len(cubic.to_bytes()),
        },
        "milliseconds": {
            name: 1e3 * best_time(function, repeats)
            for name, function in cases.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1_000_000)
    parser.add_argument("--rate", type=int, default=2_400_000_000)
    parser.add_argument("--table-size", type=int, default=4097)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(benchmark(
        args.count, args.rate, args.table_size, args.repeats,
    ), indent=2))


if __name__ == "__main__":
    main()
