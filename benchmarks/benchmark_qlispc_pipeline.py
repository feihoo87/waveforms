"""Benchmark the waveform lifecycle produced by qlispc.

The benchmark mirrors the production split:

* macOS/Windows constructs channel stacks and serializes them;
* Linux restores the stack, applies the final delay calibration, and samples;
* a controller without a Linux device performs the same final sampling locally.

The pulse shapes and timing follow ``qlispc.libs.gates``.  qlispc itself is
deliberately not imported so this benchmark can run on every wheel builder.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import statistics
import time

import numpy as np

import waveforms as wf


def measure(function, *, repeat=5, warmup=1):
    for _ in range(warmup):
        function()
    gc.disable()
    try:
        samples = []
        for _ in range(repeat):
            start = time.perf_counter_ns()
            function()
            samples.append(time.perf_counter_ns() - start)
    finally:
        gc.enable()
    return {
        "best_ms": min(samples) / 1e6,
        "median_ms": statistics.median(samples) / 1e6,
    }


def timed_value(function):
    start = time.perf_counter_ns()
    value = function()
    return value, (time.perf_counter_ns() - start) / 1e6


def make_r_stack(kind, event_count, duration, sample_rate):
    constructor = {
        "drag": wf.drag,
        "drag_sin": wf.drag_sin,
        "drag_sinx": wf.drag_sinx,
    }[kind]
    width = 20e-9
    waves = [
        0.3 * constructor(
            5e9,
            width,
            delta=-200e6,
            block_freq=80e6,
            phase=(index * 0.371) % (2 * np.pi),
            t0=index * width,
        )
        for index in range(event_count)
    ]
    stack = wf.WaveVStack(waves)
    stack.start = 0.0
    stack.stop = duration
    stack.sample_rate = sample_rate
    return stack


def make_flux_stack(event_count, duration, sample_rate):
    # CZ/iSWAP reconstruct this calibrated shape for every gate.  Do not
    # manually reuse the Python object: discovering that reuse is the job of
    # WaveVStack and is one of the production invariants measured here.
    waves = [
        0.2 * wf.coshPulse(20e-9, plateau=20e-9, eps=4.0)
        >> (index * 50e-9)
        for index in range(event_count)
    ]
    stack = wf.WaveVStack(waves)
    stack.start = 0.0
    stack.stop = duration
    stack.sample_rate = sample_rate
    return stack


def make_flux_stack_batched(event_count, duration, sample_rate):
    """Compiler-native form: one template plus columnar integer-tick events."""
    template = wf.coshPulse(20e-9, plateau=20e-9, eps=4.0)
    delay_step = wf.time_to_tick(50e-9)
    stack = wf.WaveVStack.from_events(
        (template,),
        np.zeros(event_count, dtype=np.uint32),
        np.arange(event_count, dtype=np.int64) * delay_step,
        np.full(event_count, 0.2),
    )
    stack.start = 0.0
    stack.stop = duration
    stack.sample_rate = sample_rate
    return stack


def core_counts(stack):
    core = stack._core
    return {
        "events": int(core.event_count),
        "templates": int(core.template_count),
    }


def benchmark_stack(name, factory, sample_rate, shift, repeat):
    stack, construct_ms = timed_value(factory)
    result = {
        "construct_ms": construct_ms,
        **core_counts(stack),
        "begin": measure(lambda: stack.begin, repeat=repeat),
        "end": measure(lambda: stack.end, repeat=repeat),
    }

    block, core_dump_ms = timed_value(stack.to_bytes)
    payload, pickle_dump_ms = timed_value(
        lambda: pickle.dumps(stack, protocol=5)
    )
    result.update({
        "core_bytes": len(block),
        "core_dump_ms": core_dump_ms,
        "pickle_bytes": len(payload),
        "pickle_dump_ms": pickle_dump_ms,
    })

    restored, pickle_load_ms = timed_value(lambda: pickle.loads(payload))
    shifted, shift_ms = timed_value(lambda: restored >> shift)
    result["pickle_load_ms"] = pickle_load_ms
    result["shift_ms"] = shift_ms

    # Production normally samples a transported object once.  Recreate the
    # object for every cold measurement so an implicit plan cache cannot hide
    # plan-construction cost.
    def cold_sample(dtype=None):
        candidate = pickle.loads(payload) >> shift
        return candidate.sample(sample_rate=sample_rate, dtype=dtype)

    result["cold_float"] = measure(
        cold_sample, repeat=max(1, min(3, repeat)), warmup=0
    )
    result["cold_int16"] = measure(
        lambda: cold_sample(np.int16),
        repeat=max(1, min(3, repeat)), warmup=0,
    )
    shifted.sample(sample_rate=sample_rate)
    result["warm_float"] = measure(
        lambda: shifted.sample(sample_rate=sample_rate), repeat=repeat
    )
    result["warm_int16"] = measure(
        lambda: shifted.sample(sample_rate=sample_rate, dtype=np.int16),
        repeat=repeat,
    )
    result["sample_count"] = len(
        shifted.sample(sample_rate=sample_rate, dtype=np.int16)
    )
    return name, result


def parse_rates(value):
    return tuple(int(float(item)) for item in value.split(","))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r-events", type=int, default=1_000)
    parser.add_argument("--flux-events", type=int, default=10_000)
    parser.add_argument("--duration", type=float, default=98e-6)
    parser.add_argument("--rates", type=parse_rates, default=(2_400_000_000,))
    parser.add_argument("--shift", type=float, default=2e-9)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    report = {
        "version": wf.__version__,
        "duration": args.duration,
        "shift": args.shift,
        "rates": {},
    }
    for rate in args.rates:
        cases = {}
        factories = [
            (
                kind,
                lambda kind=kind, rate=rate: make_r_stack(
                    kind, args.r_events, args.duration, rate
                ),
            )
            for kind in ("drag", "drag_sin", "drag_sinx")
        ]
        factories.append((
            "cz_iswap",
            lambda rate=rate: make_flux_stack(
                args.flux_events, args.duration, rate
            ),
        ))
        factories.append((
            "cz_iswap_batched",
            lambda rate=rate: make_flux_stack_batched(
                args.flux_events, args.duration, rate
            ),
        ))
        for name, factory in factories:
            case_name, result = benchmark_stack(
                name, factory, rate, args.shift, args.repeat
            )
            cases[case_name] = result
        report["rates"][str(rate)] = cases
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
