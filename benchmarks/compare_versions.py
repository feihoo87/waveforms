"""Cross-version benchmark driver for waveform releases.

Run this file in a fresh process for each source tree.  ``--source-root`` is
inserted at the front of ``sys.path`` before importing :mod:`waveforms`, so a
single driver can benchmark archived releases and the current checkout.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import struct
import sys
import time
from pathlib import Path

import msgpack
import numpy as np


def measure(function, target=0.06, repeats=5):
    """Return the best wall time per call with an adaptive loop count."""
    gc.collect()
    start = time.perf_counter()
    function()
    elapsed = max(time.perf_counter() - start, 1e-9)
    number = max(1, min(100_000, int(target / elapsed)))
    timings = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        for _ in range(number):
            function()
        timings.append((time.perf_counter() - start) / number)
    return min(timings)


def msgpack_default(value):
    if isinstance(value, complex):
        return msgpack.ExtType(1, struct.pack("<dd", value.real, value.imag))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported msgpack value {type(value).__name__}")


def msgpack_ext_hook(code, data):
    if code == 1:
        return complex(*struct.unpack("<dd", data))
    return msgpack.ExtType(code, data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--implementation", choices=("standard", "native"),
        default="standard",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))

    import waveforms as wf

    if args.implementation == "native":
        gaussian = wf.native_gaussian
        cosine = wf.native_cos
        sine = wf.native_sin
        square_wave = wf.native_square
        stack_type = wf.NativeWaveVStack
    else:
        gaussian = wf.gaussian
        cosine = wf.cos
        sine = wf.sin
        square_wave = wf.square
        stack_type = wf.WaveVStack

    rate = 2_400_000_000

    def make_pulse():
        return (
            0.8 * gaussian(20e-9) * cosine(2 * np.pi * 100e6)
            + 0.15 * gaussian(12e-9) * sine(2 * np.pi * 180e6, 0.2)
            + 0.05 * square_wave(6e-9)
        )

    def make_template():
        return 0.8 * gaussian(20e-9) * cosine(2 * np.pi * 100e6)

    def make_stack(event_count):
        template = make_template()
        stack = stack_type(
            [template >> (index * 40e-9) for index in range(event_count)]
        )
        stack.start = -20e-9
        stack.stop = event_count * 40e-9
        stack.sample_rate = rate
        return stack

    pulse = make_pulse()
    pulse.start = -20e-9
    pulse.stop = 20e-9
    pulse.sample_rate = rate
    pulse_equivalent = make_pulse()
    pulse_equivalent.start = pulse.start
    pulse_equivalent.stop = pulse.stop
    pulse_equivalent.sample_rate = rate
    complex_wave = pulse + 0.5j * (pulse >> 2e-9)
    complex_wave.start = pulse.start
    complex_wave.stop = pulse.stop
    complex_wave.sample_rate = rate

    stacks = {count: make_stack(count) for count in (100, 1000, 10_000)}
    simple = cosine(2 * np.pi * 100e6)
    simple_x = np.linspace(-200e-6, 200e-6, 1_000_000, endpoint=False)
    pulse_x = np.linspace(-20e-9, 20e-9, 200_000, endpoint=False)

    gaussian_for_ops = gaussian(20e-9)
    carrier_for_ops = cosine(2 * np.pi * 100e6)
    metrics = {
        "construct.gaussian": measure(lambda: gaussian(20e-9)),
        "construct.add": measure(lambda: gaussian_for_ops + carrier_for_ops),
        "construct.multiply": measure(
            lambda: gaussian_for_ops * carrier_for_ops
        ),
        "construct.pulse": measure(make_pulse),
        "construct.shift": measure(lambda: pulse >> 40e-9),
        "construct.stack1000": measure(lambda: make_stack(1000)),
        "algebra.pulse_simplify": measure(pulse.simplify),
        "algebra.pulse_equal": measure(lambda: pulse == pulse_equivalent),
        "algebra.stack100_simplify": measure(stacks[100].simplify),
        "algebra.stack1000_simplify": measure(stacks[1000].simplify),
        "evaluate.cos_1m": measure(lambda: simple(simple_x)),
        "evaluate.pulse_200k": measure(lambda: pulse(pulse_x)),
        "evaluate.complex_200k": measure(lambda: complex_wave(pulse_x)),
    }

    current_integer_sampling = False
    try:
        stacks[100].sample(dtype=np.int16)
        current_integer_sampling = True
    except TypeError:
        pass

    def external_dac16(stack):
        values = stack.sample()
        return np.clip(
            np.rint(values * 32768), -32768, 32767
        ).astype(np.int16)

    for count in (1000, 10_000):
        stack = stacks[count]
        grid = np.arange(stack.start, stack.stop, 1 / rate)
        metrics[f"evaluate.stack{count}_direct"] = measure(
            lambda stack=stack, grid=grid: stack(grid)
        )
        metrics[f"sample.stack{count}_float"] = measure(stack.sample)
        if current_integer_sampling:
            metrics[f"sample.stack{count}_dac16"] = measure(
                lambda stack=stack: stack.sample(dtype=np.int16)
            )
        else:
            metrics[f"sample.stack{count}_dac16"] = measure(
                lambda stack=stack: external_dac16(stack)
            )

    serialization = {}
    objects = {
        "pulse": pulse,
        "complex": complex_wave,
        "stack100": stacks[100],
        "stack1000": stacks[1000],
        "stack10000": stacks[10_000],
    }
    for name, obj in objects.items():
        pickle_data = pickle.dumps(obj, protocol=5)
        entry = {
            "pickle_bytes": len(pickle_data),
            "pickle_dump": measure(lambda obj=obj: pickle.dumps(obj, protocol=5)),
            "pickle_load": measure(lambda data=pickle_data: pickle.loads(data)),
        }
        if hasattr(obj, "to_bytes"):
            native_data = obj.to_bytes()
            object_type = type(obj)
            entry.update({
                "native_kind": "packed-block",
                "native_bytes": len(native_data),
                "native_dump": measure(obj.to_bytes),
                "native_load": measure(
                    lambda data=native_data, object_type=object_type:
                    object_type.from_bytes(data)
                ),
            })
        else:
            object_type = type(obj)

            def native_dump(obj=obj):
                return msgpack.packb(
                    obj.tolist(), use_bin_type=True, default=msgpack_default
                )

            native_data = native_dump()

            def native_load(data=native_data, object_type=object_type):
                payload = msgpack.unpackb(
                    data, raw=False, ext_hook=msgpack_ext_hook
                )
                return object_type.fromlist(payload)

            entry.update({
                "native_kind": "msgpack-list",
                "native_bytes": len(native_data),
                "native_dump": measure(native_dump),
                "native_load": measure(native_load),
            })
        serialization[name] = entry

    result = {
        "label": args.label,
        "reported_version": wf.__version__,
        "implementation": args.implementation,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "integer_sampling": current_integer_sampling,
        "metrics_seconds": metrics,
        "serialization": serialization,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
