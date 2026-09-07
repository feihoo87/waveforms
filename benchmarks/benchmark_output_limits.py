"""Compare native final limits with float sampling, clipping and quantization.

Use --reference-only --source-root PATH to measure a pre-change installation
without relying on its (incorrect) built-in limit semantics.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))

    import numpy as np
    import waveforms as wf
    from waveforms._waveform import quantize_samples

    def measure(function):
        function()
        start = time.perf_counter()
        function()
        elapsed = time.perf_counter() - start
        loops = max(1, min(1000, int(.015 / max(elapsed, 1e-9))))
        timings = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            for _ in range(loops):
                function()
            timings.append((time.perf_counter() - start) * 1000 / loops)
        return round(statistics.median(timings), 6)

    result = {"python": sys.version.split()[0], "numpy": np.__version__,
              "events": args.events, "source": str(Path(wf.__file__).parent),
              "warm_median_ms": {}}
    for name, spacing, scales in (
        ("repeated", 40e-9, np.ones(args.events)),
        ("many_scales", 40e-9, np.linspace(-.9, .9, args.events)),
        ("overlapping", 2e-9, np.resize([.8, -.4, .6], args.events)),
    ):
        stack = wf.WaveVStack.from_events(
            [.8 * wf.gaussian(20e-9)], np.zeros(args.events, dtype=np.uint32),
            np.arange(args.events, dtype=np.int64) * wf.time_to_tick(spacing),
            scales)
        stack.start, stack.stop = -30e-9, args.events * spacing + 30e-9
        stack.sample_rate = 2_400_000_000
        limited = wf.WaveVStack.from_bytes(stack.to_bytes())
        limited.start, limited.stop = stack.start, stack.stop
        limited.sample_rate = stack.sample_rate
        limited.min, limited.max = -.17, .23

        def reference():
            values = stack.sample()
            np.clip(values, -.17, .23, out=values)
            return quantize_samples(values, 16)

        functions = {
            "unlimited_float": lambda: stack.sample(),
            "unlimited_int16": lambda: stack.sample(dtype=np.int16),
            "float_clip_quantize_reference": reference,
        }
        if not args.reference_only:
            expected = reference()
            np.testing.assert_array_equal(limited.sample(dtype=np.int16), expected)
            np.testing.assert_array_equal(
                limited.sample(), np.clip(stack.sample(), -.17, .23))
            functions.update({
                "limited_float": lambda: limited.sample(),
                "limited_int16": lambda: limited.sample(dtype=np.int16),
            })
        result["warm_median_ms"][name] = {
            label: measure(function) for label, function in functions.items()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
