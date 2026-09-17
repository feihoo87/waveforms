"""Measure the complete filtered sampling pipeline, including limits and DAC output.

Use --source-root with a pre-change package to compare equivalent inverse
filters and baselines across the old SOS and new parameter-mapping APIs.
Every timed configuration is checked against a SciPy reference first.
"""

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import platform
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))

    import numpy as np
    import scipy
    from scipy.signal import sosfilt

    import waveforms as wf
    from waveforms._waveform import quantize_samples

    def measure(function):
        function()
        start = time.perf_counter()
        function()
        elapsed = time.perf_counter() - start
        loops = max(1, min(1000, int(.02 / max(elapsed, 1e-9))))
        timings = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            for _ in range(loops):
                function()
            timings.append((time.perf_counter() - start) * 1000 / loops)
        return round(statistics.median(timings), 6)

    raw = wf.WaveVStack.from_events(
        [.8 * wf.gaussian(20e-9)], np.zeros(args.events, dtype=np.uint32),
        np.arange(args.events, dtype=np.int64) * wf.time_to_tick(40e-9),
        np.ones(args.events))
    raw.start, raw.stop = -30e-9, args.events * 40e-9 + 30e-9
    raw.sample_rate = 2_400_000_000
    wave = wf.WaveVStack.from_bytes(raw.to_bytes())
    wave.start, wave.stop, wave.sample_rate = raw.start, raw.stop, raw.sample_rate
    wave.min, wave.max = -.17, .23
    raw_values = raw.sample()
    target = np.empty(len(raw_values), dtype=np.int16)
    result = {"python": sys.version.split()[0], "numpy": np.__version__,
              "scipy": scipy.__version__, "machine": platform.machine(),
              "events": args.events, "samples": len(raw_values),
              "source": str(Path(wf.__file__).parent),
              "unfiltered_int16_ms": measure(lambda: raw.sample(dtype=np.int16)),
              "warm_median_ms": {}}
    for sections in (1, 4, 8):
        params = dict(zip(np.geomspace(1e-9, 1e-5, sections * 2),
                          np.resize([.02, -.01], sections * 2)))
        amp, tau = wf.exp_decay_filter_from_cascade(
            [(amp, tau) for tau, amp in params.items()])
        sos = wf.exp_decay_filter(amp, tau, raw.sample_rate, inv=True, output="sos")
        initial = raw_values[0]
        # Keep cross-version comparisons on identical coefficients/baselines.
        wave.filters = params if isinstance(wave.filters, Mapping) else (sos, initial)
        expected = np.clip(sosfilt(sos, raw_values - initial) + initial, -.17, .23)
        expected16 = quantize_samples(expected, 16)

        def chunks():
            for _ in wave.sample(chunk_size=16_384, out=target):
                pass
            return target

        functions = {
            "float64": lambda: wave.sample(),
            "int16": lambda: wave.sample(dtype=np.int16),
            "int32": lambda: wave.sample(dtype=np.int32),
            "int16_out": lambda: wave.sample(out=target),
            "int16_chunk16384": chunks,
        }
        np.testing.assert_allclose(functions["float64"](), expected,
                                   rtol=3e-13, atol=3e-14)
        np.testing.assert_array_equal(functions["int16"](), expected16)
        np.testing.assert_array_equal(functions["int16_out"](), expected16)
        np.testing.assert_array_equal(functions["int32"](), quantize_samples(expected, 32))
        np.testing.assert_array_equal(chunks(), expected16)
        result["warm_median_ms"][str(sections)] = {
            label: measure(function) for label, function in functions.items()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
