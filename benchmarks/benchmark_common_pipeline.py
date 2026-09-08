"""Portable cold-load / device-sampling benchmark, without ISA tuning.

Run against separately built source trees using --source-root. Every cold
sample restores a fresh object; warm timings intentionally reuse its plan.
JSON output includes sample digests to compare numerics across builds.
"""

import argparse
import gc
import hashlib
import json
import platform
import pickle
from pathlib import Path
import statistics
import sys
import time


def measure(function, repeats):
    function()
    start = time.perf_counter_ns()
    function()
    elapsed = max(1, time.perf_counter_ns() - start)
    loops = max(1, min(2000, int(20_000_000 / elapsed)))
    timings = []
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter_ns()
            for _ in range(loops):
                function()
            timings.append((time.perf_counter_ns() - start) / loops / 1e6)
    finally:
        if enabled:
            gc.enable()
    return round(statistics.median(timings), 6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument('--repeats', type=int, default=9)
    parser.add_argument('--samples-dir', type=Path,
                        help='Optionally save sample arrays for cross-build error checks')
    args = parser.parse_args()
    if args.samples_dir is not None:
        args.samples_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_root.resolve()))
    import numpy as np
    from scipy.signal import butter
    import waveforms as wf
    from waveforms._waveform import CWaveformStackCore

    rate = 2_400_000_000
    cases = {}
    construction = {}
    # Cover repeated/unique templates, dense/sparse/overlapping destinations,
    # and more than 64 scales (which disables scale-template reuse).
    pulse = .4 * wf.drag(5e9, 20e-9, delta=-200e6, block_freq=80e6)
    for name, events, spacing, templates, varied in (
        ('drag_1000', 1000, 40e-9, [pulse], False),
        ('drag_unique_1000', 1000, 40e-9,
         [.4 * wf.drag(5e9, 20e-9, delta=-200e6, block_freq=80e6,
                      phase=index * .371) for index in range(1000)], False),
        ('flux_dense_10000', 10000, 40e-9,
         [.3 * wf.coshPulse(20e-9, plateau=20e-9, eps=4)], False),
        ('flux_sparse_10000', 10000, 160e-9,
         [.3 * wf.coshPulse(20e-9, plateau=20e-9, eps=4)], False),
        ('overlap_10000', 10000, 10e-9, [.3 * wf.gaussian(20e-9)], False),
        ('many_scales_10000', 10000, 40e-9,
         [.3 * wf.gaussian(20e-9)], True),
    ):
        ids = np.arange(events, dtype=np.uint32) % len(templates)
        delays = np.arange(events, dtype=np.int64) * wf.time_to_tick(spacing)
        scales = (np.linspace(-.9, .9, events) if varied else
                  np.resize([.5, -.75, 1.0], events))
        stack = wf.WaveVStack.from_events(templates, ids, delays, scales)
        stack.start, stack.stop = -30e-9, (events - 1) * spacing + 30e-9
        stack.sample_rate = rate
        cases[name] = stack
        construction[name] = measure(
            lambda: wf.WaveVStack.from_events(templates, ids, delays, scales),
            args.repeats)

    report = {'platform': platform.platform(), 'python': sys.version.split()[0],
              'numpy': np.__version__, 'version': wf.__version__,
              'source': str(Path(wf.__file__).parent), 'rate': rate, 'cases': {}}
    for name, stack in cases.items():
        payload = pickle.dumps(stack, protocol=5)
        block = stack.to_bytes()
        shifted = pickle.loads(payload) >> 2e-9
        result = {'pickle_bytes': len(payload), 'core_bytes': len(block),
                  'pickle_sha256': hashlib.sha256(payload).hexdigest(),
                  'core_sha256': hashlib.sha256(block).hexdigest(),
                  'ms': {}, 'sha256': {}}
        timings = result['ms']
        timings['construct_events'] = construction[name]
        timings['loads_lazy'] = measure(lambda: pickle.loads(payload), args.repeats)

        def decode():
            restored = CWaveformStackCore.from_bytes(block)
            return restored.lower_tick, restored.upper_tick

        timings['decode_native'] = measure(decode, args.repeats)

        def decode_and_hash():
            restored = CWaveformStackCore.from_bytes(block)
            return restored.lower_tick, restored.upper_tick, restored.hash64

        timings['decode_and_hash'] = measure(decode_and_hash, args.repeats)
        sample_count = len(shifted.sample())
        timings['prepare_plan'] = measure(
            lambda: stack._core.prepare_sample(
                wf.time_to_tick(stack.start), sample_count, 50, 1,
                wf.time_to_tick(2e-9)), args.repeats)
        # Explicitly include restore, final calibration and first plan build.
        for dtype in (np.float64, np.int16, np.int32):
            label = np.dtype(dtype).name
            values = shifted.sample(dtype=dtype)
            output = np.empty_like(values)
            result['samples'] = len(values)
            result['sha256'][label] = hashlib.sha256(values.tobytes()).hexdigest()
            if args.samples_dir is not None:
                np.save(args.samples_dir / f'{name}-{label}.npy', values)
            timings['cold_' + label] = measure(
                lambda: (pickle.loads(payload) >> 2e-9).sample(dtype=dtype),
                args.repeats)
            timings['warm_' + label] = measure(
                lambda: shifted.sample(dtype=dtype), args.repeats)
            timings['warm_out_' + label] = measure(
                lambda: shifted.sample(dtype=dtype, out=output), args.repeats)
            np.testing.assert_array_equal(output, values)
        shifted.nonlinear = wf.NonlinearMap.from_samples(
            np.linspace(-2, 2, 1025), np.linspace(-2, 2, 1025) ** 3 / 4,
        )
        stack.nonlinear = shifted.nonlinear
        nonlinear_payload = pickle.dumps(stack, protocol=5)
        nonlinear_values = shifted.sample(dtype=np.int16)
        result['sha256']['nonlinear_int16'] = hashlib.sha256(
            nonlinear_values.tobytes()).hexdigest()
        timings['nonlinear_int16'] = measure(
            lambda: shifted.sample(dtype=np.int16), args.repeats)
        timings['cold_nonlinear_int16'] = measure(
            lambda: (pickle.loads(nonlinear_payload) >> 2e-9).sample(dtype=np.int16),
            args.repeats)
        shifted.filters = butter(4, .2, output='sos'), .03
        stack.filters = shifted.filters
        processed_payload = pickle.dumps(stack, protocol=5)
        values = shifted.sample(dtype=np.int16)
        result['sha256']['processed_int16'] = hashlib.sha256(values.tobytes()).hexdigest()
        if args.samples_dir is not None:
            np.save(args.samples_dir / f'{name}-nonlinear_int16.npy', nonlinear_values)
            np.save(args.samples_dir / f'{name}-processed_int16.npy', values)
        timings['processed_int16'] = measure(
            lambda: shifted.sample(dtype=np.int16), args.repeats)
        timings['cold_processed_int16'] = measure(
            lambda: (pickle.loads(processed_payload) >> 2e-9).sample(dtype=np.int16),
            args.repeats)
        report['cases'][name] = result
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
