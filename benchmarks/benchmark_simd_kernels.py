"""Measure individual native SIMD kernels without Python call overhead per batch."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import statistics
import sys
import time


def measure(call):
    assert call(1) == 0
    start = time.perf_counter_ns()
    call(10)
    ns = (time.perf_counter_ns() - start) / 10
    loops = max(1, min(100000, int(5e6 / max(ns, 1))))
    samples = []
    for _ in range(7):
        start = time.perf_counter_ns()
        assert call(loops) == 0
        samples.append((time.perf_counter_ns() - start) / loops)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--counts', default='64,4096,65536,1000000')
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    import numpy as np
    import waveforms as wf
    lib = ct.CDLL(str(args.library.resolve()))
    pointer, size, integer, real = ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_double
    lib.wf_probe_quantize.argtypes = [pointer, size, integer, real, pointer, integer, size]
    lib.wf_probe_nonlinear.argtypes = [pointer, pointer, size, integer, real, pointer, integer, size]
    lib.cwaveform_nonlinear_map_from_bytes.argtypes = [pointer, size]
    lib.cwaveform_nonlinear_map_from_bytes.restype = pointer
    lib.cwaveform_nonlinear_map_release.argtypes = [pointer]
    result = {'python': sys.version, 'library': str(args.library), 'ns_per_sample': {}}
    rng = np.random.default_rng(932)
    for count in map(int, args.counts.split(',')):
        values = .99 * np.sin(np.linspace(-100, 100, count))
        for bits in (16, 32):
            output = np.empty(count, dtype=np.int16 if bits == 16 else np.int32)
            expected = None
            for mode, name in ((1, 'scalar'), (2, 'avx2'), (3, 'avx512')):
                call = lambda loops: lib.wf_probe_quantize(values.ctypes.data, count, bits, 1., output.ctypes.data, mode, loops)
                timing = measure(call) / count
                if expected is None:
                    expected = output.copy()
                np.testing.assert_array_equal(output, expected)
                result['ns_per_sample'][f'quantize{bits}/{count}/{name}'] = timing
        for method in ('linear', 'monotone_cubic'):
            for points in (257, 4097):
                x = np.linspace(-1., 1., 65)
                mapping = wf.NonlinearMap.from_samples(x, np.tanh(x * 1.7), method=method, table_size=points)
                block = mapping.to_bytes()
                handle = lib.cwaveform_nonlinear_map_from_bytes(block, len(block))
                assert handle
                try:
                    for pattern in ('smooth', 'random'):
                        values = (.99 * np.sin(np.linspace(-100, 100, count))
                                  if pattern == 'smooth' else rng.uniform(-.99, .99, count))
                        output = np.empty(count)
                        expected = None
                        for mode, name in ((1, 'scalar'), (2, 'avx2'), (3, 'avx512')):
                            call = lambda loops: lib.wf_probe_nonlinear(handle, values.ctypes.data, count, 0, 1., output.ctypes.data, mode, loops)
                            timing = measure(call) / count
                            if expected is None:
                                expected = output.copy()
                            np.testing.assert_allclose(output, expected, rtol=2e-15, atol=2e-15)
                            result['ns_per_sample'][f'{method}/{points}/{pattern}/{count}/{name}'] = timing
                finally:
                    lib.cwaveform_nonlinear_map_release(handle)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
