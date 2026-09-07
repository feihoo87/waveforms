# Native SOS filtering and final output

Measured on 2026-09-05, macOS arm64, Python 3.11.13, NumPy 2.2.4,
SciPy 1.13.1, release C extension (`clang -O3`). The comparison uses the
working implementation immediately before SOS integration, which already
applies output limits after filtering. Both versions use the same public API
and produce the same int16/int32 samples in these cases.

`benchmark_sos_filter.py` builds 10,000 repeated Gaussian events, sampling
960,144 points at 2.4 GHz. Butterworth SOS cascades use cutoff 0.15,
`initial=0.05`, and final limits `[-0.17, 0.23]`. Each measurement is a warm
median of 11 timing batches; state is reset for every complete sample call.
Baseline and native measurements ran sequentially, without concurrent tests
or compilation. The benchmark checks float values against SciPy within
`rtol=3e-13, atol=3e-14` and checks integer output exactly before timing.

| SOS sections | Output | Before (ms) | Native (ms) | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 1 | float64 | 11.717 | 6.029 | 1.94× |
| 1 | int16 | 12.325 | 6.453 | 1.91× |
| 1 | int32 | 12.652 | 6.878 | 1.84× |
| 4 | float64 | 17.713 | 14.620 | 1.21× |
| 4 | int16 | 18.267 | 15.294 | 1.19× |
| 4 | int32 | 18.483 | 15.314 | 1.21× |
| 8 | float64 | 25.289 | 21.317 | 1.19× |
| 8 | int16 | 26.079 | 21.824 | 1.20× |
| 8 | int32 | 26.618 | 21.877 | 1.22× |

With a preallocated int16 output, native medians were 6.523 / 14.959 / 21.385
ms for 1 / 4 / 8 sections, versus 12.117 / 17.664 / 26.032 ms before.
Chunked sampling into that output, with chunks of 16,384 samples, changed from
21.061 / 27.587 / 37.559 ms to 16.559 / 25.573 / 33.194 ms. Chunk timings also
include the existing waveform evaluation work in every chunk. Unfiltered
int16 sampling was 0.189 ms before and 0.185 ms after; its native path is
unchanged. These measurements are specific to this workload and machine.

## Why this helps

SciPy's `sosfilt` is already compiled code. The improvement comes from merging
baseline subtraction, SOS recurrence, baseline restoration, final limits and
real integer conversion into one C stage. It reads the sampled input and
writes final output using a 256-double scratch buffer. Float output can reuse
the sampled array; integer output does not need a full filtered float array.
The raw waveform buffer and the existing nonlinear mapping stage remain.

One-section filters keep their coefficients and delays in registers.
Multi-section cascades advance all sections for each sample. A preliminary
section-first implementation performed worse for long cascades because the
recurrence latency could not overlap between sections; it is not used.
As section count grows, recurrence work dominates and the relative benefit
of removing array passes decreases.

Real SOS coefficients support real and complex signals, whole/chunked
sampling, output buffers, and the same final amplitude limits. The delays
are saved before clipping so clipping cannot feed back into the filter.
Complex coefficients and extended precision retain SciPy. I/Q splitting and
quantization in `sample_iq()` remain a subsequent operation; unusual float
output dtypes/layouts still require their requested cast or copy.

## Reproduce

Build the extension with the Python used for measurement, then run:

```sh
python3.11 setup.py build_ext --inplace
python3.11 benchmarks/benchmark_sos_filter.py --repeats 11
python3.11 benchmarks/benchmark_sos_filter.py --repeats 11 --source-root /path/to/baseline
```

The baseline path must contain an importable `waveforms` package with a
matching compiled extension. Smaller `--events` values exercise shorter
buffers. Timing alone does not establish numerical compatibility:
`tests/test_native_sos.py` also exercises initial delays, up to 16 sections,
complex signals, changing chunk boundaries, output aliasing, readonly
coefficients, invalid inputs, output casting, and the SciPy fallback.

Validation: 186 tests passed in the full suite; the 30 native SOS cases passed
again after the final empty-input/error handling adjustment. A standalone C
harness also passed AddressSanitizer and UndefinedBehaviorSanitizer checks
for 1/2/4/8 sections, strided/in-place output, chunked int16/int32 conversion,
state continuation, empty input, and invalid bounds.
