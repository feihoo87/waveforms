# Native sample-plan benchmark report

Date: 2026-08-28. Machine: Apple arm64. Runtime: CPython 3.11.13,
NumPy 2.3.1. Timings are the best of five adaptive runs from
`benchmarks/compare_versions.py`; lower is better.

The baselines are the `v3.1.0` tag and the exact 3.1.1 release commit
`9d15f7d`. The candidate is version 3.2.0, which combines the 3.1.2 native C
core with the native sample-plan handle. A cold timing clears the per-stack plan cache
before every call. A warm timing reuses it. Version 3.1.0 has no plan cache, so
its cold and warm numbers are effectively the same. It also has no direct
integer sampling API, so its int16 rows include float sampling followed by the
equivalent external NumPy quantization.

## Repeated, non-overlapping pulses

| Operation | 3.1.0 | 3.1.1 | 3.2.0 |
| --- | ---: | ---: | ---: |
| 1K float, cold | 6,002 µs | 695 µs | **40.6 µs** |
| 1K float, warm | 6,049 µs | 33.3 µs | **31.8 µs** |
| 1K int16, cold | 6,263 µs | 686 µs | **25.2 µs** |
| 1K int16, warm | 6,339 µs | 23.3 µs | **15.8 µs** |
| 10K float, cold | 61.62 ms | 7.296 ms | **0.824 ms** |
| 10K float, warm | 62.33 ms | 0.671 ms | **0.619 ms** |
| 10K int16, cold | 64.65 ms | 6.671 ms | **0.259 ms** |
| 10K int16, warm | 64.93 ms | **0.155 ms** | 0.172 ms |

Compared with 3.1.1, native plan creation is 17–27 times faster for 1K
events and 9–26 times faster for 10K events. Warm float sampling is 5% faster
at 1K and 8% faster at 10K. Warm int16 is 47% faster at 1K; the 10K result is
in the same memory-bandwidth-limited band and was 11% slower in this
run.

Compared with the direct native sampler committed in 3.1.2, the handle changes
warm 10K float sampling from 13.60 ms to 0.619 ms and int16 from 13.99 ms to
0.172 ms. The sampled template, placement destinations, scale groups, and
overlap decision now live in one immutable C-owned object and are no longer
rebuilt as Python/NumPy structures.

## Overlap and multiple amplitude scales

| Operation | 3.1.0 | 3.1.1 | 3.2.0 |
| --- | ---: | ---: | ---: |
| Overlap 1K float, cold | 5,865 µs | 688 µs | **34.9 µs** |
| Overlap 1K float, warm | 5,924 µs | **22.9 µs** | 25.5 µs |
| Overlap 1K int16, warm | 5,955 µs | 82.1 µs | **55.4 µs** |
| Overlap 10K float, cold | 59.45 ms | 6.811 ms | **0.369 ms** |
| Overlap 10K float, warm | 59.53 ms | **0.248 ms** | 0.266 ms |
| Overlap 10K int16, warm | 61.21 ms | 0.589 ms | **0.574 ms** |
| Three scales, 1K float, warm | 6,047 µs | 34.3 µs | **34.0 µs** |
| Three scales, 1K int16, warm | 6,390 µs | 33.3 µs | **16.5 µs** |

The remaining exception is warm float accumulation with overlapping pulses:
the Cython 3.1.1 loop is 7–11% faster. The native handle
still wins the cold path by roughly 17–20 times and wins quantized sampling,
because it evaluates each template phase once and quantizes each distinct
scale once. Overlapping integer output accumulates in a temporary float buffer
before quantization, preserving the original numerical semantics.

## Construction, algebra, and direct evaluation

These operations use the 3.1.2 C waveform core and are independent of whether
the sample-plan cache is warm.

| Operation | 3.1.0 | 3.1.1 | 3.2.0 |
| --- | ---: | ---: | ---: |
| Composite pulse construction | 274.5 µs | 79.1 µs | **10.8 µs** |
| 1K stack construction | 1.069 ms | 1.074 ms | **0.811 ms** |
| Pulse equality | 78.9 µs | 56.9 µs | **0.405 µs** |
| Pulse simplify | 39.6 µs | 28.5 µs | **0.429 µs** |
| 1K stack simplify | 4.473 ms | 5.512 ms | **0.204 ms** |
| Cosine, 1M positions | 5.861 ms | 5.641 ms | **3.425 ms** |
| Pulse, 200K positions | **1.939 ms** | 1.966 ms | 2.011 ms |
| Complex pulse, 200K positions | 4.587 ms | **4.012 ms** | 4.232 ms |
| Stack 1K, direct evaluation | 6.246 ms | 5.269 ms | **1.243 ms** |
| Stack 10K, direct evaluation | 64.39 ms | 55.59 ms | **13.09 ms** |

The isolated real and complex pulse evaluations are the non-sampling
exceptions in this run: they are within 2–6% of 3.1.1. Construction, equality,
simplification, cosine evaluation, and direct stack evaluation all beat both
baselines.

## Serialization

The sample plan is an ephemeral cache and is deliberately not serialized.
Therefore these sizes are the compact WNF4/WNS4 sizes already committed in
3.1.2.

| Native block size | 3.1.0 | 3.1.1 | 3.2.0 |
| --- | ---: | ---: | ---: |
| Pulse | 490 B | 484 B | **360 B** |
| Complex pulse | 2,001 B | 980 B | **760 B** |
| Stack 100 | 2,981 B | 2,180 B | **2,128 B** |
| Stack 1K | 28,181 B | 20,180 B | **20,128 B** |
| Stack 10K | 280,181 B | 200,180 B | **200,128 B** |

| Native load | 3.1.0 | 3.1.1 | 3.2.0 |
| --- | ---: | ---: | ---: |
| Pulse | 4.102 µs | 4.126 µs | **1.115 µs** |
| Complex pulse | 4.110 µs | 9.266 µs | **2.848 µs** |
| Stack 1K | 4.180 µs | 4.298 µs | **0.357 µs** |
| Stack 10K | 4.163 µs | 4.297 µs | **0.354 µs** |

## Result

Moving the template plan into a native handle closes the major sampling gap
left by the initial 3.1.2 C core. It is especially valuable when the first
sample matters, when int16 output is requested, or when the same stack is
sampled at several offsets/full-scale settings. Warm float overlap remains a
small optimization target, but there is no longer a scenario in this suite
with the order-of-magnitude regression seen in the direct native sampler.

Only integer-tick sample steps currently build a native plan. Unsupported
rational steps and infinite-support templates return no handle and fall back
to the existing exact sampler.
