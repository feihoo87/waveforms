# Native C core benchmark report

Date: 2026-08-28. Machine: Apple arm64. Runtime: CPython 3.11.13,
NumPy 2.3.1. Every timing is the best of five adaptive runs from
`benchmarks/compare_versions.py`; lower is better. Releases 2.2.3 and 3.1.0
were rerun from local archived trees in the same process environment.

The candidate is the Python-independent C99 core exposed as the experimental
`NativeWaveform`, `NativeComplexWaveform`, and `NativeWaveVStack` classes. It
uses WNF4/WNS4 blocks, 120 GHz ticks, rational sample steps, 28-byte nodes,
template sharing, native float/int16/int32 sampling, and lazy block decoding.

## Construction and algebra

| Operation | 2.2.3 | 3.1.0 | Native C |
| --- | ---: | ---: | ---: |
| Gaussian construction | 1.089 µs | 27.455 µs | **0.583 µs** |
| Addition construction | 1.355 µs | 13.779 µs | **1.037 µs** |
| Multiplication construction | 1.568 µs | 12.187 µs | **1.118 µs** |
| Composite pulse construction | 16.105 µs | 286.362 µs | **10.922 µs** |
| Shift construction | 4.715 µs | 0.657 µs | **0.551 µs** |
| 1,000-event stack construction | 2.468 ms | 1.141 ms | **0.790 ms** |
| Pulse equality | 62.649 µs | 84.128 µs | **0.409 µs** |
| Pulse simplify | 31.231 µs | 41.092 µs | **0.422 µs** |
| 100-event stack simplify | 1.896 ms | 0.482 ms | **0.021 ms** |
| 1,000-event stack simplify | 146.762 ms | 4.762 ms | **0.204 ms** |

## Evaluation and sampling

| Operation | 2.2.3 | 3.1.0 | Native C |
| --- | ---: | ---: | ---: |
| Cosine, 1M positions | 10.721 ms | 6.391 ms | **3.048 ms** |
| Pulse, 200K positions | 2.484 ms | 2.371 ms | **1.998 ms** |
| Complex pulse, 200K positions | 6.870 ms | 5.727 ms | **4.392 ms** |
| Stack 1K, direct evaluation | 9.459 ms | 6.485 ms | **1.240 ms** |
| Stack 10K, direct evaluation | 94.945 ms | 64.965 ms | **12.846 ms** |
| Stack 1K, float sample | 9.510 ms | 6.590 ms | **1.311 ms** |
| Stack 10K, float sample | 97.578 ms | 66.273 ms | **13.602 ms** |
| Stack 1K, int16 sample | 10.046 ms | 7.112 ms | **1.390 ms** |
| Stack 10K, int16 sample | 99.257 ms | 69.061 ms | **13.991 ms** |

## Serialization

`native bytes` means the smallest supported non-pickle representation:
msgpack lists in 2.2.3 and packed blocks in 3.1.0/native C.

| Object | 2.2.3 | 3.1.0 | Native C |
| --- | ---: | ---: | ---: |
| Pulse | 478 B | 490 B | **360 B** |
| Complex pulse | 1,904 B | 2,001 B | **760 B** |
| Stack 100 | 8,330 B | 2,981 B | **2,128 B** |
| Stack 1K | 83,032 B | 28,181 B | **20,128 B** |
| Stack 10K | 830,034 B | 280,181 B | **200,128 B** |

| Native block operation | 2.2.3 | 3.1.0 | Native C |
| --- | ---: | ---: | ---: |
| Pulse dump | 6.715 µs | 0.113 µs | **0.086 µs** |
| Pulse load | 10.769 µs | 4.117 µs | **1.098 µs** |
| Complex dump | 28.349 µs | 0.116 µs | **0.045 µs** |
| Complex load | 39.265 µs | 4.280 µs | **2.844 µs** |
| Stack 10K dump | 13.184 ms | 0.075 µs | **0.070 µs** |
| Stack 10K load | 44.744 ms | 4.299 µs | **0.360 µs** |

Protocol-5 pickle size also has no remaining regression: the pulse is 470 B,
versus 474 B in 2.2.3 and 612 B in 3.1.0.

## Result and current 3.1.1 caveat

The candidate beats both requested baselines, 2.2.3 and 3.1.0, in every metric
in this benchmark suite and in every recorded serialization size.

It does not yet beat the committed 3.1.1 implementation everywhere. The
current `WaveVStack` retains a compiled repeated-template placement plan. On a
warm 10K-event sample it takes 0.714 ms for float and 0.163 ms for int16,
versus 13.602 ms and 13.991 ms in the C candidate, which currently reevaluates
each event. The current complex wrapper is also about 10% faster (3.949 ms
versus 4.392 ms), while real pulse evaluation is within roughly 2%.

For that reason this implementation remains side by side rather than replacing
the public classes. The next optimization should move the 3.1.1 template-plan
cache into an immutable native sample-plan handle, followed by strided native
real/imag output. Everything else—including construction, algebra, direct
stack evaluation, serialized size, dump, and lazy load—is already ahead of the
committed 3.1.1 backend as well.
