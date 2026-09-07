# Final output amplitude limits

The sampling order is now:

```text
waveform evaluation and accumulation -> nonlinear map -> filter/predistortion
-> min/max amplitude limits -> quantization or floating-point output
```

Real and complex stacks own their limits, defaulting to negative/positive
infinity. Child waveform and child stack limits do not affect accumulation.
Complex sampling and I/Q output limit each component independently. Whole
sampling, chunks, and caller-provided output buffers follow the same order.
Filter state is carried between chunks before limiting the output. Stack
pickle states now preserve the limits and continue to accept older states.
The raw waveform/stack binary layouts remain unchanged.

Direct `waveform(t)` calls retain amplitude limiting without calibration or
filtering. Sampling explicitly bypasses that evaluation-time limit.

## Execution paths

- Without calibration/filtering, a real waveform keeps its existing native
  evaluator/quantizer with output bounds.
- Non-overlapping native stack plans clip a repeated template once per scale
  before copying it to event destinations, including clipping the idle offset.
- For many distinct scales, limited integer output uses 256-double scratch
  blocks and the existing SIMD quantizer. It does not allocate an entire
  floating-point output buffer.
- Overlapping events are accumulated first and then clipped. No event is
  individually limited before accumulation.
- Plans remain immutable; bounds are execution arguments and do not change
  the plan cache key. Sampling without bounds retains the original hot loops.
- Integer fallback paths without a native plan preserve direct integer output:
  monotone quantization permits limiting the result to the quantized amplitude
  bounds. Only the two bounds need floating-point storage.
- After nonlinear mapping or filtering, final floating-point samples are
  clipped in place before conversion; unbounded output skips this pass.

The C ABI adds `cwaveform_sample_plan_sample_clipped()`. The original
`cwaveform_sample_plan_sample()` remains available with unbounded semantics.

## Local performance

macOS arm64, CPython 3.11.13, NumPy 2.2.4; 10,000 events, 2.4 GHz, int16,
limits `[-0.17, 0.23]`. Each measurement is a median of nine warmed batches;
each batch runs repeated calls for approximately 15 ms. The repeated and
many-scale cases produce 960,144 samples; overlapping events produce 48,144.

| Case | Native limited int16 | Float sample, clip, quantize | Ratio |
| --- | ---: | ---: | ---: |
| Repeated scale, 40 ns spacing | 0.188 ms | 2.368 ms | 12.6x |
| 10,000 distinct scales, 40 ns spacing | 0.595 ms | 2.508 ms | 4.2x |
| Overlapping pulses, 2 ns spacing | 0.241 ms | 0.279 ms | 1.2x |

Every native result was checked bit-for-bit against the reference in the
benchmark. Ratios describe this local workload, not a guarantee for other
hardware, filters, or pulse shapes.

Unbounded int16 sampling in the same final run took 0.190 / 1.207 / 0.238 ms
for these three cases. The pre-change snapshot measured 0.208 / 1.247 /
0.277 ms respectively; no slowdown was observed in these cases. Separate
runs are subject to machine and timing variation. The many-scale bounded
path is faster than the unchanged unbounded path because it uses the new
small-block SIMD conversion path.

Reproduce:

```sh
python3.11 benchmarks/benchmark_output_limits.py
python3.11 benchmarks/benchmark_output_limits.py --source-root /path/to/old/tree --reference-only
```

## Validation

- All 156 tests passed locally, including 76 new parameterized output-limit
  cases. Coverage includes filter gain/attenuation, state across chunks,
  nonlinear-domain validation before limits, component/child isolation,
  integer and fractional-Hz sample grids, rational tick steps, overlap,
  scale-group fallback, partial placements, scratch-block boundaries,
  half-code rounding, idle samples, caller buffers, and pickle compatibility.
- A standalone C harness compiled with AddressSanitizer and
  UndefinedBehaviorSanitizer passed. It compared float, int16, and int32
  plan output with scalar clipping and quantization references for overlap,
  many scales, positive/negative output intervals, and multi-block templates;
  it also checked that bounded execution left the original plan unchanged.
- The C extension rebuilt successfully. Whitespace checks passed for the
  changed implementation and README. Cross-platform CI was not run locally.
