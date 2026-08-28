# Performance audit since 3.1.0

Date: 2026-08-28

Environment: CPython 3.11.13, NumPy 2.3.1, SciPy 1.16.0, macOS arm64.
Numbers are best wall-clock measurements from repeated runs. Small differences
below roughly 5% should be treated as noise.

## Executive summary

- The real-only packed core is worthwhile for complex evaluation and storage,
  but constructing a complex serialized block on every `to_bytes()` call is a
  serious serialization-speed regression.
- The constructor regressions introduced by the 3.1.1 real/complex split have
  been recovered by trusted packed construction and affine operations.
- The 120 GHz clock is primarily a correctness/device-compatibility change. Its
  exact grid generator is around 4-10% slower than the former `np.arange` path
  for global waveforms.
- Plain `float64 -> int16` conversion cannot make sampling faster: it adds a
  read pass and a write pass. Integer sampling becomes valuable only when the
  evaluator can avoid producing the full float64 buffer.
- The retained `WaveVStack` integer path compiles and caches a placement plan.
  Equal-scale templates are quantized once and copied; many-scale templates are
  evaluated and quantized directly into the destination. Both cold and warm
  calls now beat float sampling followed by quantization in the measured 1k and
  10k pulse cases.

## Release-to-current comparison

| Case | 3.1.0 | 3.1.1 commit | Current | Assessment |
|---|---:|---:|---:|---|
| Construct Gaussian | 27.59 us | 32.58 us | 7.03 us | 3.1.1 regressed; current is 3.9x faster than 3.1.0 |
| Construct composite pulse | 289.10 us | 325.15 us | 80.09 us | 3.1.1 regressed; current is 3.6x faster |
| Construct 1k stack | 1.129 ms | 1.960 ms | 1.147 ms | 3.1.1 regression recovered; now near 3.1.0 |
| Pulse equality | 82.56 us | 102.07 us | 58.49 us | current is 1.4x faster than 3.1.0 |
| Simplify 1k stack | 4.835 ms | 5.113 ms | 5.501 ms | remaining 13.8% regression |
| Evaluate complex, 200k points | 4.981 ms | 4.340 ms | 4.327 ms | real-channel split is 15% faster |
| Evaluate 10k stack directly | 66.244 ms | 62.740 ms | 58.567 ms | current is 13% faster |
| Sample 10k stack as float, warm | 64.999 ms | 60.400 ms | 0.734 ms | cached template placement is 89x faster |
| Sample 10k stack as DAC16, warm | 69.746 ms | 65.754 ms | 0.189 ms | direct quantized placement is 369x faster than the old external conversion |
| Complex native size | 2,001 B | 980 B | 980 B | 51% smaller |
| Complex native dump | 0.119 us | 71.96 us | 33.74 us | still 283x slower than 3.1.0 |
| Complex native load | 4.334 us | 10.58 us | 9.726 us | still 2.2x slower |
| 1k stack native size | 28,181 B | 20,180 B | 20,180 B | 28.4% smaller |
| 10k stack native size | 280,181 B | 200,180 B | 200,180 B | 28.6% smaller |

The warm stack sampling rows intentionally include reuse of the compiled
placement plan. The separate cold/warm results below expose the first-call
cost.

## int16 result after the fix

Rate: 2.4 GHz. Pulses are 40 ns apart and produce 96,048 or 960,048 samples.

| Stack | Path | Cold | Warm | Speedup vs float+quantize |
|---|---|---:|---:|---:|
| 1k, same scale | float then int16 | 0.961 ms | 0.219 ms | baseline |
| 1k, same scale | direct int16 | 0.797 ms | 0.070 ms | 1.21x cold / 3.12x warm |
| 10k, same scale | float then int16 | 9.500 ms | 2.532 ms | baseline |
| 10k, same scale | direct int16 | 7.225 ms | 0.300 ms | 1.31x cold / 8.44x warm |
| 1k, varying scale | float then int16 | 0.970 ms | 0.248 ms | baseline |
| 1k, varying scale | direct int16 | 0.922 ms | 0.179 ms | 1.05x cold / 1.39x warm |
| 10k, varying scale | float then int16 | 10.000 ms | 2.561 ms | baseline |
| 10k, varying scale | direct int16 | 8.585 ms | 1.416 ms | 1.16x cold / 1.81x warm |

For a 10k equal-scale stack, direct int32 was 0.332 ms versus 2.671 ms for
float sampling followed by int32 conversion. It is slower than direct int16
(0.205 ms in the same microbenchmark) because it writes twice as many bytes.

Correctness is checked bit-for-bit against `quantize_samples(float_samples)`
for int16 and int32, including caller-provided output buffers, pickled stacks,
varying scales, overlapping pulses, filters, chunking, and rational sample-rate
fallbacks.

## Ineffective or counterproductive optimizations

### Still present, or retained for non-speed reasons

1. **Complex serialization assembled on demand.** Splitting complex waveforms
   into two real channels improves evaluation by about 15% and halves the block
   size, but `ComplexWaveform.to_bytes()` must currently materialize and join
   two blocks on every call. It is not a serialization-speed optimization. A
   cached composite block or a two-buffer pickle protocol is the next fix.

2. **Exact rational sample-grid generation.** A 960k-point 120 GHz-aligned grid
   takes about 0.739 ms versus 0.691 ms for `np.arange`; end-to-end global-cosine
   sampling at 2.4 GHz is 6.97 ms versus 6.37 ms in the 3.1.1 commit. At the 7
   GHz rational fallback it is 16.52 ms versus 15.93 ms. This is a 4-10% cost
   accepted for deterministic clock alignment, not a speed gain.

3. **Stack simplification changes.** The current 1k-pulse simplify result is
   13.8% slower than 3.1.0, and the 100-pulse case is about 12% slower. The new
   packed representation helps sampling and size, but has not improved this
   workload.

4. **Quantization by itself.** The Cython int16 converter is useful for large
   arrays (about 2x faster than the equivalent NumPy expression at 960k
   samples), but a complete float sample followed by conversion is necessarily
   slower than returning float alone. The speedup comes from avoiding that
   float buffer, not from the integer dtype itself.

### Tried during development and removed or replaced

1. **Finite-value checking inside every quantizer loop iteration.** Folding
   `isfinite` into the conversion loop increased a 960k conversion from about
   1.57 ms to 1.72 ms. The vectorized pre-scan was restored.

2. **Uncached direct template quantization.** Rebuilding groups, sorting
   intervals, and quantizing every destination made 1k/10k int16 sampling about
   0.84/8.58 ms, worse than the former 0.72/6.96 ms external conversion. It was
   replaced by a cached placement plan, one-time template evaluation,
   prequantization for repeated scales, and contiguous copies.

3. **Eager canonicalization on every internal pack.** Re-normalizing expressions
   already known to be canonical amplified constructor cost. Trusted and
   canonical constructors now bypass redundant decode/validate/normalize work.

4. **Full linear-term merge in `_expr_add`.** Merging every additive expression
   looked attractive but made large-stack simplification slower. Only zero and
   non-overlap fast paths remain; the established ordered insertion handles the
   general case.

5. **Large global normalized-expression cache.** Retaining decoded forms of
   large blocks increased memory traffic and did not repay lookup overhead.
   Persistent caching is now limited to small expressions; large merges use
   transient local reuse.

6. **Separate quantizer calls for every short composite piece.** For a 96-sample
   three-piece waveform this took about 31.1 us versus 26.9 us for one
   contiguous conversion. Short composites are now materialized once and
   quantized once (about 25.5 us); sparse large windows retain direct piece
   writes.

## Reproduction

```bash
python benchmarks/compare_versions.py --source-root <tree> --label <label>
python benchmarks/benchmark_quantization.py 1000 10000
pytest -q
```

Archived 3.1.0 and committed 3.1.1 source trees were compiled independently,
then benchmarked with the same Python and dependency versions. The current tree
was tested with the same driver after warming its sampling-plan cache where the
table says `warm`.
