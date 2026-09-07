# WNF4 / WNS4 / NLM1 cross-language waveform format

WNF4 and WNS4 are immutable, little-endian binary blocks. They can be read by
C, Rust, Python, or device-side tooling without rebuilding a Python expression
graph. The public ABI is declared in `_cwaveform.h`.

Time uses signed 64-bit integer ticks. The tick frequency is process-global,
defaults to 120 GHz, and is configured through
`cwaveform_set_ticks_per_second()` before blocks are constructed or loaded. A
sample clock is the exact rational number
`step_numerator / step_denominator` in ticks. Amplitudes and coefficients use
IEEE-754 binary64.

## WNF4 waveform block (ABI 2)

The 24-byte header is followed by `node_count` fixed-size 28-byte nodes in
topological order, then a contiguous binary64 parameter pool.

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | `char[4]` | `WNF4` |
| 4 | `u16` | ABI version, currently 2 |
| 6 | `u16` | flags, currently 0 |
| 8 | `u32` | node count |
| 12 | `u32` | root node index |
| 16 | `u32` | parameter-pool element count |
| 20 | `u32` | reserved, currently 0 |

Each node has the following serialized layout. Support bounds are derived when
the block is loaded and therefore occupy no serialized space.

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | `u8` | opcode |
| 1 | `u8` | opcode flags |
| 2 | `u16` | number of parameters after the inline parameter |
| 4 | `u32` | parameter offset when the count is nonzero; otherwise left child |
| 8 | `u32` | right child |
| 12 | `i64` | time shift or window lower bound in ticks |
| 20 | `f64` | inline parameter or bit-preserved window upper bound |

Leaf opcodes 1–20 cover constant, Gaussian, cosine, sine, square, linear, erf,
sinc, exponential, interpolation, three chirps, cosh, sinh, DRAG, mollifier,
Gaussian derivative, `drag_sin`, and `drag_sinx`. Opcodes 32–36 are add,
multiply, scale, non-negative integer power, and window. Child indices must be
smaller than the current node index. Variable-sized leaf arguments are stored
directly in the parameter pool.

Addition and multiplication are normalized in C by flattening associative
trees, sorting canonical operands, folding constants, combining identical
terms, and rebuilding a compact reachable DAG. This makes equivalent common
algebra independent of Python construction order.

## WNS4 template stack block (ABI 2)

WNS4 stores each distinct WNF4 template once, followed by structure-of-arrays
event data. Each event occupies 20 bytes while retaining binary64 amplitude and
signed 64-bit delay precision.

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | `char[4]` | `WNS4` |
| 4 | `u16` | ABI version, currently 2 |
| 6 | `u16` | flags, currently 0 |
| 8 | `u32` | template count |
| 12 | `u32` | event count |
| 16 | `u32[]` | byte size of each WNF4 template |
| … | `u8[]` | concatenated WNF4 templates |
| … | `u32[event_count]` | template indices |
| … | `i64[event_count]` | delay ticks |
| … | `f64[event_count]` | amplitude scales |

The event section occupies the final `20 * event_count` bytes. Global shift and
DC offset are folded into events when a standalone WNS4 block is requested.

## Ownership and validation

The C API returns reference-counted immutable handles. Byte pointers returned
by `cwaveform_wave_bytes()` and `cwaveform_stack_bytes()` remain valid until the
last matching release. Loaders validate magic, version, topology, sizes,
parameters, template indices, and event arrays before evaluation. Waveform
construction, algebra, derivative, filtering, simplification, stack operations,
sampling plans, evaluation, quantization, and serialization all operate on this
single C representation. Complex Python objects contain two independent real C
blocks.

The additive `cwaveform_sample_plan_sample_clipped()` API accepts final
amplitude bounds without changing the serialized formats or the original
`cwaveform_sample_plan_sample()` ABI. Limits are applied after all event
accumulation, before quantization. Non-overlapping templates are limited once
per amplitude group and copied to their destinations; overlapping events are
accumulated before limiting. Bounds are execution parameters and do not alter
the immutable plan. The Python sampling pipeline invokes these bounds in the
plan only when no nonlinear mapping or filtering follows; otherwise it limits
the final processed samples.

## SOS filtering

The additive `cwaveform_sos_filter()` API processes a real-coefficient SOS
cascade using transposed direct form II, with the same `(sections, 6)`
coefficient layout as SciPy and `a0 == 1`. Coefficients are read-only; the caller
supplies two mutable delays per section. Delays capture the unclipped cascade
output and can be carried across calls. `initial` is a baseline subtracted
before the cascade and restored afterward, not an initializer for the delays.

Input, state and float output strides count doubles; strides of two support
the real and imaginary components of complex128 buffers. Integer output must
be contiguous. Float output supports exact input/output aliasing; other
overlaps and overlap with coefficients or state must be avoided by C callers.
The Python wrapper copies overlapping inputs when necessary and returns a
new state array without modifying the supplied `zi`.

Processing uses a fixed 256-double scratch buffer. One-section filters retain
coefficients and delays in registers; cascades process all sections for each
sample so delay updates can overlap. Each block restores the baseline, applies
limits, and writes float output or invokes the existing SIMD DAC quantizer.
This removes full-array baseline, clip and quantization passes and the full
filtered float intermediate for real integer output. Raw waveform evaluation
and nonlinear mapping still precede this stage. The SOS recurrence allocates
no heap memory; conversion uses the quantizer's existing platform dispatch.
The API returns 0 on success, -1 for invalid arguments or SOS normalization,
-2 if conversion allocation fails, and -3 if integer conversion encounters a
non-finite sample after clipping.

## NLM1 nonlinear-map block (ABI 1)

NLM1 represents a scalar, memoryless nonlinear map on a uniform input grid.
Runtime evaluation dispatches to ARM64 NEON, x86 AVX2, or x86 AVX-512
batch kernels when available, with a portable scalar fallback. The serialized
format is independent of the selected kernel and produces the same boundary,
clipping, and quantization semantics on macOS, Linux, and Windows.
It is applied after waveform/event accumulation and before linear filtering or
integer quantization. The header is followed by either one ordinate per grid
point or four normalized cubic coefficients per interval.

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | `char[4]` | `NLM1` |
| 4 | `u16` | ABI version, currently 1 |
| 6 | `u8` | method: 1 linear, 2 monotone cubic |
| 7 | `u8` | payload precision: 32 or 64 bits |
| 8 | `u8` | out-of-domain policy: 0 error, 1 clip |
| 9 | `u8[3]` | reserved, currently 0 |
| 12 | `u32` | uniform-grid point count |
| 16 | `f64` | absolute input-domain minimum |
| 24 | `f64` | absolute input-domain maximum |
| 32 | `f64` | input/reference offset |
| 40 | `f64` | output/reference offset |
| 48 | `f32[]` or `f64[]` | method payload |

For a centered map, an input `v` first becomes `x = v + input_offset`; the
result is then reduced by `output_offset`. Linear payloads store
`point_count` values. Cubic payloads store `(a, b, c, d)` for each of
`point_count - 1` intervals and evaluate
`a + r * (b + r * (c + r * d))`, where `r` is the normalized position in the
interval. This avoids per-sample root finding, knot searches, and Python
callbacks.
