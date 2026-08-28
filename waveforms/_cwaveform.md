# WNF4 / WNS4 cross-language waveform format

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
