# WNF4 / WNS4 native waveform format

WNF4 and WNS4 are immutable, little-endian binary blocks. They are designed
to be read by C, Rust, Python, or device-side tooling without reconstructing a
Python object graph. The public ABI is declared in `wf_native.h`.

All times are signed 64-bit ticks at 120,000,000,000 ticks per second. A sample
clock is represented by the rational number `step_numerator / step_denominator`
in ticks. Values and coefficients use IEEE-754 binary64 in ABI version 1.

## WNF4 waveform block

The 24-byte header is followed by `node_count` fixed-size 28-byte nodes in
topological order.

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | `char[4]` | `WNF4` |
| 4 | `u16` | ABI version, currently 1 |
| 6 | `u16` | flags, currently 0 |
| 8 | `u32` | node count |
| 12 | `u32` | root node index |
| 16 | `u64` | ticks per second |

Each node has the following physical layout. Fields unused by an opcode are
zero. Bounds are derived from leaf parameters and child nodes, so they do not
occupy serialized space.

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | `u8` | opcode |
| 1 | `u8[3]` | reserved |
| 4 | `u32` | left child |
| 8 | `u32` | right child |
| 12 | `i64` | time shift in ticks |
| 20 | `f64` | opcode parameter |

Opcodes 1–5 are constant, Gaussian, cosine, sine, and square leaves. Opcodes
16–18 are add, multiply, and scale. Child indices must be smaller than the
current node index. Gaussian and square store their full width in seconds;
cosine and sine store angular frequency; scale stores its multiplier.

## WNS4 template stack block

WNS4 stores each distinct WNF4 template once, followed by structure-of-arrays
event data. This makes a large pulse train 20 bytes per event without giving
up binary64 amplitudes or 64-bit tick delays.

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | `char[4]` | `WNS4` |
| 4 | `u16` | ABI version, currently 1 |
| 6 | `u16` | flags, currently 0 |
| 8 | `u32` | template count |
| 12 | `u32` | event count |
| 16 | `u32[]` | byte size of each WNF4 template |
| … | `u8[]` | concatenated WNF4 templates |
| … | `u32[event_count]` | template indices |
| … | `i64[event_count]` | delay ticks |
| … | `f64[event_count]` | amplitude scales |

The event section occupies the last `20 * event_count` bytes, so its offset is
derived without another header field. Global shift and DC offset are folded
into events when a standalone WNS4 block is requested.

## Ownership and validation

The C API returns reference-counted immutable handles. Byte pointers returned
by `wf_native_wave_bytes` and `wf_native_stack_bytes` stay valid until the last
matching release. Constructors validate topology, finite parameters, template
indices, sizes, and the embedded tick rate before evaluation. The Python
binding retains an input `bytes` object and lazily builds the decoded sidecar
on first numerical use, matching the zero-copy deserialization behavior of the
existing packed backend.
