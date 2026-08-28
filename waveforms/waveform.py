"""Independent packed-binary waveform implementation.

``Waveform`` and ``WaveVStack`` keep their signal representation in immutable
binary blocks owned by :mod:`waveforms._waveform`. Times in those blocks are
signed 64-bit ticks. The tick duration is process-wide configuration and is
intentionally not repeated in every block.
"""

from __future__ import annotations

import struct
from functools import lru_cache
from typing import Iterable, cast

import numpy as np
from numpy import e, inf, pi
from scipy.signal import sosfilt

from ._waveform import (
    COS, COSH, D_GAUSSIAN, DRAG, DRAG_SIN, DRAG_SINX, ERF, EXP,
    EXPONENTIALCHIRP, GAUSSIAN,
    HYPERBOLICCHIRP, INTERP, LINEAR, LINEARCHIRP, MOLLIFIER, SINC, SINH,
    PackedStack, PackedWaveform, accumulate_template, affine_expression,
    basic, basic_expression, constant, get_time_resolution,
    piecewise, piecewise_canonical, place_quantized_template,
    place_template_quantized,
    quantize_samples, quantize_time, registerBaseFunc,
    registerDerivative, sample_clock, sample_grid, set_time_resolution,
    tick_to_time, time_to_tick,
)

_ZERO_EXPR = ((), ())
_ONE_EXPR = ((((), ()),), (1.0,))
_COMPLEX_WAVE_MAGIC = b"CWF1"
_COMPLEX_STACK_MAGIC = b"CWS1"
_COMPLEX_HEADER = struct.Struct("<4sII")
_SUPPORTED_SAMPLE_RATES = frozenset({
    500_000_000,
    1_000_000_000,
    1_200_000_000,
    2_000_000_000,
    2_400_000_000,
    2_500_000_000,
    4_000_000_000,
    6_000_000_000,
    8_000_000_000,
    10_000_000_000,
})
_INF_TICK = np.iinfo(np.int64).max


def _number_parts(value):
    """Return real and imaginary Python floats for a scalar number."""
    if not isinstance(value, (int, float, complex, np.number)):
        raise TypeError(f"expected a scalar number, got {type(value).__name__}")
    number = complex(value)
    return float(number.real), float(number.imag)


def _pack_complex(magic, real_data, imag_data):
    return (_COMPLEX_HEADER.pack(magic, len(real_data), len(imag_data))
            + real_data + imag_data)


def _unpack_complex(data, magic):
    data = data if isinstance(data, bytes) else bytes(data)
    if len(data) < _COMPLEX_HEADER.size:
        raise ValueError("truncated complex waveform")
    actual_magic, real_size, imag_size = _COMPLEX_HEADER.unpack_from(data)
    if actual_magic != magic:
        raise ValueError("unsupported complex waveform format")
    start = _COMPLEX_HEADER.size
    split = start + real_size
    if split + imag_size != len(data):
        raise ValueError("invalid complex waveform layout")
    return data[start:split], data[split:]


def _copy_sampling_metadata(source, target):
    target.start = source.start
    target.stop = source.stop
    target.sample_rate = source.sample_rate
    target.filters = source.filters
    target.label = source.label
    return target


def _scalar(opcode, *args, shift=0.0):
    return basic(opcode, args, shift)


def _expression(opcode, *args, shift=0.0):
    return basic_expression(opcode, args, shift)


def _affine_expression(opcode, *args, shift=0.0, scale=1.0, offset=0.0):
    return affine_expression(opcode, args, shift, scale, offset)


def _piecewise(bounds, *expressions):
    return Waveform._from_core(
        piecewise_canonical(tuple(bounds), expressions)
    )


def _is_tick_aligned(value, tick):
    reconstructed = tick_to_time(tick)
    tolerance = max(get_time_resolution() * 1e-6,
                    abs(np.spacing(float(value))) * 2)
    return abs(float(value) - reconstructed) <= tolerance


def _sampling_plan(start, stop, sample_rate):
    """Return an integer/rational tick plan for a supported device rate."""
    rate = float(sample_rate)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("sample_rate must be a finite positive number")
    integer_rate = int(rate)
    if rate != integer_rate or integer_rate not in _SUPPORTED_SAMPLE_RATES:
        return None
    start_tick = time_to_tick(start)
    stop_tick = time_to_tick(stop)
    if (not _is_tick_aligned(start, start_tick)
            or not _is_tick_aligned(stop, stop_tick)):
        return None
    step_numerator, step_denominator = sample_clock(integer_rate)
    span_numerator = (stop_tick - start_tick) * step_denominator
    count = (max(0, span_numerator) + step_numerator - 1) // step_numerator
    return (start_tick, int(count), step_numerator, step_denominator)


def _quantization_bits(dtype, out):
    if dtype is None and out is not None:
        candidate = np.asarray(out).dtype
        if candidate in (np.dtype(np.int16), np.dtype(np.int32)):
            dtype = candidate
    if dtype is None:
        return None, None
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.int16):
        return dtype, 16
    if dtype == np.dtype(np.int32):
        return dtype, 32
    if np.issubdtype(dtype, np.integer):
        raise TypeError("only int16 and int32 amplitude quantization is supported")
    return dtype, None


def _integer_output(count, bits, out=None, fill=0):
    dtype = np.dtype(np.int16 if bits == 16 else np.int32)
    if out is None:
        output = np.empty(count, dtype=dtype)
    else:
        output = np.asarray(out)
        if output.shape != (count,):
            raise ValueError("out has the wrong shape")
        if output.dtype != dtype:
            raise TypeError(f"out must have dtype {dtype}")
        if not output.flags.c_contiguous or not output.flags.writeable:
            raise ValueError("out must be a writable C-contiguous array")
    output.fill(fill)
    return output


def _finish_samples(sig, dtype, full_scale, out):
    dtype, bits = _quantization_bits(dtype, out)
    if bits is not None:
        return quantize_samples(sig, bits, full_scale, out)
    result = np.asarray(sig) if dtype is None else np.asarray(sig, dtype=dtype)
    if out is None:
        return result
    output = np.asarray(out)
    if output.shape != result.shape:
        raise ValueError("out has the wrong shape")
    if not output.flags.writeable:
        raise ValueError("out must be writable")
    output[...] = result
    return output


def _filter_samples(sig, filters, zi=None):
    if filters is None:
        return sig, zi
    sos, initial = filters
    sos = np.asarray(sos)
    if not sos.flags.writeable:
        sos = sos.copy()
    values = sig - initial if initial else sig
    if zi is None:
        filtered = sosfilt(sos, values)
    else:
        filtered, zi = sosfilt(sos, values, zi=zi)
    if initial:
        filtered = filtered + initial
    return filtered, zi


def _sample_iq(owner, sample_rate=None, out=None, chunk_size=None,
               function_lib=None, filters=None, dtype=np.int16,
               full_scale=1.0):
    """Sample a complex wrapper as separate real I/Q buffers."""
    if out is None:
        out_i = out_q = None
    else:
        if not isinstance(out, (tuple, list)) or len(out) != 2:
            raise TypeError("out must be an (I, Q) pair")
        out_i, out_q = out
    values = owner.sample(
        sample_rate=sample_rate, chunk_size=chunk_size,
        function_lib=function_lib, filters=filters,
    )
    if chunk_size is None:
        return (
            _finish_samples(values.real, dtype, full_scale, out_i),
            _finish_samples(values.imag, dtype, full_scale, out_q),
        )

    def chunks():
        offset = 0
        for values_chunk in values:
            size = len(values_chunk)
            target_i = None if out_i is None else out_i[offset:offset + size]
            target_q = None if out_q is None else out_q[offset:offset + size]
            yield (
                _finish_samples(values_chunk.real, dtype, full_scale, target_i),
                _finish_samples(values_chunk.imag, dtype, full_scale, target_q),
            )
            offset += size

    return chunks()


class _SamplingMixin:
    start: float | None
    stop: float | None
    sample_rate: float | None
    filters: tuple[np.ndarray, float] | None

    def sample(self, sample_rate=None, out: np.ndarray | None = None,
               chunk_size=None, function_lib=None,
               filters: tuple[np.ndarray, float] | None = None,
               dtype=None, full_scale=1.0):
        if function_lib is not None:
            raise NotImplementedError("custom waveform functions are not supported")
        if sample_rate is None:
            sample_rate = self.sample_rate
        if self.start is None or self.stop is None or sample_rate is None:
            raise ValueError(
                f"Waveform is not initialized. {self.start=}, {self.stop=}, "
                f"{sample_rate=}"
            )
        if filters is None:
            filters = self.filters
        if chunk_size is not None:
            return self._sample_iter(
                sample_rate, int(chunk_size), out, filters, dtype, full_scale
            )

        plan = _sampling_plan(self.start, self.stop, sample_rate)
        if plan is None:
            x = np.arange(self.start, self.stop, 1 / float(sample_rate))
            sig = cast(np.ndarray, self(x))
        else:
            _, bits = _quantization_bits(dtype, out)
            if bits is not None and filters is None:
                specialized = getattr(
                    self, "_sample_supported_quantized", None
                )
                if specialized is not None:
                    result = specialized(
                        *plan, bits, full_scale, out=out
                    )
                    if result is not NotImplemented:
                        return cast(np.ndarray, result)
            sig = self._sample_supported(*plan)
        sig, _ = _filter_samples(sig, filters)
        return cast(np.ndarray, _finish_samples(sig, dtype, full_scale, out))

    def _sample_supported(self, start_tick, count, step_numerator,
                          step_denominator, index_offset=0):
        x = sample_grid(start_tick, count, step_numerator, step_denominator,
                        index_offset)
        return cast(np.ndarray, self(x))

    def _sample_iter(self, sample_rate, chunk_size, out, filters, dtype,
                     full_scale):
        start = cast(float, self.start)
        stop = cast(float, self.stop)
        output_index = 0
        zi = None
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        plan = _sampling_plan(start, stop, sample_rate)
        if plan is None:
            rate = float(sample_rate)
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError("sample_rate must be a finite positive number")
            total = max(0, int(np.ceil((stop - start) * rate)))
        else:
            start_tick, total, step_numerator, step_denominator = plan
        if filters is not None:
            sos, _ = filters
            sos = np.asarray(sos)
            zi = np.zeros((sos.shape[0], 2))
        if out is not None:
            output = np.asarray(out)
            if output.ndim != 1 or len(output) != total:
                raise ValueError("out has the wrong shape")
        else:
            output = None

        while output_index < total:
            size = min(chunk_size, total - output_index)
            if plan is None:
                indices = output_index + np.arange(size, dtype=np.float64)
                x = start + indices / rate
                sig = cast(np.ndarray, self(x))
            else:
                sig = self._sample_supported(
                    start_tick, size, step_numerator, step_denominator,
                    output_index,
                )
            sig, zi = _filter_samples(sig, filters, zi)
            target = (None if output is None
                      else output[output_index:output_index + size])
            yield _finish_samples(sig, dtype, full_scale, target)
            output_index += size

    def _play(self, time_unit, volume):
        import pyaudio

        rate = 48_000
        dynamic_volume = 1.0
        amp = 2**15 * 0.999 * volume
        audio = pyaudio.PyAudio()
        try:
            stream = audio.open(format=pyaudio.paInt16, channels=1,
                                rate=rate, output=True)
            try:
                for data in self.sample(sample_rate=rate / time_unit,
                                        chunk_size=1024):
                    limit = np.abs(data).max()
                    if limit > 0 and dynamic_volume > 1.0 / limit:
                        dynamic_volume = 1.0 / limit
                        amp = 2**15 * 0.99 * volume * dynamic_volume
                    stream.write(bytes((amp * data).astype(np.int16).data))
            finally:
                stream.stop_stream()
                stream.close()
        finally:
            audio.terminate()

    def play(self, time_unit=1, volume=1.0):
        import multiprocessing as mp

        process = mp.Process(target=self._play, args=(time_unit, volume),
                             daemon=True)
        process.start()


class Waveform(_SamplingMixin):
    __slots__ = (
        "_core", "_delay", "_scale", "max", "min", "start", "stop",
        "sample_rate", "filters", "label",
    )

    def __init__(self, bounds=(inf,), seq=None, min=-inf, max=inf, *,
                 _core=None, _delay=0.0, _scale=1.0):
        if _core is None:
            if seq is None:
                seq = (_ZERO_EXPR,)
            _core = PackedWaveform.from_legacy(tuple(bounds), tuple(seq))
        self._core = _core
        self._delay = quantize_time(_delay)
        scale_real, scale_imag = _number_parts(_scale)
        if scale_imag != 0:
            raise TypeError("Waveform scale must be real; use ComplexWaveform")
        self._scale = scale_real
        self.max = max
        self.min = min
        self.start = None
        self.stop = None
        self.sample_rate = None
        self.filters = None
        self.label = None

    @classmethod
    def _from_core(cls, core, delay=0.0, scale=1.0):
        # Internal cores and affine values are already real and tick-aligned.
        # Bypass the public constructor's legacy conversion and type checks.
        obj = cls.__new__(cls)
        obj._core = core
        obj._delay = quantize_time(delay)
        obj._scale = scale
        obj.max = inf
        obj.min = -inf
        obj.start = None
        obj.stop = None
        obj.sample_rate = None
        obj.filters = None
        obj.label = None
        return obj

    def _materialized_core(self):
        core = self._core
        if self._scale != 1:
            core = core.scaled(self._scale)
        if self._delay != 0:
            core = core.shifted(self._delay)
        return core

    def _sample_supported_quantized(
            self, start_tick, count, step_numerator, step_denominator,
            bits, full_scale, out=None):
        """Quantize non-zero pieces directly into the integer output."""
        x = sample_grid(
            start_tick, count, step_numerator, step_denominator
        )
        output = _integer_output(count, bits, out)
        parts, _ = self._core.parts_shifted(x, self._delay)
        should_scale = self._scale != 1
        should_clip = self.min != -inf or self.max != inf
        # Several tiny pieces cost more as separate quantizer calls than one
        # contiguous pass.  Materialize short composite waveforms once, while
        # retaining sparse direct writes for large windows.
        if len(parts) > 1 and count < 2048:
            signal = np.zeros(count, dtype=np.float64)
            for start, stop, part in parts:
                if should_scale:
                    part = part * self._scale
                if should_clip:
                    part = np.clip(part, self.min, self.max)
                signal[start:stop] += part
            return quantize_samples(signal, bits, full_scale, out)
        for start, stop, part in parts:
            if should_scale:
                part = part * self._scale
            if should_clip:
                part = np.clip(part, self.min, self.max)
            quantize_samples(
                part, bits, full_scale, output[start:stop]
            )
        return output

    @property
    def bounds(self):
        return tuple(self._materialized_core().get_bounds())

    @property
    def seq(self):
        return self._materialized_core().to_legacy()[1]

    @staticmethod
    def _begin(bounds, seq):
        for index, expression in enumerate(seq):
            if expression != _ZERO_EXPR:
                return -inf if index == 0 else bounds[index - 1]
        return inf

    @staticmethod
    def _end(bounds, seq):
        count = len(bounds)
        for index, expression in enumerate(reversed(seq)):
            if expression != _ZERO_EXPR:
                return inf if index == 0 else bounds[count - index - 1]
        return -inf

    @property
    def begin(self):
        value = self._begin(*self._core.to_legacy()) + self._delay
        return value if self.start is None else max(self.start, value)

    @property
    def end(self):
        value = self._end(*self._core.to_legacy()) + self._delay
        return value if self.stop is None else min(self.stop, value)

    def to_bytes(self):
        """Return the canonical signal block; sampling metadata is separate."""
        return self._materialized_core().to_bytes()

    @classmethod
    def from_bytes(cls, data):
        if bytes(data[:4]) == _COMPLEX_WAVE_MAGIC:
            return ComplexWaveform.from_bytes(data)
        return cls._from_core(PackedWaveform.from_bytes(data))

    def simplify(self, eps=1e-15):
        return Waveform._from_core(self._materialized_core().simplify(eps))

    def filter(self, low=0, high=inf, eps=1e-15):
        return Waveform._from_core(self._materialized_core().filtered(low, high, eps))

    def __pow__(self, n):
        return Waveform._from_core(self._materialized_core().power(n))

    def __add__(self, other):
        if isinstance(other, ComplexWaveform):
            return other + self
        if isinstance(other, ComplexWaveVStack):
            return other + self
        if isinstance(other, WaveVStack):
            return other + self
        if not isinstance(other, Waveform):
            real, imag = _number_parts(other)
            if imag != 0:
                return ComplexWaveform(self + real, _real_const(imag))
            other = _real_const(real)
        if self._delay == other._delay:
            if self._scale == other._scale:
                return Waveform._from_core(
                    self._core.add(other._core), self._delay, self._scale
                )
            return Waveform._from_core(
                self._core.add_affine(
                    other._core, 0.0, self._scale, 0.0, other._scale
                ),
                self._delay,
            )
        return Waveform._from_core(self._core.add_affine(
            other._core, self._delay, self._scale,
            other._delay, other._scale,
        ))

    def __radd__(self, value):
        return self + value

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, value):
        real, imag = _number_parts(value)
        if imag != 0:
            return ComplexWaveform(real - self, _real_const(imag))
        return real + (-self)

    def __mul__(self, other):
        if isinstance(other, ComplexWaveform):
            return other * self
        if isinstance(other, ComplexWaveVStack):
            return other * self
        if isinstance(other, WaveVStack):
            return other * self
        if isinstance(other, Waveform):
            scale = self._scale * other._scale
            if self._delay == other._delay:
                return Waveform._from_core(
                    self._core.mul(other._core), self._delay, scale
                )
            return Waveform._from_core(
                self._core.mul_affine(
                    other._core, self._delay, other._delay
                ),
                scale=scale,
            )
        real, imag = _number_parts(other)
        if imag != 0:
            return ComplexWaveform(self * real, self * imag)
        return Waveform._from_core(self._core, self._delay, self._scale * real)

    def __rmul__(self, value):
        return self * value

    def __truediv__(self, other):
        if isinstance(other, (Waveform, ComplexWaveform, WaveVStack,
                              ComplexWaveVStack)):
            raise TypeError("division by waveform")
        return self * (1 / other)

    def __neg__(self):
        return self * -1

    def __rshift__(self, time):
        delay = quantize_time(self._delay + time)
        return Waveform._from_core(self._core, delay, self._scale)

    def __lshift__(self, time):
        return self >> -time

    @property
    def marker(self):
        core = self._materialized_core().simplify()
        bounds, expressions = core.to_legacy()
        return Waveform(bounds, tuple(
            _ZERO_EXPR if expression == _ZERO_EXPR else _ONE_EXPR
            for expression in expressions
        ))

    def mask(self, edge=0):
        marker = self.marker
        bounds = marker.bounds
        expressions = marker.seq
        out_bounds = []
        out_expressions = []
        in_wave = expressions[0] != _ZERO_EXPR

        if expressions[0] == _ZERO_EXPR:
            out_bounds.append(bounds[0] - edge)
            out_expressions.append(_ZERO_EXPR)

        for boundary, expression in zip(bounds[1:], expressions[1:]):
            if not in_wave and expression != _ZERO_EXPR:
                in_wave = True
                out_bounds.append(boundary + edge)
                out_expressions.append(_ONE_EXPR)
            elif in_wave and expression == _ZERO_EXPR:
                in_wave = False
                boundary -= edge
                if boundary > out_bounds[-1]:
                    out_bounds.append(boundary)
                    out_expressions.append(_ZERO_EXPR)
                else:
                    out_bounds[-1] = boundary
        return Waveform(tuple(out_bounds), tuple(out_expressions))

    def __or__(self, other):
        if not isinstance(other, Waveform):
            other = const(other)
        return (self.marker + other.marker).marker

    def __ior__(self, other):
        return self | other

    def __and__(self, other):
        if not isinstance(other, Waveform):
            other = const(other)
        return (self.marker * other.marker).marker

    def __iand__(self, other):
        return self & other

    def __call__(self, x, frag=False, out=None, accumulate=False,
                 function_lib=None):
        if function_lib is not None:
            raise NotImplementedError("custom waveform functions are not supported")
        scalar = isinstance(x, (int, float, complex, np.number))
        values = np.asarray([x]) if scalar else np.asarray(x)
        parts, _ = self._core.parts_shifted(values, self._delay)
        scaled_parts = []
        should_scale = self._scale != 1
        should_clip = self.min != -inf or self.max != inf
        for start, stop, part in parts:
            if should_scale:
                part = part * self._scale
            if should_clip:
                part = np.clip(part, self.min, self.max)
            scaled_parts.append((start, stop, part))
        if frag:
            if out is None:
                return scaled_parts
            if not accumulate:
                out.clear()
            out.extend(scaled_parts)
            return out

        if out is None:
            out = np.zeros_like(values, dtype=float)
        elif not accumulate:
            out[...] = 0
        for start, stop, part in scaled_parts:
            out[start:stop] += part
        return out[0] if scalar else out

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, (int, float, complex, np.number)):
            real, imag = _number_parts(other)
            if imag != 0:
                return False
            other = _real_const(real)
        if isinstance(other, ComplexWaveform):
            return other == self
        if not isinstance(other, Waveform):
            return False
        if (self.max, self.min, self.start, self.stop) != (
                other.max, other.min, other.start, other.stop):
            return False
        return self.simplify().to_bytes() == other.simplify().to_bytes()

    def __hash__(self):
        return hash((self.simplify().to_bytes(), self.max, self.min,
                     self.start, self.stop))

    def __repr__(self):
        return (f"Waveform(segments={len(self.bounds)}, bytes={len(self.to_bytes())}, "
                f"begin={self.begin}, end={self.end})")

    def _repr_latex_(self):
        return rf"f(t)\quad\mathrm{{on}}\ [{self.begin:g},\,{self.end:g}]"

    def __getstate__(self):
        return (self._core.to_bytes(), self._delay, self._scale, self.max,
                self.min, self.start, self.stop, self.sample_rate,
                self.filters, self.label)

    def __setstate__(self, state):
        (data, self._delay, self._scale, self.max, self.min, self.start,
         self.stop, self.sample_rate, self.filters, self.label) = state
        self._core = PackedWaveform.from_bytes(data)


class ComplexWaveform(_SamplingMixin):
    """Complex waveform represented by two independent real waveforms."""

    __slots__ = (
        "_real", "_imag", "max", "min", "start", "stop", "sample_rate",
        "filters", "label",
    )

    def __init__(self, real=0.0, imag=0.0):
        if isinstance(real, ComplexWaveform):
            if imag != 0:
                raise TypeError("imag must be zero when copying ComplexWaveform")
            self._real = real._real
            self._imag = real._imag
        else:
            if isinstance(real, Waveform):
                self._real = real
            else:
                real_value, embedded_imag = _number_parts(real)
                if embedded_imag != 0:
                    imag_value, nested_imag = _number_parts(imag)
                    if imag_value != 0 or nested_imag != 0:
                        raise TypeError(
                            "a complex first argument cannot be combined with imag"
                        )
                    self._real = _real_const(real_value)
                    self._imag = _real_const(embedded_imag)
                else:
                    self._real = _real_const(real_value)
            if not hasattr(self, "_imag"):
                self._imag = imag if isinstance(imag, Waveform) else _real_const(imag)
        if not isinstance(self._real, Waveform) or not isinstance(self._imag, Waveform):
            raise TypeError("ComplexWaveform components must be real Waveform objects")
        self.max = inf
        self.min = -inf
        self.start = None
        self.stop = None
        self.sample_rate = None
        self.filters = None
        self.label = None

    @classmethod
    def _from_components(cls, real, imag):
        return cls(real, imag)

    @property
    def real(self):
        return self._real

    @property
    def imag(self):
        return self._imag

    @property
    def begin(self):
        value = min(self._real.begin, self._imag.begin)
        return value if self.start is None else max(self.start, value)

    @property
    def end(self):
        value = max(self._real.end, self._imag.end)
        return value if self.stop is None else min(self.stop, value)

    def to_bytes(self):
        return _pack_complex(
            _COMPLEX_WAVE_MAGIC, self._real.to_bytes(), self._imag.to_bytes()
        )

    def sample_iq(self, sample_rate=None, out=None, chunk_size=None,
                  function_lib=None, filters=None, dtype=np.int16,
                  full_scale=1.0):
        return _sample_iq(
            self, sample_rate, out, chunk_size, function_lib, filters,
            dtype, full_scale,
        )

    @classmethod
    def from_bytes(cls, data):
        real_data, imag_data = _unpack_complex(data, _COMPLEX_WAVE_MAGIC)
        return cls(Waveform.from_bytes(real_data), Waveform.from_bytes(imag_data))

    def simplify(self, eps=1e-15):
        return ComplexWaveform(self._real.simplify(eps),
                               self._imag.simplify(eps))

    def filter(self, low=0, high=inf, eps=1e-15):
        return ComplexWaveform(self._real.filter(low, high, eps),
                               self._imag.filter(low, high, eps))

    def __pow__(self, n):
        if not isinstance(n, int) or n < 0:
            raise ValueError("complex waveform powers must be non-negative integers")
        result = _real_const(1.0)
        base = self
        while n:
            if n & 1:
                result = result * base
            n >>= 1
            if n:
                base = base * base
        return result

    def __add__(self, other):
        if isinstance(other, ComplexWaveVStack):
            return other + self
        if isinstance(other, WaveVStack):
            return ComplexWaveVStack(other) + self
        if isinstance(other, ComplexWaveform):
            return ComplexWaveform(self._real + other._real,
                                   self._imag + other._imag)
        if isinstance(other, Waveform):
            return ComplexWaveform(self._real + other, self._imag)
        real, imag = _number_parts(other)
        return ComplexWaveform(self._real + real, self._imag + imag)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, ComplexWaveVStack):
            return other * self
        if isinstance(other, WaveVStack):
            return ComplexWaveVStack(other * self._real,
                                     other * self._imag)
        if isinstance(other, ComplexWaveform):
            return ComplexWaveform(
                self._real * other._real - self._imag * other._imag,
                self._real * other._imag + self._imag * other._real,
            )
        if isinstance(other, Waveform):
            return ComplexWaveform(self._real * other, self._imag * other)
        real, imag = _number_parts(other)
        return ComplexWaveform(
            self._real * real - self._imag * imag,
            self._real * imag + self._imag * real,
        )

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, (Waveform, ComplexWaveform, WaveVStack,
                              ComplexWaveVStack)):
            raise TypeError("division by waveform")
        return self * (1 / other)

    def __neg__(self):
        return ComplexWaveform(-self._real, -self._imag)

    def __rshift__(self, time):
        return ComplexWaveform(self._real >> time, self._imag >> time)

    def __lshift__(self, time):
        return self >> -time

    @property
    def marker(self):
        return self._real.marker | self._imag.marker

    def mask(self, edge=0):
        return self.marker.mask(edge)

    def __or__(self, other):
        if not isinstance(other, (Waveform, ComplexWaveform, WaveVStack,
                                  ComplexWaveVStack)):
            other = const(other)
        return self.marker | other.marker

    def __ior__(self, other):
        return self | other

    def __and__(self, other):
        if not isinstance(other, (Waveform, ComplexWaveform, WaveVStack,
                                  ComplexWaveVStack)):
            other = const(other)
        return self.marker & other.marker

    def __iand__(self, other):
        return self & other

    def __call__(self, x, frag=False, out=None, accumulate=False,
                 function_lib=None):
        if function_lib is not None:
            raise NotImplementedError("custom waveform functions are not supported")
        scalar = isinstance(x, (int, float, np.number))
        values = np.asarray([x]) if scalar else np.asarray(x)
        if frag:
            result = self(values)
            parts = [] if not np.any(result) else [(0, len(values), result)]
            if out is None:
                return parts
            if not accumulate:
                out.clear()
            out.extend(parts)
            return out

        if out is None:
            out = np.zeros(values.shape, dtype=np.complex128)
        elif not np.iscomplexobj(out):
            raise TypeError("complex waveform output must have a complex dtype")
        elif not accumulate:
            out[...] = 0

        should_clip = self.min != -inf or self.max != inf
        if should_clip:
            real = np.clip(self._real(values), self.min, self.max)
            imag = np.clip(self._imag(values), self.min, self.max)
            if accumulate:
                out.real[...] += real
                out.imag[...] += imag
            else:
                out.real[...] = real
                out.imag[...] = imag
        else:
            self._real(values, out=out.real, accumulate=True)
            self._imag(values, out=out.imag, accumulate=True)
        return out[0] if scalar else out

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, (int, float, complex, np.number)):
            real, imag = _number_parts(other)
            other = ComplexWaveform(real, imag)
        elif isinstance(other, Waveform):
            other = ComplexWaveform(other, _real_const(0))
        if not isinstance(other, ComplexWaveform):
            return False
        if (self.max, self.min, self.start, self.stop) != (
                other.max, other.min, other.start, other.stop):
            return False
        return self._real == other._real and self._imag == other._imag

    def __hash__(self):
        if (self.max, self.min, self.start, self.stop) == (
                inf, -inf, None, None) and self._imag == 0:
            return hash(self._real)
        return hash((self._real, self._imag, self.max, self.min,
                     self.start, self.stop))

    def __repr__(self):
        return (f"ComplexWaveform(real={self._real!r}, imag={self._imag!r}, "
                f"bytes={len(self.to_bytes())})")

    def _repr_latex_(self):
        return rf"f_I(t)+i f_Q(t)\quad\mathrm{{on}}\ [{self.begin:g},\,{self.end:g}]"

    def __getstate__(self):
        return (self.to_bytes(), self.max, self.min, self.start, self.stop,
                self.sample_rate, self.filters, self.label)

    def __setstate__(self, state):
        (data, self.max, self.min, self.start, self.stop, self.sample_rate,
         self.filters, self.label) = state
        restored = type(self).from_bytes(data)
        self._real = restored._real
        self._imag = restored._imag


class WaveVStack(_SamplingMixin):
    __slots__ = (
        "_stack", "start", "stop", "sample_rate", "offset", "shift",
        "filters", "label", "function_lib", "_sample_plan_cache",
    )

    def __init__(self, wlist=()):
        if isinstance(wlist, WaveVStack):
            self._stack = wlist._stack
        else:
            events = []
            for wav in wlist:
                if isinstance(wav, ComplexWaveform):
                    raise TypeError(
                        "WaveVStack is real-only; use ComplexWaveVStack"
                    )
                if not isinstance(wav, Waveform):
                    raise TypeError("WaveVStack accepts Waveform objects")
                events.append((wav._core, wav._delay, wav._scale))
            self._stack = PackedStack.from_events(events)
        self.start = None
        self.stop = None
        self.sample_rate = None
        self.offset = 0
        self.shift = 0
        self.filters = None
        self.label = None
        self.function_lib = None
        self._sample_plan_cache = None

    @classmethod
    def _from_stack(cls, stack):
        result = cls()
        result._stack = stack
        return result

    @property
    def wlist(self):
        return [Waveform._from_core(core, delay, scale)
                for core, delay, scale in self._stack.events()]

    @property
    def begin(self):
        events = self._stack.events()
        if events:
            value = min(Waveform._from_core(core, delay + self.shift, scale).begin
                        for core, delay, scale in events)
        else:
            value = -inf
        return value if self.start is None else max(self.start, value)

    @property
    def end(self):
        events = self._stack.events()
        if events:
            value = max(Waveform._from_core(core, delay + self.shift, scale).end
                        for core, delay, scale in events)
        else:
            value = inf
        return value if self.stop is None else min(self.stop, value)

    def __call__(self, x, frag=False, out=None, accumulate=False,
                 function_lib=None):
        if frag:
            raise AssertionError("WaveVStack does not support frag mode")
        if function_lib is not None or self.function_lib is not None:
            raise NotImplementedError("custom waveform functions are not supported")
        scalar = isinstance(x, (int, float, complex, np.number))
        values = self._stack.evaluate(
            np.asarray([x]) if scalar else np.asarray(x), self.offset, self.shift
        )
        if out is not None:
            if accumulate:
                out[...] += values
            else:
                out[...] = values
            values = out
        return values[0] if scalar else values

    def _template_sample_plan(self, start_tick, count, step_numerator,
                              index_offset=0):
        window_start = start_tick + index_offset * step_numerator
        cache_key = (
            window_start, count, step_numerator, self.shift,
        )
        if (self._sample_plan_cache is not None
                and self._sample_plan_cache[0] == cache_key):
            return self._sample_plan_cache[1]

        groups = {}
        support_cache = {}
        non_overlapping = True
        previous_start = -1
        previous_stop = -1
        for core, delay, scale in self._stack.events():
            if scale == 0:
                continue
            delay_tick = time_to_tick(delay + self.shift)
            support = support_cache.get(core)
            if support is None:
                support = core.support_ticks()
                support_cache[core] = support
            lower_tick, upper_tick = support
            if lower_tick >= upper_tick:
                continue
            if lower_tick == -_INF_TICK or upper_tick == _INF_TICK:
                self._sample_plan_cache = cache_key, None
                return None

            phase = (window_start - delay_tick) % step_numerator
            key = (core, phase)
            group = groups.get(key)
            if group is None:
                first_tick = lower_tick + (
                    (phase - lower_tick) % step_numerator
                )
                template_count = max(
                    0,
                    (upper_tick - first_tick + step_numerator - 1)
                    // step_numerator,
                )
                template_grid = sample_grid(
                    first_tick, template_count, step_numerator, 1
                )
                group = [first_tick, core.evaluate(template_grid), [], []]
                groups[key] = group
            first_tick, template, destinations, scales = group
            destination = (
                delay_tick + first_tick - window_start
            ) // step_numerator
            start = max(0, destination)
            stop = min(count, destination + len(template))
            if start < stop:
                if start < previous_start or start < previous_stop:
                    non_overlapping = False
                previous_start = start
                previous_stop = max(previous_stop, stop)
                destinations.append(destination)
                scales.append(scale)

        compiled = []
        for _, template, destinations, scales in groups.values():
            destinations = np.asarray(destinations, dtype=np.int64)
            scales = np.asarray(scales, dtype=np.float64)
            scale_groups = None
            if len(scales):
                unique_scales = np.unique(scales)
                if len(unique_scales) <= 64:
                    scale_groups = tuple(
                        (float(scale), destinations[scales == scale])
                        for scale in unique_scales
                    )
            compiled.append(
                (template, destinations, scales, scale_groups)
            )
        plan = tuple(compiled), non_overlapping
        self._sample_plan_cache = cache_key, plan
        return plan

    def _sample_supported(self, start_tick, count, step_numerator,
                          step_denominator, index_offset=0):
        # The common 120 GHz/device-rate case evaluates each distinct pulse
        # only once per grid phase.  The compiled placement plan is retained
        # for repeated float or integer sampling.
        if step_denominator != 1:
            return super()._sample_supported(
                start_tick, count, step_numerator, step_denominator,
                index_offset,
            )
        plan = self._template_sample_plan(
            start_tick, count, step_numerator, index_offset
        )
        if plan is None:
            return super()._sample_supported(
                start_tick, count, step_numerator, 1, index_offset
            )
        groups, _ = plan
        result = np.full(count, self.offset, dtype=np.float64)
        for template, destinations, scales, _ in groups:
            accumulate_template(result, template, destinations, scales)
        return result

    def _sample_supported_quantized(
            self, start_tick, count, step_numerator, step_denominator,
            bits, full_scale, out=None):
        """Write separated repeated templates directly to integer output."""
        if step_denominator != 1:
            return NotImplemented
        plan = self._template_sample_plan(
            start_tick, count, step_numerator
        )
        if plan is None:
            return NotImplemented
        groups, non_overlapping = plan
        if not non_overlapping:
            return NotImplemented

        base = quantize_samples(
            np.asarray([self.offset]), bits, full_scale
        )[0]
        output = _integer_output(count, bits, out, fill=base)
        for template, destinations, scales, scale_groups in groups:
            if scale_groups is not None:
                for scale, scale_destinations in scale_groups:
                    quantized = quantize_samples(
                        self.offset + scale * template,
                        bits, full_scale,
                    )
                    place_quantized_template(
                        output, quantized, scale_destinations
                    )
            elif len(destinations):
                place_template_quantized(
                    output, template, destinations, scales,
                    self.offset, full_scale,
                )
        return output

    def to_bytes(self):
        if self.shift == 0 and self.offset == 0:
            return self._stack.to_bytes()
        events = [(core, quantize_time(delay + self.shift), scale)
                  for core, delay, scale in self._stack.events()]
        if self.offset != 0:
            events.append((constant(self.offset), 0.0, 1.0))
        return PackedStack.from_events(events).to_bytes()

    @classmethod
    def from_bytes(cls, data):
        if bytes(data[:4]) == _COMPLEX_STACK_MAGIC:
            return ComplexWaveVStack.from_bytes(data)
        return cls._from_stack(PackedStack.from_bytes(data))

    def simplify(self, eps=1e-15):
        wav = Waveform._from_core(
            self._stack.simplified(self.shift, self.offset, eps)
        )
        return _copy_sampling_metadata(self, wav)

    def __rshift__(self, time):
        result = self._from_stack(self._stack)
        _copy_sampling_metadata(self, result)
        result.offset = self.offset
        result.shift = quantize_time(self.shift + time)
        return result

    def __lshift__(self, time):
        return self >> -time

    def __add__(self, other):
        if isinstance(other, ComplexWaveVStack):
            return other + self
        if isinstance(other, ComplexWaveform):
            return ComplexWaveVStack(self) + other
        if isinstance(other, WaveVStack):
            left = [(core, delay + self.shift, scale)
                    for core, delay, scale in self._stack.events()]
            right = [(core, delay + other.shift, scale)
                     for core, delay, scale in other._stack.events()]
            result = self._from_stack(PackedStack.from_events((*left, *right)))
            result.offset = self.offset + other.offset
        elif isinstance(other, Waveform):
            events = [*self._stack.events(),
                      (other._core, other._delay - self.shift, other._scale)]
            result = self._from_stack(PackedStack.from_events(events))
            result.offset = self.offset
            result.shift = self.shift
        else:
            real, imag = _number_parts(other)
            if imag != 0:
                return ComplexWaveVStack(self + real,
                                         WaveVStack() + imag)
            result = self._from_stack(self._stack)
            result.offset = self.offset + real
            result.shift = self.shift
        result.filters = self.filters
        result.label = self.label
        return result

    def __radd__(self, value):
        return self + value

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, value):
        return (-self) + value

    def __mul__(self, other):
        if isinstance(other, ComplexWaveVStack):
            return other * self
        if isinstance(other, ComplexWaveform):
            return ComplexWaveVStack(self * other.real,
                                     self * other.imag)
        if not isinstance(other, Waveform):
            real, imag = _number_parts(other)
            if imag != 0:
                return ComplexWaveVStack(self * real, self * imag)
            result = self._from_stack(self._stack.scaled(real))
            result.offset = self.offset * real
            result.shift = self.shift
            result.filters = self.filters
            result.label = self.label
            return result

        waves = [Waveform._from_core(core, delay + self.shift, scale) * other
                 for core, delay, scale in self._stack.events()]
        if self.offset != 0:
            waves.append(self.offset * other)
        result = WaveVStack(waves)
        result.filters = self.filters
        result.label = self.label
        return result

    def __rmul__(self, value):
        return self * value

    def __truediv__(self, other):
        if isinstance(other, (Waveform, ComplexWaveform, WaveVStack,
                              ComplexWaveVStack)):
            raise TypeError("division by waveform")
        return self * (1 / other)

    def __neg__(self):
        return self * -1

    def __pow__(self, n):
        return self.simplify() ** n

    def filter(self, low=0, high=inf, eps=1e-15):
        return self.simplify(eps).filter(low, high, eps)

    @property
    def marker(self):
        return self.simplify().marker

    def mask(self, edge=0):
        return self.simplify().mask(edge)

    def __or__(self, other):
        return self.simplify() | other

    def __and__(self, other):
        return self.simplify() & other

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, ComplexWaveVStack):
            return other == self
        if isinstance(other, ComplexWaveform):
            return other == self.simplify()
        if isinstance(other, WaveVStack):
            return self.simplify() == other.simplify()
        return self.simplify() == other

    __hash__ = None

    def __repr__(self):
        return (f"WaveVStack(events={len(self._stack.events())}, "
                f"bytes={len(self.to_bytes())})")

    def _repr_latex_(self):
        return rf"\sum_{{i=1}}^{{{len(self._stack.events())}}}f_i(t)"

    def __getstate__(self):
        return (self._stack.to_bytes(), self.start, self.stop,
                self.sample_rate, self.offset, self.shift, self.filters,
                self.label)

    def __setstate__(self, state):
        (data, self.start, self.stop, self.sample_rate, self.offset,
         self.shift, self.filters, self.label) = state
        self._stack = PackedStack.from_bytes(data)
        self.function_lib = None
        self._sample_plan_cache = None


class ComplexWaveVStack(_SamplingMixin):
    """Complex stack represented by two real packed stacks."""

    __slots__ = (
        "_real_stack", "_imag_stack", "start", "stop", "sample_rate",
        "offset", "shift", "filters", "label", "function_lib",
    )

    def __init__(self, wlist=(), imag=None):
        if isinstance(wlist, ComplexWaveVStack) and imag is None:
            self._real_stack = wlist._real_stack
            self._imag_stack = wlist._imag_stack
            self.offset = wlist.offset
            self.shift = wlist.shift
        elif isinstance(wlist, ComplexWaveform) and imag is None:
            real_stack = WaveVStack((wlist.real,))
            imag_stack = WaveVStack((wlist.imag,))
            self._real_stack, real_offset = self._normalize_stack(real_stack)
            self._imag_stack, imag_offset = self._normalize_stack(imag_stack)
            self.offset = complex(real_offset, imag_offset)
            self.shift = 0.0
        elif imag is not None or isinstance(wlist, (Waveform, WaveVStack)):
            real_stack = self._coerce_stack(wlist)
            imag_stack = self._coerce_stack(WaveVStack() if imag is None else imag)
            self._real_stack, real_offset = self._normalize_stack(real_stack)
            self._imag_stack, imag_offset = self._normalize_stack(imag_stack)
            self.offset = complex(real_offset, imag_offset)
            self.shift = 0.0
        else:
            real_waves = []
            imag_waves = []
            for wav in wlist:
                if isinstance(wav, ComplexWaveform):
                    real_waves.append(wav.real)
                    imag_waves.append(wav.imag)
                elif isinstance(wav, Waveform):
                    real_waves.append(wav)
                else:
                    raise TypeError(
                        "ComplexWaveVStack accepts Waveform and ComplexWaveform objects"
                    )
            self._real_stack = WaveVStack(real_waves)
            self._imag_stack = WaveVStack(imag_waves)
            self.offset = 0j
            self.shift = 0.0
        self.start = None
        self.stop = None
        self.sample_rate = None
        self.filters = None
        self.label = None
        self.function_lib = None

    @staticmethod
    def _coerce_stack(value):
        if isinstance(value, WaveVStack):
            return value
        if isinstance(value, Waveform):
            return WaveVStack((value,))
        raise TypeError("complex stack components must be real waveforms or stacks")

    @staticmethod
    def _normalize_stack(stack):
        events = tuple((core, quantize_time(delay + stack.shift), scale)
                       for core, delay, scale in stack._stack.events())
        return WaveVStack._from_stack(PackedStack.from_events(events)), stack.offset

    @property
    def real(self):
        result = WaveVStack._from_stack(self._real_stack._stack)
        result.offset = self.offset.real
        result.shift = self.shift
        return _copy_sampling_metadata(self, result)

    @property
    def imag(self):
        result = WaveVStack._from_stack(self._imag_stack._stack)
        result.offset = self.offset.imag
        result.shift = self.shift
        return _copy_sampling_metadata(self, result)

    @property
    def wlist(self):
        return [*self.real.wlist,
                *(ComplexWaveform(_real_const(0), wav)
                  for wav in self.imag.wlist)]

    @property
    def begin(self):
        values = []
        if self._real_stack._stack.events() or self.offset.real != 0:
            values.append(self.real.begin)
        if self._imag_stack._stack.events() or self.offset.imag != 0:
            values.append(self.imag.begin)
        value = min(values) if values else inf
        return value if self.start is None else max(self.start, value)

    @property
    def end(self):
        values = []
        if self._real_stack._stack.events() or self.offset.real != 0:
            values.append(self.real.end)
        if self._imag_stack._stack.events() or self.offset.imag != 0:
            values.append(self.imag.end)
        value = max(values) if values else -inf
        return value if self.stop is None else min(self.stop, value)

    def __call__(self, x, frag=False, out=None, accumulate=False,
                 function_lib=None):
        if frag:
            raise AssertionError("ComplexWaveVStack does not support frag mode")
        if function_lib is not None or self.function_lib is not None:
            raise NotImplementedError("custom waveform functions are not supported")
        scalar = isinstance(x, (int, float, np.number))
        values = np.asarray([x]) if scalar else np.asarray(x)
        if out is None:
            out = np.zeros(values.shape, dtype=np.complex128)
        elif not np.iscomplexobj(out):
            raise TypeError("complex waveform output must have a complex dtype")
        elif not accumulate:
            out[...] = 0
        self.real(values, out=out.real, accumulate=True)
        self.imag(values, out=out.imag, accumulate=True)
        return out[0] if scalar else out

    def to_bytes(self):
        return _pack_complex(
            _COMPLEX_STACK_MAGIC, self.real.to_bytes(), self.imag.to_bytes()
        )

    def sample_iq(self, sample_rate=None, out=None, chunk_size=None,
                  function_lib=None, filters=None, dtype=np.int16,
                  full_scale=1.0):
        return _sample_iq(
            self, sample_rate, out, chunk_size, function_lib, filters,
            dtype, full_scale,
        )

    @classmethod
    def from_bytes(cls, data):
        real_data, imag_data = _unpack_complex(data, _COMPLEX_STACK_MAGIC)
        return cls(WaveVStack.from_bytes(real_data),
                   WaveVStack.from_bytes(imag_data))

    def simplify(self, eps=1e-15):
        return ComplexWaveform(self.real.simplify(eps),
                               self.imag.simplify(eps))

    def __rshift__(self, time):
        result = ComplexWaveVStack(self)
        _copy_sampling_metadata(self, result)
        result.shift = quantize_time(self.shift + time)
        return result

    def __lshift__(self, time):
        return self >> -time

    def __add__(self, other):
        if isinstance(other, ComplexWaveVStack):
            result = ComplexWaveVStack(self.real + other.real,
                                       self.imag + other.imag)
        elif isinstance(other, WaveVStack):
            result = ComplexWaveVStack(self.real + other, self.imag)
        elif isinstance(other, ComplexWaveform):
            result = ComplexWaveVStack(self.real + other.real,
                                       self.imag + other.imag)
        elif isinstance(other, Waveform):
            result = ComplexWaveVStack(self.real + other, self.imag)
        else:
            real, imag = _number_parts(other)
            result = ComplexWaveVStack(self.real + real, self.imag + imag)
        result.filters = self.filters
        result.label = self.label
        return result

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, ComplexWaveVStack):
            raise TypeError("multiplication of two waveform stacks is unsupported")
        if isinstance(other, WaveVStack):
            raise TypeError("multiplication of two waveform stacks is unsupported")
        if isinstance(other, ComplexWaveform):
            result = ComplexWaveVStack(
                self.real * other.real - self.imag * other.imag,
                self.real * other.imag + self.imag * other.real,
            )
        elif isinstance(other, Waveform):
            result = ComplexWaveVStack(self.real * other,
                                       self.imag * other)
        else:
            real, imag = _number_parts(other)
            result = ComplexWaveVStack(
                self.real * real - self.imag * imag,
                self.real * imag + self.imag * real,
            )
        result.filters = self.filters
        result.label = self.label
        return result

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, (Waveform, ComplexWaveform, WaveVStack,
                              ComplexWaveVStack)):
            raise TypeError("division by waveform")
        return self * (1 / other)

    def __neg__(self):
        return ComplexWaveVStack(-self.real, -self.imag)

    def __pow__(self, n):
        return self.simplify() ** n

    def filter(self, low=0, high=inf, eps=1e-15):
        return self.simplify(eps).filter(low, high, eps)

    @property
    def marker(self):
        return self.real.marker | self.imag.marker

    def mask(self, edge=0):
        return self.marker.mask(edge)

    def __or__(self, other):
        return self.marker | other

    def __and__(self, other):
        return self.marker & other

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, ComplexWaveVStack):
            return self.simplify() == other.simplify()
        if isinstance(other, WaveVStack):
            return self.simplify() == other.simplify()
        return self.simplify() == other

    __hash__ = None

    def __repr__(self):
        return (f"ComplexWaveVStack(real_events={len(self._real_stack.wlist)}, "
                f"imag_events={len(self._imag_stack.wlist)}, "
                f"bytes={len(self.to_bytes())})")

    def _repr_latex_(self):
        return rf"\sum f_I(t)+i\sum f_Q(t)\quad\mathrm{{on}}\ [{self.begin:g},\,{self.end:g}]"

    def __getstate__(self):
        return (self.to_bytes(), self.start, self.stop, self.sample_rate,
                self.filters, self.label)

    def __setstate__(self, state):
        (data, self.start, self.stop, self.sample_rate,
         self.filters, self.label) = state
        restored = type(self).from_bytes(data)
        self._real_stack = restored._real_stack
        self._imag_stack = restored._imag_stack
        self.offset = restored.offset
        self.shift = restored.shift
        self.function_lib = None


def _real_const(value):
    real, imag = _number_parts(value)
    if imag != 0:
        raise TypeError("real waveform constant cannot have an imaginary part")
    return Waveform._from_core(constant(real))


def zero():
    return _real_const(0)


def one():
    return _real_const(1.0)


def const(value):
    real, imag = _number_parts(value)
    if imag != 0:
        return ComplexWaveform(_real_const(real), _real_const(imag))
    return _real_const(real)


def D(wav: Waveform | ComplexWaveform, d: int = 1):
    if isinstance(wav, ComplexWaveform):
        return ComplexWaveform(D(wav.real, d), D(wav.imag, d))
    if not isinstance(wav, Waveform):
        raise TypeError("D expects a Waveform or ComplexWaveform")
    if d < 0 or not isinstance(d, int):
        raise ValueError("d must be a non-negative integer")
    return Waveform._from_core(wav._materialized_core().derivative(d))


def sign():
    return _piecewise((0, inf), -1, 1)


def step(edge, type="erf"):
    if edge == 0:
        return _piecewise((0, inf), 0, 1)
    if type == "cos":
        rise = _affine_expression(
            COS, pi / edge, shift=0.5 * edge, scale=0.5, offset=0.5
        )
        return _piecewise((-edge / 2, edge / 2, inf), 0, rise, 1)
    if type == "linear":
        rise = _affine_expression(LINEAR, scale=1 / edge, offset=0.5)
        return _piecewise((-edge / 2, edge / 2, inf), 0, rise, 1)
    rise = _affine_expression(ERF, edge / 5, scale=0.5, offset=0.5)
    return _piecewise((-edge, edge, inf), 0, rise, 1)


def square(width, edge=0, type="erf"):
    if width <= 0:
        return zero()
    if edge == 0:
        return _piecewise((-width / 2, width / 2, inf), 0, 1, 0)
    return (step(edge, type=type) << width / 2) - (step(edge, type=type) >> width / 2)


def gaussian(width, plateau=0.0, d=None):
    if width <= 0 and plateau <= 0:
        return zero()
    std_sq2 = width / 3.3302184446307908
    opcode = GAUSSIAN if d is None else D_GAUSSIAN

    def base(shift):
        args = (std_sq2,) if d is None else (std_sq2, d)
        return _expression(opcode, *args, shift=shift)

    if quantize_time(plateau / 2) <= 0:
        return _piecewise((-0.75 * width, 0.75 * width, inf), 0, base(0), 0)
    return _piecewise(
        (-0.75 * width - plateau / 2, -plateau / 2, plateau / 2,
         0.75 * width + plateau / 2, inf),
        0, base(-plateau / 2), 1, base(plateau / 2), 0,
    )


def cos(w, phi=0):
    if w == 0:
        return const(np.cos(phi))
    if w < 0:
        phi = -phi
        w = -w
    return Waveform._from_core(_scalar(COS, w, shift=-phi / w))


def sin(w, phi=0):
    if w == 0:
        return const(np.sin(phi))
    if w < 0:
        phi = -phi + pi
        w = -w
    return Waveform._from_core(_scalar(COS, w, shift=(pi / 2 - phi) / w))


def exp(alpha):
    if np.iscomplexobj(alpha):
        alpha = complex(alpha)
        carrier = cos(alpha.imag) + 1j * sin(alpha.imag)
        return carrier if alpha.real == 0 else exp(alpha.real) * carrier
    return Waveform._from_core(_scalar(EXP, alpha))


def sinc(bw):
    if bw <= 0:
        return zero()
    width = 100 / bw
    return _piecewise(
        (-width / 2, width / 2, inf), 0, _expression(SINC, bw), 0
    )


def cosPulse(width, plateau=0.0):
    if quantize_time(plateau / 2) > 0:
        return square(plateau + width / 2, edge=width / 2, type="cos")
    if width <= 0:
        return zero()
    pulse = _affine_expression(
        COS, 2 * pi / width, scale=0.5, offset=0.5
    )
    return _piecewise((-width / 2, width / 2, inf), 0, pulse, 0)


def hanning(width, plateau=0.0):
    return cosPulse(width, plateau)


def cosh(w):
    return Waveform._from_core(_scalar(COSH, w))


def sinh(w):
    return Waveform._from_core(_scalar(SINH, w))


def coshPulse(width, eps=1.0, plateau=0.0):
    if width <= 0 and plateau <= 0:
        return zero()
    w = eps / width
    amplitude = np.cosh(eps / 2)
    scale = -1 / (amplitude - 1)

    def edge(shift):
        return _affine_expression(
            COSH, w, shift=shift, scale=scale,
            offset=amplitude / (amplitude - 1),
        )

    if plateau == 0 or quantize_time(-plateau / 2) == quantize_time(plateau / 2):
        return _piecewise((-width / 2, width / 2, inf), 0, edge(0), 0)
    return _piecewise(
        (-width / 2 - plateau / 2, -plateau / 2, plateau / 2,
         width / 2 + plateau / 2, inf),
        0, edge(-plateau / 2), 1, edge(plateau / 2), 0,
    )


def general_cosine(duration, *arg):
    coefficients = np.asarray(arg, dtype=float)
    if not len(coefficients):
        return zero()
    coefficients = coefficients / coefficients[::2].sum()
    wav = zero()
    for index, coefficient in enumerate(coefficients, start=1):
        wav += coefficient / 2 * (
            1 - (-1) ** index * cos(index * 2 * pi / duration)
        )
    return wav * square(duration)


def slepian(duration, *arg):
    return general_cosine(duration, *arg)


def mollifier(width, plateau=0.0, d=0):
    if d < 0 or not isinstance(d, int):
        raise ValueError("d must be a non-negative integer")
    if width <= 0:
        raise ValueError("width must be positive")
    if plateau <= 0:
        return _piecewise((-width / 2, width / 2, inf), 0,
                          _expression(MOLLIFIER, width / 2, d), 0)
    return _piecewise(
        (-width / 2 - plateau / 2, -plateau / 2, plateau / 2,
         width / 2 + plateau / 2, inf),
        0, _expression(MOLLIFIER, width / 2, d, shift=-plateau / 2), 1,
        _expression(MOLLIFIER, width / 2, d, shift=plateau / 2), 0,
    )


def poly(a):
    wav = zero()
    variable = t()
    for degree, coefficient in enumerate(a):
        if coefficient:
            wav += coefficient if degree == 0 else coefficient * variable ** degree
    return wav


def t():
    return Waveform._from_core(_scalar(LINEAR))


def drag(freq, width, plateau=0, delta=0, block_freq=None, phase=0, t0=0):
    phase += pi * delta * (width + plateau)
    if plateau <= 0:
        return _piecewise(
            (t0, t0 + width, inf), 0,
            _expression(DRAG, t0, freq, width, delta, block_freq, phase), 0,
        )
    if width <= 0:
        w = 2 * pi * (freq + delta)
        carrier = _expression(
            COS, w, shift=(phase + 2 * pi * delta * t0) / w
        )
        return _piecewise((t0, t0 + plateau, inf), 0, carrier, 0)
    w = 2 * pi * (freq + delta)
    carrier = _expression(COS, w, shift=(phase + 2 * pi * delta * t0) / w)
    return _piecewise(
        (t0, t0 + width / 2, t0 + width / 2 + plateau,
         t0 + width + plateau, inf),
        0, _expression(DRAG, t0, freq, width, delta, block_freq, phase),
        carrier,
        _expression(DRAG, t0 + plateau, freq, width, delta, block_freq,
                    phase - 2 * pi * delta * plateau), 0,
    )


def _block_frequencies(block_freq):
    if block_freq is None:
        return ()
    if np.isscalar(block_freq):
        return (float(block_freq),)
    return tuple(float(value) for value in block_freq)


def drag_sin(freq, width, plateau=0, delta=0, block_freq=None, phase=0,
             t0=0):
    """Return a sine-power DRAG pulse with spectral blocking constraints."""
    if width <= 0:
        raise ValueError("width must be positive")
    phase += pi * delta * (width + plateau)
    core = _expression(
        DRAG_SIN, t0, freq, width, delta, _block_frequencies(block_freq),
        phase, plateau,
    )
    return _piecewise((t0, t0 + width + plateau, inf), 0, core, 0)


def drag_sinx(freq, width, plateau=0, delta=0, block_freq=None, phase=0,
              t0=0, tab=0.618):
    """Return a flat-top sine-power DRAG pulse with polynomial joins."""
    if width <= 0:
        raise ValueError("width must be positive")
    if not 0 < tab <= 1:
        raise ValueError("tab must be in the interval (0, 1]")
    phase += pi * delta * (width + plateau)
    core = _expression(
        DRAG_SINX, t0, freq, width, delta, _block_frequencies(block_freq),
        phase, plateau, tab,
    )
    return _piecewise((t0, t0 + width + plateau, inf), 0, core, 0)


def chirp(f0, f1, T, phi0=0, type="linear"):
    if f0 == f1:
        return sin(f0, phi0)
    if T <= 0:
        raise ValueError("T must be positive")
    if type == "linear":
        core = _expression(LINEARCHIRP, f0, f1, T, phi0)
    elif type in ("exp", "exponential", "geometric"):
        if f0 == 0:
            raise ValueError("f0 must be non-zero")
        core = _expression(EXPONENTIALCHIRP, f0, np.log(f1 / f0) / T, phi0)
    elif type in ("hyperbolic", "hyp"):
        if f0 * f1 == 0:
            return const(np.sin(phi0))
        core = _expression(
            HYPERBOLICCHIRP, f0, (f0 - f1) / (f1 * T), phi0
        )
    else:
        raise ValueError(f"unknown type {type}")
    return _piecewise((0, T, inf), 0, core, 0)


def interp(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) == 0:
        raise ValueError("x and y must be non-empty one-dimensional arrays of equal size")
    if np.iscomplexobj(y):
        return ComplexWaveform(interp(x, y.real), interp(x, y.imag))
    bounds = [x[0]]
    expressions = [0]
    for x1, x2, y1, y2 in zip(x[:-1], x[1:], y[:-1], y[1:]):
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        expressions.append(_affine_expression(
            LINEAR, shift=x1, scale=slope, offset=y1
        ))
        bounds.append(x2)
    bounds.append(inf)
    expressions.append(0)
    return _piecewise(bounds, *expressions).simplify()


def cut(wav, start=None, stop=None, head=None, tail=None, min=None, max=None):
    offset = 0
    if start is not None and head is not None:
        offset = head - wav(start)
    elif stop is not None and tail is not None:
        offset = tail - wav(stop)
    result = wav + offset
    if start is not None:
        result = result * (step(0) >> start)
    if stop is not None:
        result = result * ((1 - step(0)) >> stop)
    if min is not None:
        result.min = min
    if max is not None:
        result.max = max
    return result


def function(fun, *args, start=None, stop=None):
    raise NotImplementedError("custom waveform functions are not supported")


def samplingPoints(start, stop, points):
    points = np.asarray(points)
    if np.iscomplexobj(points):
        return ComplexWaveform(
            samplingPoints(start, stop, points.real),
            samplingPoints(start, stop, points.imag),
        )
    core = _expression(INTERP, start, stop, tuple(points))
    return _piecewise((start, stop, inf), 0, core, 0)


def mixing(I, Q=None, *, phase=0.0, freq=0.0, ratioIQ=1.0,
           phaseDiff=0.0, block_freq=None, DRAGScaling=None):
    if Q is None:
        Q = zero()
    w = 2 * pi * freq
    if freq != 0:
        Iout = I * cos(w, -phase) + Q * sin(w, -phase)
        Qout = -I * sin(w, -phase + phaseDiff) + Q * cos(w, -phase + phaseDiff)
    else:
        Iout = I * np.cos(-phase) + Q * np.sin(-phase)
        Qout = -I * np.sin(-phase) + Q * np.cos(-phase)
    if block_freq is not None and block_freq != freq:
        a = block_freq / (block_freq - freq)
        b = 1 / (block_freq - freq)
        Iout, Qout = (a * Iout + b / (2 * pi) * D(Qout),
                      a * Qout - b / (2 * pi) * D(Iout))
    elif DRAGScaling is not None and DRAGScaling != 0:
        Iout, Qout = ((1 - w * DRAGScaling) * Iout - DRAGScaling * D(Qout),
                      (1 - w * DRAGScaling) * Qout + DRAGScaling * D(Iout))
    return Iout, ratioIQ * Qout


def play(data, rate=48_000):
    import io
    import pyaudio

    data = np.asarray(data)
    maximum = np.max(np.abs(data))
    if maximum > 1:
        data = data / maximum
    buffer = io.BytesIO(np.asarray(2**15 * 0.999 * data, dtype=np.int16).data)
    audio = pyaudio.PyAudio()
    try:
        stream = audio.open(format=pyaudio.paInt16, channels=1,
                            rate=rate, output=True)
        try:
            while block := buffer.read(1024):
                stream.write(block)
        finally:
            stream.stop_stream()
            stream.close()
    finally:
        audio.terminate()


@lru_cache(maxsize=1024)
def wave_eval(expr: str) -> Waveform | ComplexWaveform:
    """Parse an expression directly against the packed waveform backend."""
    import sys

    from .waveform_parser import WaveformParseError, parse_waveform_expression

    try:
        return parse_waveform_expression(expr, backend=sys.modules[__name__],
                                         extra_modules=())
    except WaveformParseError as exc:
        raise SyntaxError(f"Failed to parse expression {expr!r}: {exc}") from exc


__all__ = [
    "D", "ComplexWaveform", "ComplexWaveVStack", "Waveform", "WaveVStack",
    "chirp", "const", "cos", "cosh",
    "coshPulse", "cosPulse", "cut", "drag", "drag_sin", "drag_sinx",
    "exp", "function",
    "gaussian", "general_cosine", "get_time_resolution", "hanning",
    "interp", "mixing", "mollifier", "one", "play", "poly",
    "registerBaseFunc", "registerDerivative", "samplingPoints",
    "set_time_resolution", "sign", "sin", "sinc", "sinh", "slepian",
    "square", "step", "t", "wave_eval", "zero", "e", "inf", "pi",
]
