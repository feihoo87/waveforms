"""Unified waveform API backed by immutable binary signal blocks.

``Waveform`` is the common base for real, complex, and stacked signals.
Common real expressions and repeated-pulse stacks use the compact C core;
complex signals are represented as explicit pairs of real C blocks.
"""

from __future__ import annotations

import io
import struct
from functools import lru_cache
from typing import Iterable, cast

import numpy as np
from numpy import e, inf, pi
from scipy.signal import sosfilt

from .nonlinear import NonlinearMap

from ._waveform import (
    COS, COSH, D_GAUSSIAN, DRAG, DRAG_SIN, DRAG_SINX, ERF, EXP,
    EXPONENTIALCHIRP, GAUSSIAN,
    HYPERBOLICCHIRP, INTERP, LINEAR, LINEARCHIRP, MOLLIFIER, SINC, SINH,
    accumulate_template, get_time_resolution,
    lock_time_resolution,
    place_quantized_template, place_template_quantized,
    quantize_samples, quantize_time, registerBaseFunc, sosfilt_samples,
    registerDerivative, sample_clock, sample_grid,
    set_time_resolution as _set_time_resolution,
    tick_to_time, time_to_tick,
)
from ._waveform import (
    CWaveformCore as _CWaveformCore,
    CWaveformStackCore as _CWaveformStackCore,
    TICKS_PER_SECOND as _CORE_TICKS_PER_SECOND,
    set_c_ticks_per_second as _set_c_ticks_per_second,
)

_ZERO_EXPR = ((), ())
_ONE_EXPR = ((((), ()),), (1.0,))
_COMPLEX_WAVE_MAGIC = b"CWF1"
_COMPLEX_STACK_MAGIC = b"CWS1"
_COMPLEX_HEADER = struct.Struct("<4sII")
_INF_TICK = np.iinfo(np.int64).max
_C_CLOCK_ACTIVE = get_time_resolution() == 1 / _CORE_TICKS_PER_SECOND
_C_CLOCK_LOCKED = False


def set_time_resolution(value):
    """Configure the global clock before the first waveform is created."""
    global _C_CLOCK_ACTIVE, _CORE_TICKS_PER_SECOND
    reciprocal = 1.0 / float(value)
    ticks_per_second = int(round(reciprocal))
    if not np.isclose(reciprocal, ticks_per_second, rtol=1e-15, atol=0.0):
        raise ValueError(
            "the C core requires a global tick that divides one second"
        )
    _set_time_resolution(value)
    _set_c_ticks_per_second(ticks_per_second)
    _CORE_TICKS_PER_SECOND = ticks_per_second
    _C_CLOCK_ACTIVE = True


def _ensure_c_clock_locked():
    global _C_CLOCK_LOCKED
    if not _C_CLOCK_LOCKED:
        lock_time_resolution()
        _C_CLOCK_LOCKED = True


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
    target.min = source.min
    target.max = source.max
    target.start = source.start
    target.stop = source.stop
    target.sample_rate = source.sample_rate
    target.filters = source.filters
    target.nonlinear = source.nonlinear
    target.label = source.label
    return target


def _scalar(opcode, *args, shift=0.0):
    parameters = _builtin_parameters(opcode, args)
    return _CWaveformCore.builtin(
        opcode, parameters, _time_to_c_tick(shift)
    )


def _expression(opcode, *args, shift=0.0):
    return RealWaveform._from_core(_scalar(opcode, *args, shift=shift))


def _affine_expression(opcode, *args, shift=0.0, scale=1.0, offset=0.0):
    return scale * _expression(opcode, *args, shift=shift) + offset


def _piecewise(bounds, *expressions):
    if len(bounds) != len(expressions) or not bounds or bounds[-1] != inf:
        raise ValueError("bounds and expressions must have equal lengths ending at +inf")
    result = _c_zero()
    lower = -inf
    for upper, expression in zip(bounds, expressions):
        wave = expression if isinstance(expression, RealWaveform) else _c_const(expression)
        if not isinstance(wave, RealWaveform):
            raise TypeError("piecewise expressions must be real C waveforms")
        if not wave._is_zero() and lower < upper:
            core = wave._materialized_core().window(
                _time_to_c_tick(lower), _time_to_c_tick(upper)
            )
            result = result + RealWaveform._from_core(core)
        lower = upper
    return result


def _builtin_parameters(opcode, args):
    """Flatten public builtin arguments into the language-neutral C ABI."""
    def time_value(value):
        return _c_tick_to_time(_time_to_c_tick(value))

    if opcode == INTERP:
        start, stop, points = args
        return (time_value(start), time_value(stop),
                *(float(value) for value in points))
    if opcode == DRAG:
        t0, freq, width, delta, block_freq, phase = args
        if block_freq is not None and not np.isscalar(block_freq):
            values = tuple(block_freq)
            if len(values) > 1:
                raise ValueError("drag accepts at most one block frequency")
            block_freq = None if not values else values[0]
        return (time_value(t0), float(freq), time_value(width), float(delta),
                np.nan if block_freq is None else float(block_freq),
                float(phase))
    if opcode in (DRAG_SIN, DRAG_SINX):
        t0, freq, width, delta, block_freq, phase, plateau = args[:7]
        tab = np.nan if opcode == DRAG_SIN else float(args[7])
        frequencies = () if block_freq is None else tuple(block_freq)
        return (time_value(t0), float(freq), time_value(width), float(delta),
                float(phase), time_value(plateau), tab,
                *(float(value) for value in frequencies))
    if opcode in (GAUSSIAN, ERF, MOLLIFIER, D_GAUSSIAN):
        return (time_value(args[0]), *(float(value) for value in args[1:]))
    if opcode == LINEARCHIRP:
        return (float(args[0]), float(args[1]), time_value(args[2]),
                float(args[3]))
    return tuple(float(value) for value in args)


def _legacy_expression_to_c(expression):
    """Translate the historic public ``seq`` expression into the C core."""
    if isinstance(expression, RealWaveform):
        return expression
    if isinstance(expression, (int, float, np.number)):
        return _c_const(expression)
    terms, coefficients = expression
    result = _c_zero()
    for (functions, powers), coefficient in zip(terms, coefficients):
        term = _c_const(coefficient)
        for function, power in zip(functions, powers):
            opcode, *args, shift = function
            term = term * (_expression(opcode, *args, shift=shift) ** int(power))
        result = result + term
    return result


def _is_tick_aligned(value, tick):
    reconstructed = tick_to_time(tick)
    tolerance = max(get_time_resolution() * 1e-6,
                    abs(np.spacing(float(value))) * 2)
    return abs(float(value) - reconstructed) <= tolerance


def _sampling_plan(start, stop, sample_rate):
    """Return an integer/rational tick plan for an integral-Hz device rate."""
    rate = float(sample_rate)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("sample_rate must be a finite positive number")
    integer_rate = int(rate)
    if rate != integer_rate:
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


def _array_tick_plan(positions):
    """Recognize a one-dimensional, uniform integer-tick input grid."""
    if positions.ndim != 1 or len(positions) < 2:
        return None
    scaled = positions * _CORE_TICKS_PER_SECOND
    ticks = np.rint(scaled)
    if np.any(np.abs(scaled - ticks) > 1e-5):
        return None
    steps = np.diff(ticks)
    step = int(steps[0])
    if step <= 0 or np.any(steps != step):
        return None
    return int(ticks[0]), len(positions), step


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


def _amplitude_limits(owner):
    minimum, maximum = float(owner.min), float(owner.max)
    if not minimum <= maximum or minimum == inf or maximum == -inf:
        raise ValueError("min and max must bound finite amplitudes, without NaN")
    return minimum, maximum


def _clip_samples(sig, minimum=-inf, maximum=inf):
    """Limit final amplitudes in place, independently for I and Q."""
    if minimum == -inf and maximum == inf:
        return sig
    values = np.asarray(sig)
    if not values.flags.writeable:
        values = values.copy()
    if np.iscomplexobj(values):
        np.clip(values.real, minimum, maximum, out=values.real)
        np.clip(values.imag, minimum, maximum, out=values.imag)
    else:
        np.clip(values, minimum, maximum, out=values)
    return values


def _clip_quantized(sig, bits, full_scale, minimum, maximum):
    """Q(clip(x)) == clip(Q(x), Q(min), Q(max)) for monotone DAC rounding.

    Keep the direct integer fallback for grids without a native sample plan.
    Only the two bounds need floating-point storage.
    """
    if minimum != -inf or maximum != inf:
        bounds = np.clip([minimum, maximum], -full_scale, full_scale)
        lower, upper = quantize_samples(bounds, bits, full_scale)
        np.clip(sig, lower, upper, out=sig)
    return sig


def _finish_samples(sig, dtype, full_scale, out, minimum=-inf, maximum=inf):
    sig = _clip_samples(sig, minimum, maximum)
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
        zi = np.zeros((np.atleast_2d(sos).shape[0], 2),
                      dtype=np.result_type(sos, values))
    filtered, zi = sosfilt(sos, values, zi=zi)
    if initial:
        filtered = filtered + initial
    return filtered, zi


def _filter_and_finish(sig, filters, dtype, full_scale, out,
                       minimum, maximum, zi=None):
    if filters is None:
        return _finish_samples(
            sig, dtype, full_scale, out, minimum, maximum), zi
    sos, initial = filters
    sos = np.asarray(sos)
    initial = initial if initial else 0.0
    values_dtype = np.result_type(sig, np.asarray(initial), sos,
                                  zi if zi is not None else np.float64)
    # Preserve SciPy's complex-coefficient and extended-precision behavior.
    if (sos.dtype.kind not in 'biuf' or sos.dtype.itemsize > 8
            or values_dtype not in (np.dtype(np.float64), np.dtype(np.complex128))):
        sig, zi = _filter_samples(sig, filters, zi)
        return _finish_samples(
            sig, dtype, full_scale, out, minimum, maximum), zi
    dtype, bits = _quantization_bits(dtype, out)
    native_dtype = (np.dtype(np.int16 if bits == 16 else np.int32)
                    if bits is not None else values_dtype)
    if bits is not None:
        # Integer output validation and conversion belong to the native call.
        direct, target = True, out
    elif out is not None:
        output = np.asarray(out)
        direct = ((dtype is None or dtype == native_dtype)
                  and output.dtype == native_dtype
                  and output.flags.c_contiguous and output.flags.aligned)
        target = output if direct else None
    else:
        direct = dtype is None or dtype == native_dtype
        # Sampling owns this buffer; float filtering can safely reuse it.
        target = (sig if (bits is None and sig.dtype == native_dtype
                          and sig.flags.c_contiguous and sig.flags.writeable
                          and sig.flags.aligned) else None)
    sig, zi = sosfilt_samples(
        sig, sos, initial, zi, bits or 0, full_scale,
        minimum, maximum, target)
    if direct:
        return sig, zi
    return _finish_samples(sig, dtype, full_scale, out), zi


def _apply_nonlinear(sig, nonlinear):
    if nonlinear is None:
        return sig
    values = np.asarray(sig)
    if np.iscomplexobj(values):
        if (not isinstance(nonlinear, (tuple, list))
                or len(nonlinear) != 2):
            raise TypeError(
                "complex waveforms require nonlinear=(real_map, imag_map)"
            )
        real_map, imag_map = nonlinear
        if real_map is not None and not isinstance(real_map, NonlinearMap):
            raise TypeError("real nonlinear component must be NonlinearMap or None")
        if imag_map is not None and not isinstance(imag_map, NonlinearMap):
            raise TypeError("imag nonlinear component must be NonlinearMap or None")
        real = values.real if real_map is None else real_map(values.real)
        imag = values.imag if imag_map is None else imag_map(values.imag)
        return np.asarray(real) + 1j * np.asarray(imag)
    if not isinstance(nonlinear, NonlinearMap):
        raise TypeError("nonlinear must be a NonlinearMap for real waveforms")
    if (values.dtype == np.dtype(np.float64) and values.flags.c_contiguous
            and values.flags.writeable):
        return nonlinear(values, out=values)
    return nonlinear(values)


def _nonlinear_component(nonlinear, index):
    if isinstance(nonlinear, (tuple, list)) and len(nonlinear) == 2:
        return nonlinear[index]
    return None


def _sample_iq(owner, sample_rate=None, out=None, chunk_size=None,
               function_lib=None, filters=None, dtype=np.int16,
               full_scale=1.0, nonlinear=None):
    """Sample a complex wrapper as separate real I/Q buffers."""
    if out is None:
        out_i = out_q = None
    else:
        if not isinstance(out, (tuple, list)) or len(out) != 2:
            raise TypeError("out must be an (I, Q) pair")
        out_i, out_q = out
    values = owner.sample(
        sample_rate=sample_rate, chunk_size=chunk_size,
        function_lib=function_lib, filters=filters, nonlinear=nonlinear,
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


class _WaveformMeta(type):
    """Keep the 2.2.x ``Waveform(...)`` constructor as a real-wave factory."""

    def __call__(cls, *args, **kwargs):
        if cls is Waveform:
            return RealWaveform(*args, **kwargs)
        return super().__call__(*args, **kwargs)


class Waveform(metaclass=_WaveformMeta):
    """Common base class for every real, complex, and stacked waveform."""

    start: float | None
    stop: float | None
    sample_rate: float | None
    filters: tuple[np.ndarray, float] | None
    nonlinear: NonlinearMap | tuple[NonlinearMap | None, NonlinearMap | None] | None
    min: float
    max: float

    def _evaluate_raw(self, x):
        """Evaluate before output limits, nonlinear calibration or filtering."""
        return self(x, _clip=False)

    def is_zero(self):
        """Return whether this object has no non-zero waveform support."""
        return self.begin >= self.end

    @classmethod
    def from_bytes(cls, data):
        """Restore any waveform kind from its canonical binary block."""
        magic = bytes(data[:4])
        if magic in (_COMPLEX_WAVE_MAGIC, b"WNC4"):
            return ComplexWaveform.from_bytes(data)
        if magic == _COMPLEX_STACK_MAGIC:
            return ComplexWaveVStack.from_bytes(data)
        if magic == b"WNS4":
            return RealWaveVStack.from_bytes(data)
        if magic == b"WNF4":
            return RealWaveform.from_bytes(data)
        raise ValueError("unsupported waveform block")

    def sample(self, sample_rate=None, out: np.ndarray | None = None,
               chunk_size=None, function_lib=None,
               filters: tuple[np.ndarray, float] | None = None,
               dtype=None, full_scale=1.0, nonlinear=None):
        minimum, maximum = _amplitude_limits(self)
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
        if nonlinear is None:
            nonlinear = self.nonlinear
        dtype, bits = _quantization_bits(dtype, out)
        if chunk_size is not None:
            return self._sample_iter(
                sample_rate, int(chunk_size), out, filters, dtype, full_scale,
                nonlinear,
            )

        plan = _sampling_plan(self.start, self.stop, sample_rate)
        if plan is None:
            x = np.arange(self.start, self.stop, 1 / float(sample_rate))
            sig = cast(np.ndarray, self._evaluate_raw(x))
        else:
            if bits is not None and filters is None and nonlinear is None:
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
        sig = _apply_nonlinear(sig, nonlinear)
        result, _ = _filter_and_finish(
            sig, filters, dtype, full_scale, out, minimum, maximum)
        return cast(np.ndarray, result)

    def _sample_supported(self, start_tick, count, step_numerator,
                          step_denominator, index_offset=0):
        x = sample_grid(start_tick, count, step_numerator, step_denominator,
                        index_offset)
        return cast(np.ndarray, self._evaluate_raw(x))

    def _sample_iter(self, sample_rate, chunk_size, out, filters, dtype,
                     full_scale, nonlinear):
        minimum, maximum = _amplitude_limits(self)
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
                sig = cast(np.ndarray, self._evaluate_raw(x))
            else:
                sig = self._sample_supported(
                    start_tick, size, step_numerator, step_denominator,
                    output_index,
                )
            target = (None if output is None
                      else output[output_index:output_index + size])
            sig = _apply_nonlinear(sig, nonlinear)
            sig, zi = _filter_and_finish(
                sig, filters, dtype, full_scale, target, minimum, maximum, zi)
            yield sig
            output_index += size

    def _play(self, time_unit, volume):
        import pyaudio  # pyright: ignore[reportMissingModuleSource]

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


class _RealWaveformBase(Waveform):
    """Common real-waveform type; concrete storage is provided by the C core."""

    __slots__ = (
        "_core", "max", "min", "start", "stop", "sample_rate",
        "filters", "nonlinear", "label",
    )


class ComplexWaveform(Waveform):
    """Complex waveform represented by two independent real waveforms."""

    __slots__ = (
        "_real", "_imag", "max", "min", "start", "stop", "sample_rate",
        "filters", "nonlinear", "label",
    )

    def __init__(self, real=0.0, imag=0.0):
        if isinstance(real, ComplexWaveform):
            if imag != 0:
                raise TypeError("imag must be zero when copying ComplexWaveform")
            self._real = real._real
            self._imag = real._imag
        else:
            if isinstance(real, RealWaveform):
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
                self._imag = imag if isinstance(imag, RealWaveform) else _real_const(imag)
        if (not isinstance(self._real, RealWaveform)
                or not isinstance(self._imag, RealWaveform)):
            raise TypeError("ComplexWaveform components must be real Waveform objects")
        self.max = inf
        self.min = -inf
        self.start = None
        self.stop = None
        self.sample_rate = None
        self.filters = None
        self.nonlinear = None
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
                  full_scale=1.0, nonlinear=None):
        return _sample_iq(
            self, sample_rate, out, chunk_size, function_lib, filters,
            dtype, full_scale, nonlinear,
        )

    @classmethod
    def from_bytes(cls, data):
        magic = bytes(data[:4])
        if magic not in (_COMPLEX_WAVE_MAGIC, b"WNC4"):
            raise ValueError("unsupported complex waveform format")
        real_data, imag_data = _unpack_complex(data, magic)
        return cls(Waveform.from_bytes(real_data),
                   Waveform.from_bytes(imag_data))

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
        if isinstance(other, RealWaveform):
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
        if isinstance(other, RealWaveform):
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
                 function_lib=None, *, _clip=True):
        if function_lib is not None:
            raise NotImplementedError("custom waveform functions are not supported")
        scalar = isinstance(x, (int, float, np.number))
        values = np.asarray([x]) if scalar else np.asarray(x)
        if frag:
            result = self(values, _clip=_clip)
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

        should_clip = _clip and (self.min != -inf or self.max != inf)
        if should_clip:
            real = np.clip(self._real(values), self.min, self.max)
            imag = np.clip(self._imag(values), self.min, self.max)
            if accumulate:
                out.real[...] += real
                out.imag[...] += imag
            else:
                out.real[...] = real
                out.imag[...] = imag
        elif (isinstance(self._real, RealWaveform)
              and isinstance(self._imag, RealWaveform)):
            real = self._real._core.evaluate(
                values, self._real._delay_tick, self._real._scale,
                self._real.min if _clip else -inf,
                self._real.max if _clip else inf,
            )
            imag = self._imag._core.evaluate(
                values, self._imag._delay_tick, self._imag._scale,
                self._imag.min if _clip else -inf,
                self._imag.max if _clip else inf,
            )
            if accumulate:
                out.real[...] += real
                out.imag[...] += imag
            else:
                out.real[...] = real
                out.imag[...] = imag
        else:
            self._real(values, out=out.real, accumulate=True, _clip=_clip)
            self._imag(values, out=out.imag, accumulate=True, _clip=_clip)
        return out[0] if scalar else out

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, (int, float, complex, np.number)):
            real, imag = _number_parts(other)
            other = ComplexWaveform(real, imag)
        elif isinstance(other, RealWaveform):
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
                self.sample_rate, self.filters, self.nonlinear, self.label)

    def __setstate__(self, state):
        if len(state) == 8:
            (data, self.max, self.min, self.start, self.stop, self.sample_rate,
             self.filters, self.label) = state
            self.nonlinear = None
        else:
            (data, self.max, self.min, self.start, self.stop, self.sample_rate,
             self.filters, self.nonlinear, self.label) = state
        restored = type(self).from_bytes(data)
        self._real = restored._real
        self._imag = restored._imag


class _WaveVStackMeta(_WaveformMeta):
    """Route the abstract stack constructor to the native real stack."""

    def __call__(cls, *args, **kwargs):
        if cls is WaveVStack:
            return _make_real_stack(*args, **kwargs)
        return super().__call__(*args, **kwargs)


class WaveVStack(Waveform, metaclass=_WaveVStackMeta):
    """Common base class for real and complex waveform stacks."""

    @classmethod
    def from_bytes(cls, data):
        magic = bytes(data[:4])
        if magic == _COMPLEX_STACK_MAGIC:
            return ComplexWaveVStack.from_bytes(data)
        if magic == b"WNS4":
            return RealWaveVStack.from_bytes(data)
        return RealWaveVStack.from_bytes(data)

    @classmethod
    def from_events(cls, templates, template_ids, delay_ticks, scales):
        """Build a real stack from compiler-native event columns.

        ``delay_ticks`` uses the process-wide integer clock.  This entry point
        avoids constructing one temporary Python waveform per event.
        """
        return RealWaveVStack.from_events(
            templates, template_ids, delay_ticks, scales
        )


class _RealWaveVStackBase(WaveVStack):
    """Common real-stack type; concrete storage is provided by the C core."""

    __slots__ = (
        "start", "stop", "sample_rate", "offset", "filters", "nonlinear", "label",
        "function_lib", "_sample_plan_cache", "min", "max",
    )


class ComplexWaveVStack(WaveVStack):
    """Complex stack represented by two real C stacks."""

    __slots__ = (
        "_real_stack", "_imag_stack", "start", "stop", "sample_rate",
        "offset", "shift", "filters", "nonlinear", "label", "function_lib",
        "min", "max",
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
        elif imag is not None or isinstance(wlist, (RealWaveform, WaveVStack)):
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
                elif isinstance(wav, RealWaveform):
                    real_waves.append(wav)
                else:
                    raise TypeError(
                        "ComplexWaveVStack accepts Waveform and ComplexWaveform objects"
                    )
            self._real_stack = WaveVStack(real_waves)
            self._imag_stack = WaveVStack(imag_waves)
            self.offset = 0j
            self.shift = 0.0
        self.min = -inf
        self.max = inf
        self.start = None
        self.stop = None
        self.sample_rate = None
        self.filters = None
        self.nonlinear = None
        self.label = None
        self.function_lib = None

    @staticmethod
    def _coerce_stack(value):
        if isinstance(value, WaveVStack):
            return value
        if isinstance(value, RealWaveform):
            return WaveVStack((value,))
        raise TypeError("complex stack components must be real waveforms or stacks")

    @staticmethod
    def _normalize_stack(stack):
        result = WaveVStack(stack.wlist)
        if stack.shift:
            result = result >> stack.shift
        return result, stack.offset

    @property
    def real(self):
        result = self._real_stack + self.offset.real
        if self.shift:
            result = result >> self.shift
        _copy_sampling_metadata(self, result)
        result.nonlinear = _nonlinear_component(self.nonlinear, 0)
        return result

    @property
    def imag(self):
        result = self._imag_stack + self.offset.imag
        if self.shift:
            result = result >> self.shift
        _copy_sampling_metadata(self, result)
        result.nonlinear = _nonlinear_component(self.nonlinear, 1)
        return result

    @property
    def wlist(self):
        return [*self.real.wlist,
                *(ComplexWaveform(_real_const(0), wav)
                  for wav in self.imag.wlist)]

    @property
    def begin(self):
        values = []
        if self._real_stack.wlist or self.offset.real != 0:
            values.append(self.real.begin)
        if self._imag_stack.wlist or self.offset.imag != 0:
            values.append(self.imag.begin)
        value = min(values) if values else inf
        return value if self.start is None else max(self.start, value)

    @property
    def end(self):
        values = []
        if self._real_stack.wlist or self.offset.real != 0:
            values.append(self.real.end)
        if self._imag_stack.wlist or self.offset.imag != 0:
            values.append(self.imag.end)
        value = max(values) if values else -inf
        return value if self.stop is None else min(self.stop, value)

    def __call__(self, x, frag=False, out=None, accumulate=False,
                 function_lib=None, *, _clip=True):
        if frag:
            raise AssertionError("ComplexWaveVStack does not support frag mode")
        if function_lib is not None or self.function_lib is not None:
            raise NotImplementedError("custom waveform functions are not supported")
        scalar = isinstance(x, (int, float, np.number))
        values = np.asarray([x]) if scalar else np.asarray(x)
        minimum, maximum = _amplitude_limits(self) if _clip else (-inf, inf)
        limited = minimum != -inf or maximum != inf
        target = out
        if out is not None and not np.iscomplexobj(out):
            raise TypeError("complex waveform output must have a complex dtype")
        if out is None or (limited and accumulate):
            out = np.zeros(values.shape, dtype=np.complex128)
        elif not accumulate:
            out[...] = 0
        self.real(values, out=out.real, accumulate=True, _clip=False)
        self.imag(values, out=out.imag, accumulate=True, _clip=False)
        _clip_samples(out, minimum, maximum)
        if target is not None and target is not out:
            target[...] += out
            out = target
        return out[0] if scalar else out

    def to_bytes(self):
        return _pack_complex(
            _COMPLEX_STACK_MAGIC, self.real.to_bytes(), self.imag.to_bytes()
        )

    def sample_iq(self, sample_rate=None, out=None, chunk_size=None,
                  function_lib=None, filters=None, dtype=np.int16,
                  full_scale=1.0, nonlinear=None):
        return _sample_iq(
            self, sample_rate, out, chunk_size, function_lib, filters,
            dtype, full_scale, nonlinear,
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
        elif isinstance(other, RealWaveform):
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
        elif isinstance(other, RealWaveform):
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
                self.filters, self.nonlinear, self.label, self.min, self.max)

    def __setstate__(self, state):
        self.min, self.max = state[7:] if len(state) == 9 else (-inf, inf)
        state = state[:7]
        if len(state) == 6:
            (data, self.start, self.stop, self.sample_rate,
             self.filters, self.label) = state
            self.nonlinear = None
        else:
            (data, self.start, self.stop, self.sample_rate,
             self.filters, self.nonlinear, self.label) = state
        restored = type(self).from_bytes(data)
        self._real_stack = restored._real_stack
        self._imag_stack = restored._imag_stack
        self.offset = restored.offset
        self.shift = restored.shift
        self.function_lib = None


def _time_to_c_tick(value):
    if value == inf:
        return np.iinfo(np.int64).max
    if value == -inf:
        return np.iinfo(np.int64).min
    return int(round(float(value) * _CORE_TICKS_PER_SECOND))


def _c_tick_to_time(value):
    if value == np.iinfo(np.int64).max:
        return inf
    if value == np.iinfo(np.int64).min:
        return -inf
    return int(value) / _CORE_TICKS_PER_SECOND


class RealWaveform(_RealWaveformBase):
    """Thin Python owner for the language-neutral WNF4 C core."""

    __slots__ = ("_delay_tick", "_canonical_core_cache")

    def __init__(self, bounds=(inf,), seq=None, min=-inf, max=inf, *,
                 _core=None, _delay_tick=0, _scale=1.0):
        _ensure_c_clock_locked()
        if _core is None:
            if isinstance(bounds, (int, float, complex, np.number)) and seq is None:
                real, imag = _number_parts(bounds)
                if imag != 0:
                    raise TypeError("RealWaveform is real-only")
                _core = _CWaveformCore.constant(real)
            else:
                bounds = tuple(bounds)
                if seq is None:
                    seq = (_ZERO_EXPR,)
                seq = tuple(seq)
                result = _piecewise(
                    bounds, *(_legacy_expression_to_c(item) for item in seq)
                )
                _core = result._materialized_core()
        self._core = _core
        self._delay_tick = int(_delay_tick)
        self._scale = float(_scale)
        self._canonical_core_cache = None
        self.max = max
        self.min = min
        self.start = None
        self.stop = None
        self.sample_rate = None
        self.filters = None
        self.nonlinear = None
        self.label = None

    @classmethod
    def _from_core(cls, core, delay_tick=0, scale=1.0):
        return cls(_core=core, _delay_tick=delay_tick, _scale=scale)

    def _materialized_core(self):
        if self._delay_tick == 0 and self._scale == 1:
            return self._core
        return self._core.materialize(self._delay_tick, self._scale)

    def _canonical_core(self):
        if self._canonical_core_cache is None:
            self._canonical_core_cache = self._materialized_core().simplify(1e-15)
        return self._canonical_core_cache

    def _is_zero(self):
        return (self._scale == 0.0
                or self._core.lower_tick >= self._core.upper_tick)

    def is_zero(self):
        return self._is_zero()

    @property
    def bounds(self):
        lower = self.begin
        upper = self.end
        if lower == -inf and upper == inf:
            return (inf,)
        if lower == -inf:
            return (upper, inf)
        if upper == inf:
            return (lower, inf)
        return (lower, upper, inf)

    @property
    def seq(self):
        token = ("C", self.to_bytes())
        if self.begin == -inf and self.end == inf:
            return (token,)
        if self.begin == -inf or self.end == inf:
            return (token, 0)
        return (0, token, 0)

    @property
    def begin(self):
        tick = self._core.lower_tick
        if tick != np.iinfo(np.int64).max:
            tick += self._delay_tick
        value = _c_tick_to_time(tick)
        return value if self.start is None else max(self.start, value)

    @property
    def end(self):
        tick = self._core.upper_tick
        if tick != np.iinfo(np.int64).min:
            tick += self._delay_tick
        value = _c_tick_to_time(tick)
        return value if self.stop is None else min(self.stop, value)

    def __call__(self, x, frag=False, out=None, accumulate=False,
                 function_lib=None, *, _clip=True):
        if frag:
            values = self(x, frag=False, _clip=_clip)
            values = np.asarray([values]) if np.isscalar(values) else values
            parts = [] if not np.any(values) else [(0, len(values), values)]
            if out is None:
                return parts
            if not accumulate:
                out.clear()
            out.extend(parts)
            return out
        if function_lib is not None:
            raise NotImplementedError("custom C-core functions are unsupported")
        scalar = isinstance(x, (int, float, np.number))
        raw_positions = np.asarray([x] if scalar else x)
        if np.iscomplexobj(raw_positions):
            raise TypeError("waveform positions must be real")
        positions = np.asarray(raw_positions, dtype=np.float64)
        values = self._core.evaluate(
            positions, self._delay_tick, self._scale,
            self.min if _clip else -inf, self.max if _clip else inf,
        )
        if out is not None:
            if accumulate:
                out[...] += values
            else:
                out[...] = values
            values = out
        return values[0] if scalar else values

    def sample(self, sample_rate=None, out=None, chunk_size=None,
               function_lib=None, filters=None, dtype=None, full_scale=1.0,
               nonlinear=None):
        minimum, maximum = _amplitude_limits(self)
        if chunk_size is not None:
            return Waveform.sample(
                self, sample_rate, out, chunk_size, function_lib, filters,
                dtype, full_scale, nonlinear,
            )
        if function_lib is not None:
            raise NotImplementedError("custom C-core functions are unsupported")
        if sample_rate is None:
            sample_rate = self.sample_rate
        if self.start is None or self.stop is None or sample_rate is None:
            raise ValueError("RealWaveform sampling metadata is incomplete")
        if filters is None:
            filters = self.filters
        if nonlinear is None:
            nonlinear = self.nonlinear
        plan = _sampling_plan(self.start, self.stop, sample_rate)
        dtype, bits = _quantization_bits(dtype, out)
        core_clock = get_time_resolution() == 1 / _CORE_TICKS_PER_SECOND
        if plan is None or not core_clock:
            positions = np.arange(self.start, self.stop, 1 / float(sample_rate))
            values = self._evaluate_raw(positions)
        else:
            start_tick, count, step_numerator, step_denominator = plan
            if filters is None and nonlinear is None:
                values = self._core.sample(
                    start_tick, count, step_numerator, step_denominator,
                    self._delay_tick, self._scale, minimum, maximum,
                    bits or 0, full_scale, out if bits is not None else None,
                )
                return (values if bits is not None else
                        _finish_samples(values, dtype, full_scale, out))
            values = self._core.sample(
                start_tick, count, step_numerator, step_denominator,
                self._delay_tick, self._scale,
            )
        values = _apply_nonlinear(values, nonlinear)
        result, _ = _filter_and_finish(
            values, filters, dtype, full_scale, out, minimum, maximum)
        return result

    def to_bytes(self):
        return self._materialized_core().to_bytes()

    @classmethod
    def from_bytes(cls, data):
        return cls._from_core(_CWaveformCore.from_bytes(data))

    def simplify(self, eps=1e-15):
        return RealWaveform._from_core(
            self._materialized_core().simplify(eps)
        )

    def filter(self, low=0, high=inf, eps=1e-15):
        return RealWaveform._from_core(
            self._materialized_core().filtered(low, high, eps)
        )

    def __pow__(self, n):
        if not isinstance(n, int) or n < 0:
            raise ValueError("non-constant waveform powers must be non-negative integers")
        return RealWaveform._from_core(
            self._materialized_core().power(n)
        )

    def __add__(self, other):
        if isinstance(other, ComplexWaveform):
            return other + self
        if isinstance(other, WaveVStack):
            return other + self
        if not isinstance(other, RealWaveform):
            real, imag = _number_parts(other)
            if imag:
                return ComplexWaveform(self + real, _c_const(imag))
            other = _c_const(real)
        if self._is_zero():
            return RealWaveform._from_core(
                other._core, other._delay_tick, other._scale)
        if other._is_zero():
            return RealWaveform._from_core(
                self._core, self._delay_tick, self._scale)
        return RealWaveform._from_core(self._core.add_affine(
            other._core, self._delay_tick, self._scale,
            other._delay_tick, other._scale,
        ))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        if isinstance(other, (RealWaveform, WaveVStack)):
            return other + (-self)
        return _c_const(other) + (-self)

    def __mul__(self, other):
        if isinstance(other, ComplexWaveform):
            return other * self
        if isinstance(other, WaveVStack):
            return other * self
        if isinstance(other, RealWaveform):
            if self._is_zero() or other._is_zero():
                return _c_zero()
            return RealWaveform._from_core(
                self._core.mul_affine(
                    other._core, self._delay_tick, 1.0,
                    other._delay_tick, 1.0,
                ),
                scale=self._scale * other._scale,
            )
        real, imag = _number_parts(other)
        if imag:
            return ComplexWaveform(self * real, self * imag)
        if real == 0.0 or self._is_zero():
            return _c_zero()
        return RealWaveform._from_core(
            self._core, self._delay_tick, self._scale * real
        )

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (1 / other)

    def __neg__(self):
        return RealWaveform._from_core(
            self._core, self._delay_tick, -self._scale
        )

    def __rshift__(self, seconds):
        return RealWaveform._from_core(
            self._core,
            self._delay_tick + _time_to_c_tick(seconds),
            self._scale,
        )

    def __lshift__(self, seconds):
        return self >> -seconds

    def __eq__(self, other):
        if isinstance(other, (int, float, np.number)):
            other = _c_const(other)
        if not isinstance(other, RealWaveform):
            return False
        if (self.max, self.min, self.start, self.stop) != (
                other.max, other.min, other.start, other.stop):
            return False
        return self._canonical_core() == other._canonical_core()

    def __hash__(self):
        canonical = self._canonical_core().to_bytes()
        return hash((canonical, self.max, self.min,
                     self.start, self.stop))

    @property
    def marker(self):
        if self.begin == -inf and self.end == inf:
            return one()
        return _piecewise((self.begin, self.end, inf), 0, 1, 0)

    def mask(self, edge=0):
        begin = self.begin
        end = self.end
        if begin != -inf:
            begin += edge
        if end != inf:
            end -= edge
        return _piecewise((begin, end, inf), 0, 1, 0)

    def __or__(self, other):
        if not isinstance(other, RealWaveform):
            other = const(other)
        return (self.marker + other.marker).marker

    def __and__(self, other):
        if not isinstance(other, RealWaveform):
            other = const(other)
        return (self.marker * other.marker).marker

    def __repr__(self):
        return (f"RealWaveform(nodes={self._core.node_count}, "
                f"bytes={len(self.to_bytes())}, begin={self.begin}, "
                f"end={self.end})")

    def __getstate__(self):
        return (self.to_bytes(), self.max, self.min, self.start, self.stop,
                self.sample_rate, self.filters, self.nonlinear, self.label)

    def __setstate__(self, state):
        _ensure_c_clock_locked()
        if len(state) == 8:
            (data, self.max, self.min, self.start, self.stop, self.sample_rate,
             self.filters, self.label) = state
            self.nonlinear = None
        else:
            (data, self.max, self.min, self.start, self.stop, self.sample_rate,
             self.filters, self.nonlinear, self.label) = state
        self._core = _CWaveformCore.from_bytes(data)
        self._delay_tick = 0
        self._scale = 1.0
        self._canonical_core_cache = None


def _c_stack_waves(data):
    """Decode WNS4 event metadata while leaving templates in C blocks."""
    data = bytes(data)
    if len(data) < 16 or data[:8] != b"WNS4\x02\x00\x00\x00":
        raise ValueError("invalid WNS4 stack block")
    template_count, event_count = struct.unpack_from("<II", data, 8)
    cursor = 16 + 4 * template_count
    event_offset = len(data) - 20 * event_count
    templates = []
    for index in range(template_count):
        size = struct.unpack_from("<I", data, 16 + 4 * index)[0]
        templates.append(RealWaveform.from_bytes(data[cursor:cursor + size]))
        cursor += size
    if cursor != event_offset:
        raise ValueError("invalid WNS4 stack layout")
    waves = []
    for index in range(event_count):
        template_id = struct.unpack_from("<I", data, event_offset + 4 * index)[0]
        delay_tick = struct.unpack_from(
            "<q", data, event_offset + 4 * event_count + 8 * index
        )[0]
        scale = struct.unpack_from(
            "<d", data, event_offset + 12 * event_count + 8 * index
        )[0]
        waves.append(RealWaveform._from_core(
            templates[template_id]._core, delay_tick, scale
        ))
    return tuple(waves)


class RealWaveVStack(_RealWaveVStackBase):
    """Thin Python metadata wrapper around a WNS4 C template stack."""

    __slots__ = ("_core", "_shift_tick", "_waves_cache",
                 "_eval_core_cache")

    def __init__(self, waves=(), *, _core=None):
        _ensure_c_clock_locked()
        if _core is None:
            waves = tuple(waves)
            templates = []
            template_map = {}
            ids = []
            delays = []
            scales = []
            event_waves = []
            for wave in waves:
                if not isinstance(wave, RealWaveform):
                    raise TypeError("RealWaveVStack accepts RealWaveform objects")
                if wave._is_zero():
                    continue
                key = wave._core.hash64
                template_id = None
                for candidate in template_map.get(key, ()):
                    if templates[candidate] == wave._core:
                        template_id = candidate
                        break
                if template_id is None:
                    template_id = len(templates)
                    template_map.setdefault(key, []).append(template_id)
                    templates.append(wave._core)
                ids.append(template_id)
                delays.append(wave._delay_tick)
                scales.append(wave._scale)
                event_waves.append(wave)
            _core = _CWaveformStackCore.from_events(templates, ids, delays, scales)
            waves = tuple(event_waves)
        self._core = _core
        self.min = -inf
        self.max = inf
        self.start = None
        self.stop = None
        self.sample_rate = None
        self.offset = 0.0
        self._shift_tick = 0
        self.filters = None
        self.nonlinear = None
        self.label = None
        self.function_lib = None
        self._sample_plan_cache = None
        self._waves_cache = tuple(waves) if waves else None
        self._eval_core_cache = None

    @classmethod
    def from_events(cls, templates, template_ids, delay_ticks, scales):
        _ensure_c_clock_locked()
        templates = tuple(templates)
        if not all(isinstance(template, RealWaveform)
                   for template in templates):
            raise TypeError("templates must contain RealWaveform objects")
        cores = [template._materialized_core() for template in templates]
        core = _CWaveformStackCore.from_events(
            cores, template_ids, delay_ticks, scales
        )
        return cls(_core=core)

    @property
    def shift(self):
        return _c_tick_to_time(self._shift_tick)

    @shift.setter
    def shift(self, value):
        self._shift_tick = _time_to_c_tick(value)

    @property
    def wlist(self):
        if self._waves_cache is None:
            self._waves_cache = _c_stack_waves(self._core.to_bytes())
        return list(self._waves_cache)

    @property
    def begin(self):
        tick = self._core.lower_tick
        value = (-inf if tick == np.iinfo(np.int64).min
                 else _c_tick_to_time(tick + self._shift_tick))
        return value if self.start is None else max(self.start, value)

    @property
    def end(self):
        tick = self._core.upper_tick
        value = (inf if tick == np.iinfo(np.int64).max
                 else _c_tick_to_time(tick + self._shift_tick))
        return value if self.stop is None else min(self.stop, value)

    @property
    def event_count(self):
        return self._core.event_count

    @property
    def template_count(self):
        return self._core.template_count

    def __len__(self):
        return self._core.event_count

    def __call__(self, x, out=None, accumulate=False, *, _clip=True, **kwargs):
        scalar = isinstance(x, (int, float, np.number))
        raw_positions = np.asarray([x] if scalar else x)
        if np.iscomplexobj(raw_positions):
            raise TypeError("waveform positions must be real")
        positions = np.asarray(raw_positions, dtype=np.float64)
        plan = _array_tick_plan(positions)
        if plan is not None:
            cache_key = (plan[0], plan[1], plan[2], 1, self._shift_tick)
            if (self._sample_plan_cache is not None
                    and self._sample_plan_cache[0] == cache_key):
                sample_plan = self._sample_plan_cache[1]
            else:
                sample_plan = self._core.prepare_sample(
                    plan[0], plan[1], plan[2], 1, self._shift_tick
                )
                self._sample_plan_cache = cache_key, sample_plan
            if sample_plan is None:
                values = self._core.sample(
                    plan[0], plan[1], plan[2], 1,
                    self._shift_tick, self.offset,
                )
            else:
                values = sample_plan.sample(self.offset)
        else:
            if self._eval_core_cache is None:
                self._eval_core_cache = self._core.simplify(
                    self._shift_tick, self.offset
                )
            values = self._eval_core_cache.evaluate(positions)
        if _clip:
            values = _clip_samples(values, *_amplitude_limits(self))
        if out is not None:
            if accumulate:
                out[...] += values
            else:
                out[...] = values
            values = out
        return values[0] if scalar else values

    def sample(self, sample_rate=None, out=None, chunk_size=None,
               filters=None, dtype=None, full_scale=1.0, nonlinear=None,
               **kwargs):
        minimum, maximum = _amplitude_limits(self)
        if chunk_size is not None:
            return Waveform.sample(
                self, sample_rate, out, chunk_size, filters=filters,
                dtype=dtype, full_scale=full_scale, nonlinear=nonlinear,
            )
        if sample_rate is None:
            sample_rate = self.sample_rate
        if self.start is None or self.stop is None or sample_rate is None:
            raise ValueError("RealWaveVStack sampling metadata is incomplete")
        if filters is None:
            filters = self.filters
        if nonlinear is None:
            nonlinear = self.nonlinear
        plan = _sampling_plan(self.start, self.stop, sample_rate)
        dtype, bits = _quantization_bits(dtype, out)
        if plan is None or get_time_resolution() != 1 / _CORE_TICKS_PER_SECOND:
            positions = np.arange(self.start, self.stop, 1 / float(sample_rate))
            values = self._evaluate_raw(positions)
        else:
            start_tick, count, step_numerator, step_denominator = plan
            cache_key = (
                start_tick, count, step_numerator, step_denominator,
                self._shift_tick,
            )
            if (self._sample_plan_cache is not None
                    and self._sample_plan_cache[0] == cache_key):
                sample_plan = self._sample_plan_cache[1]
            else:
                sample_plan = self._core.prepare_sample(
                    start_tick, count, step_numerator, step_denominator,
                    self._shift_tick,
                )
                self._sample_plan_cache = cache_key, sample_plan
            if sample_plan is not None:
                if filters is None and nonlinear is None:
                    values = sample_plan.sample(
                        self.offset, bits or 0, full_scale,
                        out if bits is not None else None, minimum, maximum,
                    )
                    return (values if bits is not None else
                            _finish_samples(values, dtype, full_scale, out))
                values = sample_plan.sample(self.offset)
            else:
                if (bits is not None and filters is None
                        and nonlinear is None):
                    values = self._core.sample(
                        start_tick, count, step_numerator, step_denominator,
                        self._shift_tick, self.offset, bits, full_scale, out,
                    )
                    return _clip_quantized(
                        values, bits, full_scale, minimum, maximum)
                values = self._core.sample(
                    start_tick, count, step_numerator, step_denominator,
                    self._shift_tick, self.offset,
                )
        values = _apply_nonlinear(values, nonlinear)
        result, _ = _filter_and_finish(
            values, filters, dtype, full_scale, out, minimum, maximum)
        return result

    def simplify(self, eps=1e-15):
        return RealWaveform._from_core(
            self._core.simplify(self._shift_tick, self.offset).simplify(eps)
        )

    def to_bytes(self):
        if self._shift_tick == 0 and self.offset == 0:
            return self._core.to_bytes()
        return self._core.materialize(
            self._shift_tick, self.offset).to_bytes()

    @classmethod
    def from_bytes(cls, data):
        return cls(_core=_CWaveformStackCore.from_bytes(data))

    def __rshift__(self, seconds):
        result = RealWaveVStack(_core=self._core)
        result._shift_tick = self._shift_tick + _time_to_c_tick(seconds)
        result.offset = self.offset
        _copy_sampling_metadata(self, result)
        return result

    def __lshift__(self, seconds):
        return self >> -seconds

    def __add__(self, other):
        if isinstance(other, ComplexWaveVStack):
            return other + self
        if isinstance(other, ComplexWaveform):
            return ComplexWaveVStack(self) + other
        if isinstance(other, RealWaveVStack):
            result = RealWaveVStack(_core=self._core.combine(
                other._core, self._shift_tick, other._shift_tick
            ))
            result.offset = self.offset + other.offset
            _copy_sampling_metadata(self, result)
            return result
        if isinstance(other, RealWaveform):
            result = RealWaveVStack(_core=self._core.append(
                other._core, self._shift_tick,
                other._delay_tick, other._scale,
            ))
            result.offset = self.offset
            _copy_sampling_metadata(self, result)
            return result
        if isinstance(other, (int, float, complex, np.number)):
            real, imag = _number_parts(other)
            if imag:
                return ComplexWaveVStack(self + real,
                                         WaveVStack() + imag)
            result = RealWaveVStack(_core=self._core)
            result.offset = self.offset + real
            result._shift_tick = self._shift_tick
            _copy_sampling_metadata(self, result)
            return result
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, (ComplexWaveform, ComplexWaveVStack)):
            return ComplexWaveVStack(self) * other
        if isinstance(other, RealWaveform):
            waves = [(wave >> self.shift) * other for wave in self.wlist]
            if self.offset:
                waves.append(self.offset * other)
            return WaveVStack(waves)
        real, imag = _number_parts(other)
        if imag:
            return ComplexWaveVStack(self * real, self * imag)
        result = RealWaveVStack(_core=self._core.scaled(real))
        result.offset = self.offset * real
        result._shift_tick = self._shift_tick
        _copy_sampling_metadata(self, result)
        return result

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Waveform):
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
        return self.marker.mask(edge)

    def __or__(self, other):
        return self.simplify() | other

    def __and__(self, other):
        return self.simplify() & other

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, RealWaveVStack):
            if (self._shift_tick == other._shift_tick
                    and self.offset == other.offset
                    and self._core.to_bytes() == other._core.to_bytes()
                    and (self.start, self.stop) == (other.start, other.stop)):
                return True
            return self.simplify() == other.simplify()
        if isinstance(other, WaveVStack):
            return self.simplify() == other.simplify()
        return self.simplify() == other

    __hash__ = None

    def __repr__(self):
        return (f"RealWaveVStack(events={self._core.event_count}, "
                f"templates={self._core.template_count}, "
                f"bytes={len(self.to_bytes())})")

    def _repr_latex_(self):
        return rf"\sum_{{i=1}}^{{{self._core.event_count}}}f_i(t)"

    def __getstate__(self):
        return (self._core.to_bytes(), self.start, self.stop,
                self.sample_rate, self.offset, self._shift_tick,
                self.filters, self.nonlinear, self.label, self.min, self.max)

    def __setstate__(self, state):
        _ensure_c_clock_locked()
        self.min, self.max = state[9:] if len(state) == 11 else (-inf, inf)
        state = state[:9]
        if len(state) == 8:
            (data, self.start, self.stop, self.sample_rate, self.offset,
             self._shift_tick, self.filters, self.label) = state
            self.nonlinear = None
        else:
            (data, self.start, self.stop, self.sample_rate, self.offset,
             self._shift_tick, self.filters, self.nonlinear, self.label) = state
        self._core = _CWaveformStackCore.from_bytes(data)
        self.function_lib = None
        self._sample_plan_cache = None
        self._waves_cache = None
        self._eval_core_cache = None


def _make_real_stack(wlist=()):
    """Build the single native real-stack representation."""
    if isinstance(wlist, ComplexWaveVStack):
        return ComplexWaveVStack(wlist)
    if isinstance(wlist, RealWaveVStack):
        return RealWaveVStack(_core=wlist._core)
    if isinstance(wlist, RealWaveVStack):
        return RealWaveVStack(wlist.wlist)
    waves = tuple(wlist)
    if any(isinstance(wave, ComplexWaveform) for wave in waves):
        raise TypeError("WaveVStack is real-only; use ComplexWaveVStack")
    if not all(isinstance(wave, RealWaveform) for wave in waves):
        raise TypeError("WaveVStack accepts RealWaveform objects")
    if not all(isinstance(wave, RealWaveform) for wave in waves):
        raise TypeError("all real waveforms must use the C core")
    return RealWaveVStack(waves)


def _c_zero():
    return RealWaveform._from_core(_CWaveformCore.constant(0.0))


def _c_one():
    return RealWaveform._from_core(_CWaveformCore.constant(1.0))


def _c_const(value):
    real, imag = _number_parts(value)
    if imag:
        return ComplexWaveform(_c_const(real), _c_const(imag))
    return RealWaveform._from_core(_CWaveformCore.constant(real))


def _c_gaussian(width):
    return RealWaveform._from_core(_CWaveformCore.gaussian(width))


def _c_cos(w, phi=0.0):
    return RealWaveform._from_core(_CWaveformCore.cos(w, phi))


def _c_sin(w, phi=0.0):
    return RealWaveform._from_core(_CWaveformCore.sin(w, phi))


def _c_square(width):
    return RealWaveform._from_core(_CWaveformCore.square(width))


def _real_const(value):
    real, imag = _number_parts(value)
    if imag != 0:
        raise TypeError("real waveform constant cannot have an imaginary part")
    return _c_const(real)


def zero():
    return _real_const(0)


def one():
    return _real_const(1.0)


def const(value):
    real, imag = _number_parts(value)
    if imag != 0:
        return ComplexWaveform(_real_const(real), _real_const(imag))
    return _real_const(real)


def D(wav: Waveform, d: int = 1):
    if isinstance(wav, ComplexWaveform):
        return ComplexWaveform(D(wav.real, d), D(wav.imag, d))
    if not isinstance(wav, RealWaveform):
        raise TypeError("D expects a Waveform or ComplexWaveform")
    if d < 0 or not isinstance(d, int):
        raise ValueError("d must be a non-negative integer")
    return RealWaveform._from_core(
        wav._materialized_core().derivative(d)
    )


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
    if edge == 0 and _C_CLOCK_ACTIVE:
        return _c_square(width)
    if edge == 0:
        return _piecewise((-width / 2, width / 2, inf), 0, 1, 0)
    return (step(edge, type=type) << width / 2) - (step(edge, type=type) >> width / 2)


def gaussian(width, plateau=0.0, d=None):
    if width <= 0 and plateau <= 0:
        return zero()
    if d is None and plateau == 0 and _C_CLOCK_ACTIVE:
        return _c_gaussian(width)
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
    if _C_CLOCK_ACTIVE:
        return _c_cos(w, phi)
    return RealWaveform._from_core(_scalar(COS, w, shift=-phi / w))


def sin(w, phi=0):
    if w == 0:
        return const(np.sin(phi))
    if w < 0:
        phi = -phi + pi
        w = -w
    if _C_CLOCK_ACTIVE:
        return _c_sin(w, phi)
    return RealWaveform._from_core(
        _scalar(COS, w, shift=(pi / 2 - phi) / w)
    )


def exp(alpha):
    if np.iscomplexobj(alpha):
        alpha = complex(alpha)
        carrier = cos(alpha.imag) + 1j * sin(alpha.imag)
        return carrier if alpha.real == 0 else exp(alpha.real) * carrier
    return RealWaveform._from_core(_scalar(EXP, alpha))


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
    return RealWaveform._from_core(_scalar(COSH, w))


def sinh(w):
    return RealWaveform._from_core(_scalar(SINH, w))


@lru_cache(maxsize=1024)
def _cosh_pulse_core(width, eps, plateau):
    w = eps / width
    amplitude = np.cosh(eps / 2)
    scale = -1 / (amplitude - 1)

    def edge(shift):
        return _affine_expression(
            COSH, w, shift=shift, scale=scale,
            offset=amplitude / (amplitude - 1),
        )

    if (plateau == 0
            or quantize_time(-plateau / 2) == quantize_time(plateau / 2)):
        waveform = _piecewise(
            (-width / 2, width / 2, inf), 0, edge(0), 0
        )
    else:
        waveform = _piecewise(
            (-width / 2 - plateau / 2, -plateau / 2, plateau / 2,
             width / 2 + plateau / 2, inf),
            0, edge(-plateau / 2), 1, edge(plateau / 2), 0,
        )
    return waveform._materialized_core()


def coshPulse(width, eps=1.0, plateau=0.0):
    if width <= 0 and plateau <= 0:
        return zero()
    return RealWaveform._from_core(
        _cosh_pulse_core(float(width), float(eps), float(plateau))
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
    return RealWaveform._from_core(_scalar(LINEAR))


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
    import pyaudio  # pyright: ignore[reportMissingModuleSource]

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
    """Parse an expression through the unified waveform constructors."""
    import sys

    from .waveform_parser import WaveformParseError, parse_waveform_expression

    try:
        return parse_waveform_expression(expr, backend=sys.modules[__name__],
                                         extra_modules=())
    except WaveformParseError as exc:
        raise SyntaxError(f"Failed to parse expression {expr!r}: {exc}") from exc


__all__ = [
    "D", "ComplexWaveform", "ComplexWaveVStack", "NonlinearMap", "RealWaveform",
    "RealWaveVStack", "Waveform", "WaveVStack",
    "chirp", "const", "cos", "cosh",
    "coshPulse", "cosPulse", "cut", "drag", "drag_sin", "drag_sinx",
    "exp", "function",
    "gaussian", "general_cosine", "get_time_resolution", "hanning",
    "interp", "mixing", "mollifier", "one", "play", "poly",
    "registerBaseFunc", "registerDerivative", "samplingPoints",
    "sample_clock", "sample_grid", "quantize_time", "time_to_tick",
    "tick_to_time", "set_time_resolution", "sign", "sin", "sinc",
    "sinh", "slepian",
    "square", "step", "t", "wave_eval", "zero", "e", "inf", "pi",
]
