# cython: language_level=3
"""Cython ownership layer for the language-neutral C waveform core."""

cimport cython
from fractions import Fraction

from libc.math cimport isfinite as c_isfinite, round as c_round
from libc.stdint cimport (
    int16_t, int32_t, int64_t, uint8_t, uint32_t, uint64_t,
)
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize

import numpy as np
from numpy import inf

# Stable built-in opcodes.  They intentionally match the original module.
LINEAR = 1
GAUSSIAN = 2
ERF = 3
COS = 4
SINC = 5
EXP = 6
INTERP = 7
LINEARCHIRP = 8
EXPONENTIALCHIRP = 9
HYPERBOLICCHIRP = 10
COSH = 11
SINH = 12
DRAG = 13
MOLLIFIER = 14
D_GAUSSIAN = 15
DRAG_SIN = 16
DRAG_SINX = 17

# The process-wide time quantum is exact even though the public API exposes
# seconds as floats. A 120 GHz clock covers common instrument rates with
# integer sample steps while ``sample_clock`` retains a rational fallback.
_TIME_NUMERATOR = 1
_TIME_DENOMINATOR = 120_000_000_000
_TIME_RESOLUTION = _TIME_NUMERATOR / _TIME_DENOMINATOR
_TIME_LOCKED = False
_INF_TICK = np.iinfo(np.int64).max


def set_time_resolution(value):
    """Set the process-wide tick duration before creating any waveform block."""
    global _TIME_NUMERATOR, _TIME_DENOMINATOR, _TIME_RESOLUTION, _TIME_LOCKED
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0:
        raise ValueError("time resolution must be a finite positive number")
    # Preserve the exact built-in fraction when callers repeat the public
    # float value (notably ``set_time_resolution(get_time_resolution())``).
    if numeric == _TIME_RESOLUTION:
        return
    fraction = value if isinstance(value, Fraction) else Fraction(str(value))
    if (_TIME_LOCKED
            and (fraction.numerator != _TIME_NUMERATOR
                 or fraction.denominator != _TIME_DENOMINATOR)
            and numeric != _TIME_RESOLUTION):
        raise RuntimeError("time resolution is locked by existing waveform objects")
    _TIME_NUMERATOR = fraction.numerator
    _TIME_DENOMINATOR = fraction.denominator
    _TIME_RESOLUTION = numeric


def get_time_resolution():
    return _TIME_RESOLUTION


def quantize_time(value):
    if value == inf:
        return inf
    if value == -inf:
        return -inf
    return _tick_to_time(_time_to_tick(value))


def time_to_tick(value):
    """Return the nearest process-wide integer tick for *value* seconds."""
    return _time_to_tick(value)


def tick_to_time(tick):
    """Convert an integer process-wide tick to seconds."""
    return _tick_to_time(tick)


def sample_clock(sample_rate):
    """Return the reduced rational number of ticks per sample."""
    rate = (sample_rate if isinstance(sample_rate, Fraction)
            else Fraction(str(sample_rate)))
    if rate <= 0:
        raise ValueError("sample_rate must be positive")
    step = Fraction(_TIME_DENOMINATOR, _TIME_NUMERATOR) / rate
    return step.numerator, step.denominator


@cython.boundscheck(False)
@cython.wraparound(False)
def sample_grid(long long start_tick, Py_ssize_t count,
                long long step_numerator, long long step_denominator=1,
                long long index_offset=0):
    """Generate an exact-clock sample grid, converting to float only at output."""
    cdef object result
    cdef double[::1] view
    cdef Py_ssize_t index
    cdef double clock_denominator
    cdef double time_numerator = _TIME_NUMERATOR
    if count < 0:
        raise ValueError("count must be non-negative")
    if step_numerator <= 0 or step_denominator <= 0:
        raise ValueError("sample step must be positive")
    result = np.empty(count, dtype=np.float64)
    view = result
    clock_denominator = (<double>step_denominator
                         * <double>_TIME_DENOMINATOR)
    with nogil:
        for index in range(count):
            view[index] = (
                ((<double>start_tick * <double>step_denominator
                  + (<double>index + <double>index_offset)
                  * <double>step_numerator)
                 * time_numerator)
                / clock_denominator
            )
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def quantize_samples(values, bits, full_scale=1.0, out=None):
    """Saturating real-signal quantizer for signed 16- and 32-bit DAC data."""
    cdef object source
    cdef object target
    cdef Py_ssize_t size
    cdef double full_scale_value
    cdef const double *source_pointer
    cdef void *target_pointer
    cdef int status
    cdef int bit_count

    full_scale_value = float(full_scale)
    if not np.isfinite(full_scale_value) or full_scale_value <= 0:
        raise ValueError("full_scale must be a finite positive number")
    if bits not in (16, 32):
        raise ValueError("bits must be 16 or 32")
    bit_count = bits
    if np.iscomplexobj(values):
        raise TypeError("integer quantization requires a real signal")
    source = np.ascontiguousarray(values, dtype=np.float64)
    dtype = np.dtype(np.int16 if bits == 16 else np.int32)
    if out is None:
        target = np.empty(source.shape, dtype=dtype)
    else:
        target = np.asarray(out)
        if target.shape != source.shape:
            raise ValueError("out has the wrong shape")
        if target.dtype != dtype:
            raise TypeError(f"out must have dtype {dtype}")
        if not target.flags.c_contiguous or not target.flags.writeable:
            raise ValueError("out must be a writable C-contiguous array")

    size = source.size
    if size == 0:
        return target
    source_pointer = <const double *><size_t>source.ctypes.data
    target_pointer = <void *><size_t>target.ctypes.data
    with nogil:
        status = cwaveform_quantize(
            source_pointer, size, bit_count, full_scale_value, target_pointer,
        )
    if status == -3:
        raise ValueError("cannot quantize non-finite samples")
    if status == -2:
        raise MemoryError("C quantization allocation failed")
    if status != 0:
        raise RuntimeError(f"C quantization failed with status {status}")
    return target


@cython.boundscheck(False)
@cython.wraparound(False)
def place_template_quantized(out, template, destinations, scales,
                             offset=0.0, full_scale=1.0):
    """Write non-overlapping scaled template slices into an integer buffer."""
    cdef object output = np.asarray(out)
    cdef object values = np.ascontiguousarray(template, dtype=np.float64)
    cdef object starts = np.ascontiguousarray(destinations, dtype=np.int64)
    cdef object factors = np.ascontiguousarray(scales, dtype=np.float64)
    cdef double[::1] values_view
    cdef int64_t[::1] starts_view
    cdef double[::1] factors_view
    cdef int16_t[::1] target16
    cdef int32_t[::1] target32
    cdef Py_ssize_t event_index, value_index, source, destination, count
    cdef Py_ssize_t output_size, template_size, event_count
    cdef double factor, value, scaled
    cdef double full_scale_value = float(full_scale)
    cdef double offset_value = float(offset)
    cdef double scale
    cdef double maximum
    cdef double minimum
    cdef bint invalid = False

    if not c_isfinite(full_scale_value) or full_scale_value <= 0:
        raise ValueError("full_scale must be a finite positive number")
    if not c_isfinite(offset_value):
        raise ValueError("cannot quantize non-finite samples")
    if output.ndim != 1:
        raise TypeError("out must be a one-dimensional int16 or int32 array")
    if output.dtype not in (np.dtype(np.int16), np.dtype(np.int32)):
        raise TypeError("out must have dtype int16 or int32")
    if not output.flags.c_contiguous or not output.flags.writeable:
        raise ValueError("out must be a writable C-contiguous array")
    if starts.ndim != 1 or factors.ndim != 1 or len(starts) != len(factors):
        raise ValueError("destinations and scales must be equal-length vectors")

    values_view = values.reshape(-1)
    starts_view = starts
    factors_view = factors
    output_size = output.shape[0]
    template_size = values_view.shape[0]
    event_count = starts_view.shape[0]
    if output.dtype == np.dtype(np.int16):
        target16 = output
        scale = 32768.0 / full_scale_value
        minimum = -32768.0
        maximum = 32767.0
        with nogil:
            for event_index in range(event_count):
                destination = starts_view[event_index]
                source = 0
                if destination < 0:
                    source = -destination
                    destination = 0
                count = template_size - source
                if count > output_size - destination:
                    count = output_size - destination
                factor = factors_view[event_index]
                if not c_isfinite(factor):
                    invalid = True
                    continue
                for value_index in range(count if count > 0 else 0):
                    value = (offset_value
                             + factor * values_view[source + value_index])
                    if not c_isfinite(value):
                        invalid = True
                        target16[destination + value_index] = 0
                    elif value <= -full_scale_value:
                        target16[destination + value_index] = <int16_t>-32768
                    elif value >= full_scale_value:
                        target16[destination + value_index] = <int16_t>32767
                    else:
                        scaled = c_round(value * scale)
                        if scaled < minimum:
                            scaled = minimum
                        elif scaled > maximum:
                            scaled = maximum
                        target16[destination + value_index] = <int16_t>scaled
    else:
        target32 = output
        scale = 2147483648.0 / full_scale_value
        minimum = -2147483648.0
        maximum = 2147483647.0
        with nogil:
            for event_index in range(event_count):
                destination = starts_view[event_index]
                source = 0
                if destination < 0:
                    source = -destination
                    destination = 0
                count = template_size - source
                if count > output_size - destination:
                    count = output_size - destination
                factor = factors_view[event_index]
                if not c_isfinite(factor):
                    invalid = True
                    continue
                for value_index in range(count if count > 0 else 0):
                    value = (offset_value
                             + factor * values_view[source + value_index])
                    if not c_isfinite(value):
                        invalid = True
                        target32[destination + value_index] = 0
                    elif value <= -full_scale_value:
                        target32[destination + value_index] = <int32_t>-2147483648
                    elif value >= full_scale_value:
                        target32[destination + value_index] = <int32_t>2147483647
                    else:
                        scaled = c_round(value * scale)
                        if scaled < minimum:
                            scaled = minimum
                        elif scaled > maximum:
                            scaled = maximum
                        target32[destination + value_index] = <int32_t>scaled
    if invalid:
        raise ValueError("cannot quantize non-finite samples")
    return output


@cython.boundscheck(False)
@cython.wraparound(False)
def place_quantized_template(out, template, destinations):
    """Copy one quantized template to multiple non-overlapping positions."""
    cdef object output = np.asarray(out)
    cdef object values
    cdef object starts = np.ascontiguousarray(destinations, dtype=np.int64)
    cdef int64_t[::1] starts_view
    cdef int16_t[::1] target16
    cdef int16_t[::1] values16
    cdef int32_t[::1] target32
    cdef int32_t[::1] values32
    cdef Py_ssize_t event_index, source, destination, count
    cdef Py_ssize_t output_size, template_size, event_count

    if output.ndim != 1 or output.dtype not in (
            np.dtype(np.int16), np.dtype(np.int32)):
        raise TypeError("out must be a one-dimensional int16 or int32 array")
    if not output.flags.c_contiguous or not output.flags.writeable:
        raise ValueError("out must be a writable C-contiguous array")
    values = np.ascontiguousarray(template, dtype=output.dtype).reshape(-1)
    if starts.ndim != 1:
        raise ValueError("destinations must be one-dimensional")
    starts_view = starts
    output_size = output.shape[0]
    template_size = values.shape[0]
    event_count = starts.shape[0]
    if output.dtype == np.dtype(np.int16):
        target16 = output
        values16 = values
        with nogil:
            for event_index in range(event_count):
                destination = starts_view[event_index]
                source = 0
                if destination < 0:
                    source = -destination
                    destination = 0
                count = template_size - source
                if count > output_size - destination:
                    count = output_size - destination
                if count > 0:
                    memcpy(&target16[destination], &values16[source],
                           count * sizeof(int16_t))
    else:
        target32 = output
        values32 = values
        with nogil:
            for event_index in range(event_count):
                destination = starts_view[event_index]
                source = 0
                if destination < 0:
                    source = -destination
                    destination = 0
                count = template_size - source
                if count > output_size - destination:
                    count = output_size - destination
                if count > 0:
                    memcpy(&target32[destination], &values32[source],
                           count * sizeof(int32_t))
    return output


@cython.boundscheck(False)
@cython.wraparound(False)
def accumulate_template(out, template, destinations, scales):
    """Accumulate repeated, scaled template slices into a float64 buffer."""
    cdef object output = np.asarray(out)
    cdef object values = np.ascontiguousarray(template, dtype=np.float64)
    cdef object starts = np.ascontiguousarray(destinations, dtype=np.int64)
    cdef object factors = np.ascontiguousarray(scales, dtype=np.float64)
    cdef double[::1] output_view
    cdef double[::1] values_view
    cdef int64_t[::1] starts_view
    cdef double[::1] factors_view
    cdef Py_ssize_t event_index, value_index, source, destination, count
    cdef Py_ssize_t output_size, template_size, event_count
    cdef double factor
    if output.dtype != np.dtype(np.float64) or output.ndim != 1:
        raise TypeError("out must be a one-dimensional float64 array")
    if not output.flags.c_contiguous or not output.flags.writeable:
        raise ValueError("out must be a writable C-contiguous array")
    if starts.ndim != 1 or factors.ndim != 1 or len(starts) != len(factors):
        raise ValueError("destinations and scales must be equal-length vectors")
    output_view = output
    values_view = values.reshape(-1)
    starts_view = starts
    factors_view = factors
    output_size = output_view.shape[0]
    template_size = values_view.shape[0]
    event_count = starts_view.shape[0]
    with nogil:
        for event_index in range(event_count):
            destination = starts_view[event_index]
            source = 0
            if destination < 0:
                source = -destination
                destination = 0
            count = template_size - source
            if count > output_size - destination:
                count = output_size - destination
            if count > 0:
                factor = factors_view[event_index]
                for value_index in range(count):
                    output_view[destination + value_index] += (
                        factor * values_view[source + value_index]
                    )
    return output


def _lock_time_resolution():
    global _TIME_LOCKED
    _TIME_LOCKED = True


def lock_time_resolution():
    """Lock the process clock for waveform block owners."""
    _lock_time_resolution()


def _time_to_tick(value):
    if value == inf:
        return int(_INF_TICK)
    if value == -inf:
        return -int(_INF_TICK)
    tick = int(round(float(value) * _TIME_DENOMINATOR / _TIME_NUMERATOR))
    if tick <= -int(_INF_TICK) or tick >= int(_INF_TICK):
        raise OverflowError("time is outside the waveform tick range")
    return tick


def _tick_to_time(tick):
    if tick == int(_INF_TICK):
        return inf
    if tick == -int(_INF_TICK):
        return -inf
    return int(tick) * _TIME_NUMERATOR / _TIME_DENOMINATOR


def registerBaseFunc(*args, **kwargs):
    raise NotImplementedError("custom waveform base functions are not supported")


def registerDerivative(*args, **kwargs):
    raise NotImplementedError("custom waveform derivatives are not supported")


cdef extern from "_cwaveform.h":
    ctypedef struct cwaveform_wave:
        pass
    ctypedef struct cwaveform_stack:
        pass
    ctypedef struct cwaveform_sample_plan:
        pass

    uint64_t cwaveform_ticks_per_second() noexcept nogil
    int cwaveform_set_ticks_per_second(uint64_t) noexcept nogil
    cwaveform_wave *cwaveform_wave_constant(double) noexcept nogil
    cwaveform_wave *cwaveform_wave_gaussian(double) noexcept nogil
    cwaveform_wave *cwaveform_wave_cos(double, double) noexcept nogil
    cwaveform_wave *cwaveform_wave_sin(double, double) noexcept nogil
    cwaveform_wave *cwaveform_wave_square(double) noexcept nogil
    cwaveform_wave *cwaveform_wave_builtin(
        int, const double *, size_t, int64_t) noexcept nogil
    cwaveform_wave *cwaveform_wave_window(
        const cwaveform_wave *, int64_t, int64_t) noexcept nogil
    cwaveform_wave *cwaveform_wave_power(
        const cwaveform_wave *, int) noexcept nogil
    cwaveform_wave *cwaveform_wave_derivative(
        const cwaveform_wave *, unsigned int) noexcept nogil
    cwaveform_wave *cwaveform_wave_filter(
        const cwaveform_wave *, double, double, double) noexcept nogil
    cwaveform_wave *cwaveform_wave_simplify(
        const cwaveform_wave *, double) noexcept nogil
    cwaveform_wave *cwaveform_wave_from_bytes(const uint8_t *, size_t) noexcept nogil
    cwaveform_wave *cwaveform_wave_add_affine(
        const cwaveform_wave *, int64_t, double,
        const cwaveform_wave *, int64_t, double) noexcept nogil
    cwaveform_wave *cwaveform_wave_mul_affine(
        const cwaveform_wave *, int64_t, double,
        const cwaveform_wave *, int64_t, double) noexcept nogil
    cwaveform_wave *cwaveform_wave_materialize(
        const cwaveform_wave *, int64_t, double) noexcept nogil
    void cwaveform_wave_retain(cwaveform_wave *) noexcept nogil
    void cwaveform_wave_release(cwaveform_wave *) noexcept nogil
    const uint8_t *cwaveform_wave_bytes(const cwaveform_wave *, size_t *) noexcept nogil
    uint64_t cwaveform_wave_hash(const cwaveform_wave *) noexcept nogil
    int cwaveform_wave_equal(const cwaveform_wave *, const cwaveform_wave *) noexcept nogil
    int64_t cwaveform_wave_lower_tick(const cwaveform_wave *) noexcept nogil
    int64_t cwaveform_wave_upper_tick(const cwaveform_wave *) noexcept nogil
    uint32_t cwaveform_wave_node_count(const cwaveform_wave *) noexcept nogil
    int cwaveform_wave_evaluate(
        const cwaveform_wave *, const double *, size_t, int64_t, double,
        double, double, double *) noexcept nogil
    int cwaveform_wave_sample(
        const cwaveform_wave *, int64_t, size_t, int64_t, int64_t, int64_t,
        double, double, double, int, double, void *) noexcept nogil
    int cwaveform_quantize(
        const double *, size_t, int, double, void *) noexcept nogil

    cwaveform_stack *cwaveform_stack_create(
        cwaveform_wave *const *, const uint32_t *, const int64_t *,
        const double *, size_t, size_t) noexcept nogil
    cwaveform_stack *cwaveform_stack_from_bytes(
        const uint8_t *, size_t) noexcept nogil
    cwaveform_stack *cwaveform_stack_materialize(
        const cwaveform_stack *, int64_t, double) noexcept nogil
    cwaveform_stack *cwaveform_stack_combine(
        const cwaveform_stack *, int64_t,
        const cwaveform_stack *, int64_t) noexcept nogil
    cwaveform_stack *cwaveform_stack_append(
        const cwaveform_stack *, int64_t,
        const cwaveform_wave *, int64_t, double) noexcept nogil
    cwaveform_stack *cwaveform_stack_scale(
        const cwaveform_stack *, double) noexcept nogil
    void cwaveform_stack_retain(cwaveform_stack *) noexcept nogil
    void cwaveform_stack_release(cwaveform_stack *) noexcept nogil
    const uint8_t *cwaveform_stack_bytes(
        const cwaveform_stack *, size_t *) noexcept nogil
    uint64_t cwaveform_stack_hash(const cwaveform_stack *) noexcept nogil
    size_t cwaveform_stack_event_count(const cwaveform_stack *) noexcept nogil
    size_t cwaveform_stack_template_count(const cwaveform_stack *) noexcept nogil
    int cwaveform_stack_evaluate(
        const cwaveform_stack *, const double *, size_t, int64_t, double,
        double *) noexcept nogil
    int cwaveform_stack_sample(
        const cwaveform_stack *, int64_t, size_t, int64_t, int64_t, int64_t,
        double, int, double, void *) noexcept nogil
    cwaveform_sample_plan *cwaveform_sample_plan_create(
        const cwaveform_stack *, int64_t, size_t, int64_t, int64_t,
        int64_t) noexcept nogil
    void cwaveform_sample_plan_retain(cwaveform_sample_plan *) noexcept nogil
    void cwaveform_sample_plan_release(cwaveform_sample_plan *) noexcept nogil
    size_t cwaveform_sample_plan_count(
        const cwaveform_sample_plan *) noexcept nogil
    size_t cwaveform_sample_plan_group_count(
        const cwaveform_sample_plan *) noexcept nogil
    int cwaveform_sample_plan_non_overlapping(
        const cwaveform_sample_plan *) noexcept nogil
    int cwaveform_sample_plan_sample(
        const cwaveform_sample_plan *, double, int, double,
        void *) noexcept nogil
    cwaveform_wave *cwaveform_stack_simplify(
        const cwaveform_stack *, int64_t, double) noexcept nogil
    const char *cwaveform_format_description() noexcept nogil


cdef CWaveformCore _wrap_wave(cwaveform_wave *pointer):
    if pointer == NULL:
        raise MemoryError("C waveform construction failed")
    cdef CWaveformCore result = CWaveformCore.__new__(CWaveformCore)
    result._pointer = pointer
    result._serialized = None
    return result


cdef CWaveformStackCore _wrap_stack(cwaveform_stack *pointer):
    if pointer == NULL:
        raise MemoryError("C stack construction failed")
    cdef CWaveformStackCore result = CWaveformStackCore.__new__(CWaveformStackCore)
    result._pointer = pointer
    result._serialized = None
    return result


cdef CWaveformSamplePlan _wrap_plan(cwaveform_sample_plan *pointer):
    if pointer == NULL:
        return None
    cdef CWaveformSamplePlan result = CWaveformSamplePlan.__new__(CWaveformSamplePlan)
    result._pointer = pointer
    return result


cdef class CWaveformCore:
    cdef cwaveform_wave *_pointer
    cdef object _serialized

    def __cinit__(self):
        self._pointer = NULL
        self._serialized = None

    def __dealloc__(self):
        if self._pointer != NULL:
            cwaveform_wave_release(self._pointer)

    @staticmethod
    def constant(double value):
        return _wrap_wave(cwaveform_wave_constant(value))

    @staticmethod
    def gaussian(double width):
        return _wrap_wave(cwaveform_wave_gaussian(width))

    @staticmethod
    def cos(double angular_frequency, double phase=0.0):
        return _wrap_wave(cwaveform_wave_cos(angular_frequency, phase))

    @staticmethod
    def sin(double angular_frequency, double phase=0.0):
        return _wrap_wave(cwaveform_wave_sin(angular_frequency, phase))

    @staticmethod
    def square(double width):
        return _wrap_wave(cwaveform_wave_square(width))

    @staticmethod
    def builtin(int opcode, parameters=(), int64_t shift=0):
        cdef object array = np.ascontiguousarray(parameters, dtype=np.float64)
        cdef double[::1] view = array.reshape(-1)
        cdef size_t count = view.shape[0]
        cdef cwaveform_wave *wave
        with nogil:
            wave = cwaveform_wave_builtin(
                opcode, NULL if count == 0 else &view[0], count, shift)
        if wave == NULL:
            raise ValueError("invalid builtin waveform parameters")
        return _wrap_wave(wave)

    @classmethod
    def from_bytes(cls, data):
        cdef bytes block = bytes(data)
        cdef size_t size = len(block)
        cdef const uint8_t *pointer = <const uint8_t *>PyBytes_AS_STRING(block)
        cdef cwaveform_wave *wave
        cdef CWaveformCore result
        with nogil:
            wave = cwaveform_wave_from_bytes(pointer, size)
        if wave == NULL:
            raise ValueError("invalid WNF4 waveform block")
        result = _wrap_wave(wave)
        result._serialized = block
        return result

    def to_bytes(self):
        cdef size_t size = 0
        cdef const uint8_t *data
        if self._serialized is None:
            data = cwaveform_wave_bytes(self._pointer, &size)
            self._serialized = PyBytes_FromStringAndSize(
                <const char *>data, size)
        return self._serialized

    def add_affine(self, CWaveformCore other, int64_t left_delay=0,
                   double left_scale=1.0, int64_t right_delay=0,
                   double right_scale=1.0):
        cdef cwaveform_wave *wave
        with nogil:
            wave = cwaveform_wave_add_affine(
                self._pointer, left_delay, left_scale,
                other._pointer, right_delay, right_scale,
            )
        return _wrap_wave(wave)

    def mul_affine(self, CWaveformCore other, int64_t left_delay=0,
                   double left_scale=1.0, int64_t right_delay=0,
                   double right_scale=1.0):
        cdef cwaveform_wave *wave
        with nogil:
            wave = cwaveform_wave_mul_affine(
                self._pointer, left_delay, left_scale,
                other._pointer, right_delay, right_scale,
            )
        return _wrap_wave(wave)

    def materialize(self, int64_t delay=0, double scale=1.0):
        cdef cwaveform_wave *wave
        with nogil:
            wave = cwaveform_wave_materialize(self._pointer, delay, scale)
        return _wrap_wave(wave)

    def window(self, int64_t lower, int64_t upper):
        cdef cwaveform_wave *wave
        with nogil:
            wave = cwaveform_wave_window(self._pointer, lower, upper)
        return _wrap_wave(wave)

    def power(self, int exponent):
        cdef cwaveform_wave *wave
        with nogil:
            wave = cwaveform_wave_power(self._pointer, exponent)
        if wave == NULL:
            raise ValueError("waveform power must be an integer")
        return _wrap_wave(wave)

    def derivative(self, unsigned int order=1):
        cdef cwaveform_wave *wave
        with nogil:
            wave = cwaveform_wave_derivative(self._pointer, order)
        if wave == NULL:
            raise ValueError("derivative is not supported for this waveform")
        return _wrap_wave(wave)

    def filtered(self, double low=0.0, double high=np.inf,
                 double epsilon=1e-15):
        cdef cwaveform_wave *wave
        with nogil:
            wave = cwaveform_wave_filter(self._pointer, low, high, epsilon)
        return _wrap_wave(wave)

    def simplify(self, double epsilon=1e-15):
        cdef cwaveform_wave *wave
        with nogil:
            wave = cwaveform_wave_simplify(self._pointer, epsilon)
        return _wrap_wave(wave)

    def evaluate(self, positions, int64_t delay=0, double scale=1.0,
                 double lower=-np.inf, double upper=np.inf, out=None):
        cdef object source = np.ascontiguousarray(positions, dtype=np.float64)
        cdef object target
        cdef double[::1] source_view = source.reshape(-1)
        cdef double[::1] target_view
        cdef size_t count = source_view.shape[0]
        cdef int status
        if out is None:
            target = np.empty(source.shape, dtype=np.float64)
        else:
            target = np.asarray(out)
            if target.shape != source.shape or target.dtype != np.dtype(np.float64):
                raise ValueError("out must be a matching float64 array")
            if not target.flags.c_contiguous or not target.flags.writeable:
                raise ValueError("out must be writable and C-contiguous")
        if count == 0:
            return target
        target_view = target.reshape(-1)
        with nogil:
            status = cwaveform_wave_evaluate(
                self._pointer, &source_view[0], count, delay, scale,
                lower, upper, &target_view[0],
            )
        if status != 0:
            raise RuntimeError(f"C evaluate failed with status {status}")
        return target

    def sample(self, int64_t start_tick, Py_ssize_t count,
               int64_t step_numerator, int64_t step_denominator=1,
               int64_t delay=0, double scale=1.0,
               double lower=-np.inf, double upper=np.inf,
               int bits=0, double full_scale=1.0, out=None):
        cdef object target
        cdef void *target_pointer
        cdef int status
        if count < 0:
            raise ValueError("count must be non-negative")
        dtype = np.float64 if bits == 0 else (
            np.int16 if bits == 16 else np.int32 if bits == 32 else None
        )
        if dtype is None:
            raise ValueError("bits must be 0, 16, or 32")
        if out is None:
            target = np.empty(count, dtype=dtype)
        else:
            target = np.asarray(out)
            if target.shape != (count,) or target.dtype != np.dtype(dtype):
                raise ValueError("out has the wrong shape or dtype")
            if not target.flags.c_contiguous or not target.flags.writeable:
                raise ValueError("out must be writable and C-contiguous")
        if count == 0:
            return target
        target_pointer = <void *><size_t>target.ctypes.data
        with nogil:
            status = cwaveform_wave_sample(
                self._pointer, start_tick, count, step_numerator,
                step_denominator, delay, scale, lower, upper,
                bits, full_scale, target_pointer,
            )
        if status != 0:
            raise RuntimeError(f"C sample failed with status {status}")
        return target

    @property
    def lower_tick(self):
        return cwaveform_wave_lower_tick(self._pointer)

    @property
    def upper_tick(self):
        return cwaveform_wave_upper_tick(self._pointer)

    @property
    def node_count(self):
        return cwaveform_wave_node_count(self._pointer)

    @property
    def hash64(self):
        return cwaveform_wave_hash(self._pointer)

    def __eq__(self, other):
        return isinstance(other, CWaveformCore) and cwaveform_wave_equal(
            self._pointer, (<CWaveformCore>other)._pointer
        )

    def __hash__(self):
        return hash(self.to_bytes())

    def __reduce__(self):
        return (type(self).from_bytes, (self.to_bytes(),))


cdef class CWaveformStackCore:
    cdef cwaveform_stack *_pointer
    cdef object _serialized

    def __cinit__(self):
        self._pointer = NULL
        self._serialized = None

    def __dealloc__(self):
        if self._pointer != NULL:
            cwaveform_stack_release(self._pointer)

    cdef void _ensure_pointer(self) except *:
        cdef bytes block
        cdef const uint8_t *data
        cdef size_t size
        cdef cwaveform_stack *stack
        if self._pointer != NULL:
            return
        block = self._serialized
        size = len(block)
        data = <const uint8_t *>PyBytes_AS_STRING(block)
        with nogil:
            stack = cwaveform_stack_from_bytes(data, size)
        if stack == NULL:
            raise ValueError("invalid WNS4 stack block")
        self._pointer = stack

    @classmethod
    def from_events(cls, templates, template_ids, delays, scales):
        cdef Py_ssize_t template_count = len(templates)
        cdef object ids_array = np.ascontiguousarray(template_ids, dtype=np.uint32)
        cdef object delays_array = np.ascontiguousarray(delays, dtype=np.int64)
        cdef object scales_array = np.ascontiguousarray(scales, dtype=np.float64)
        cdef uint32_t[::1] ids_view = ids_array
        cdef int64_t[::1] delays_view = delays_array
        cdef double[::1] scales_view = scales_array
        cdef Py_ssize_t event_count = ids_view.shape[0]
        cdef cwaveform_wave **wave_pointers = NULL
        cdef cwaveform_stack *stack
        cdef Py_ssize_t index
        if delays_view.shape[0] != event_count or scales_view.shape[0] != event_count:
            raise ValueError("event arrays must have equal lengths")
        if template_count:
            wave_pointers = <cwaveform_wave **>malloc(
                template_count * sizeof(cwaveform_wave *)
            )
            if wave_pointers == NULL:
                raise MemoryError()
        try:
            for index in range(template_count):
                if not isinstance(templates[index], CWaveformCore):
                    raise TypeError("templates must contain CWaveformCore objects")
                wave_pointers[index] = (<CWaveformCore>templates[index])._pointer
            with nogil:
                stack = cwaveform_stack_create(
                    wave_pointers,
                    NULL if event_count == 0 else &ids_view[0],
                    NULL if event_count == 0 else &delays_view[0],
                    NULL if event_count == 0 else &scales_view[0],
                    template_count, event_count,
                )
        finally:
            free(wave_pointers)
        return _wrap_stack(stack)

    @classmethod
    def from_bytes(cls, data):
        cdef bytes block = bytes(data)
        cdef size_t size = len(block)
        cdef size_t template_count
        cdef size_t event_count
        cdef size_t cursor
        cdef size_t event_offset
        cdef size_t index
        cdef const uint8_t *pointer
        cdef CWaveformStackCore result
        if size < 16 or block[:8] != b"WNS4\x02\x00\x00\x00":
            raise ValueError("invalid WNS4 stack block")
        pointer = <const uint8_t *>PyBytes_AS_STRING(block)
        template_count = (pointer[8] | (pointer[9] << 8)
                          | (pointer[10] << 16) | (pointer[11] << 24))
        event_count = (pointer[12] | (pointer[13] << 8)
                       | (pointer[14] << 16) | (pointer[15] << 24))
        if template_count > (size - 16) // 4 or event_count > size // 20:
            raise ValueError("invalid WNS4 stack block")
        cursor = 16 + 4 * template_count
        event_offset = size - 20 * event_count
        if cursor > event_offset:
            raise ValueError("invalid WNS4 stack block")
        for index in range(template_count):
            cursor += (pointer[16 + 4 * index]
                       | (pointer[17 + 4 * index] << 8)
                       | (pointer[18 + 4 * index] << 16)
                       | (pointer[19 + 4 * index] << 24))
            if cursor > event_offset:
                raise ValueError("invalid WNS4 stack block")
        if cursor != event_offset:
            raise ValueError("invalid WNS4 stack block")
        result = cls.__new__(cls)
        result._serialized = block
        return result

    def to_bytes(self):
        cdef size_t size = 0
        cdef const uint8_t *data
        if self._serialized is None:
            self._ensure_pointer()
            data = cwaveform_stack_bytes(self._pointer, &size)
            self._serialized = PyBytes_FromStringAndSize(
                <const char *>data, size)
        return self._serialized

    def evaluate(self, positions, int64_t shift=0, double offset=0.0,
                 out=None):
        cdef object source = np.ascontiguousarray(positions, dtype=np.float64)
        cdef object target
        cdef double[::1] source_view = source.reshape(-1)
        cdef double[::1] target_view
        cdef size_t count = source_view.shape[0]
        cdef int status
        self._ensure_pointer()
        if out is None:
            target = np.empty(source.shape, dtype=np.float64)
        else:
            target = np.asarray(out)
            if target.shape != source.shape or target.dtype != np.dtype(np.float64):
                raise ValueError("out must be a matching float64 array")
            if not target.flags.c_contiguous or not target.flags.writeable:
                raise ValueError("out must be writable and C-contiguous")
        if count == 0:
            return target
        target_view = target.reshape(-1)
        with nogil:
            status = cwaveform_stack_evaluate(
                self._pointer, &source_view[0], count, shift, offset,
                &target_view[0],
            )
        if status != 0:
            raise RuntimeError(f"C stack evaluate failed with status {status}")
        return target

    def sample(self, int64_t start_tick, Py_ssize_t count,
               int64_t step_numerator, int64_t step_denominator=1,
               int64_t shift=0, double offset=0.0, int bits=0,
               double full_scale=1.0, out=None):
        cdef object target
        cdef void *target_pointer
        cdef int status
        self._ensure_pointer()
        if count < 0:
            raise ValueError("count must be non-negative")
        dtype = np.float64 if bits == 0 else (
            np.int16 if bits == 16 else np.int32 if bits == 32 else None
        )
        if dtype is None:
            raise ValueError("bits must be 0, 16, or 32")
        if out is None:
            target = np.empty(count, dtype=dtype)
        else:
            target = np.asarray(out)
            if target.shape != (count,) or target.dtype != np.dtype(dtype):
                raise ValueError("out has the wrong shape or dtype")
            if not target.flags.c_contiguous or not target.flags.writeable:
                raise ValueError("out must be writable and C-contiguous")
        if count == 0:
            return target
        target_pointer = <void *><size_t>target.ctypes.data
        with nogil:
            status = cwaveform_stack_sample(
                self._pointer, start_tick, count, step_numerator,
                step_denominator, shift, offset, bits, full_scale,
                target_pointer,
            )
        if status != 0:
            raise RuntimeError(f"C stack sample failed with status {status}")
        return target

    def prepare_sample(self, int64_t start_tick, Py_ssize_t count,
                       int64_t step_numerator,
                       int64_t step_denominator=1,
                       int64_t shift=0):
        cdef cwaveform_sample_plan *plan
        if count < 0:
            raise ValueError("count must be non-negative")
        self._ensure_pointer()
        with nogil:
            plan = cwaveform_sample_plan_create(
                self._pointer, start_tick, count, step_numerator,
                step_denominator, shift,
            )
        return _wrap_plan(plan)

    def simplify(self, int64_t shift=0, double offset=0.0):
        cdef cwaveform_wave *wave
        self._ensure_pointer()
        with nogil:
            wave = cwaveform_stack_simplify(self._pointer, shift, offset)
        return _wrap_wave(wave)

    def materialize(self, int64_t shift=0, double offset=0.0):
        cdef cwaveform_stack *stack
        self._ensure_pointer()
        with nogil:
            stack = cwaveform_stack_materialize(self._pointer, shift, offset)
        return _wrap_stack(stack)

    def combine(self, CWaveformStackCore other, int64_t left_shift=0,
                int64_t right_shift=0):
        cdef cwaveform_stack *stack
        self._ensure_pointer()
        other._ensure_pointer()
        with nogil:
            stack = cwaveform_stack_combine(
                self._pointer, left_shift, other._pointer, right_shift)
        return _wrap_stack(stack)

    def append(self, CWaveformCore wave, int64_t global_shift=0,
               int64_t wave_delay=0, double wave_scale=1.0):
        cdef cwaveform_stack *stack
        self._ensure_pointer()
        with nogil:
            stack = cwaveform_stack_append(
                self._pointer, global_shift, wave._pointer,
                wave_delay, wave_scale)
        return _wrap_stack(stack)

    def scaled(self, double scale):
        cdef cwaveform_stack *stack
        self._ensure_pointer()
        with nogil:
            stack = cwaveform_stack_scale(self._pointer, scale)
        return _wrap_stack(stack)

    @property
    def event_count(self):
        self._ensure_pointer()
        return cwaveform_stack_event_count(self._pointer)

    @property
    def template_count(self):
        self._ensure_pointer()
        return cwaveform_stack_template_count(self._pointer)

    @property
    def hash64(self):
        self._ensure_pointer()
        return cwaveform_stack_hash(self._pointer)

    def __hash__(self):
        return hash(self.to_bytes())

    def __reduce__(self):
        return (type(self).from_bytes, (self.to_bytes(),))


cdef class CWaveformSamplePlan:
    cdef cwaveform_sample_plan *_pointer

    def __cinit__(self):
        self._pointer = NULL

    def __dealloc__(self):
        if self._pointer != NULL:
            cwaveform_sample_plan_release(self._pointer)

    def sample(self, double offset=0.0, int bits=0,
               double full_scale=1.0, out=None):
        cdef object target
        cdef void *target_pointer
        cdef int status
        cdef size_t count = cwaveform_sample_plan_count(self._pointer)
        dtype = np.float64 if bits == 0 else (
            np.int16 if bits == 16 else np.int32 if bits == 32 else None
        )
        if dtype is None:
            raise ValueError("bits must be 0, 16, or 32")
        if out is None:
            target = np.empty(count, dtype=dtype)
        else:
            target = np.asarray(out)
            if target.shape != (count,) or target.dtype != np.dtype(dtype):
                raise ValueError("out has the wrong shape or dtype")
            if not target.flags.c_contiguous or not target.flags.writeable:
                raise ValueError("out must be writable and C-contiguous")
        if count == 0:
            return target
        target_pointer = <void *><size_t>target.ctypes.data
        with nogil:
            status = cwaveform_sample_plan_sample(
                self._pointer, offset, bits, full_scale, target_pointer,
            )
        if status != 0:
            raise RuntimeError(
                f"C sample plan failed with status {status}"
            )
        return target

    @property
    def count(self):
        return cwaveform_sample_plan_count(self._pointer)

    @property
    def group_count(self):
        return cwaveform_sample_plan_group_count(self._pointer)

    @property
    def non_overlapping(self):
        return bool(cwaveform_sample_plan_non_overlapping(self._pointer))


def c_format_description():
    return (<bytes>cwaveform_format_description()).decode("ascii")


TICKS_PER_SECOND = cwaveform_ticks_per_second()


def set_c_ticks_per_second(value):
    """Configure the process clock before constructing C handles."""
    cdef uint64_t ticks = int(value)
    if ticks == 0 or cwaveform_set_ticks_per_second(ticks) != 0:
        raise ValueError("ticks_per_second must be a positive integer")


def get_c_ticks_per_second():
    return cwaveform_ticks_per_second()
