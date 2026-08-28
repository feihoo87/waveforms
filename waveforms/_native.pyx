# cython: language_level=3
"""Thin CPython binding for the Python-independent WNF4/WNS4 C core."""

from libc.stdint cimport int16_t, int32_t, int64_t, uint8_t, uint32_t, uint64_t
from libc.stdlib cimport free, malloc
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize

import numpy as np


cdef extern from "wf_native.h":
    ctypedef struct wf_native_wave:
        pass
    ctypedef struct wf_native_stack:
        pass

    uint64_t wf_native_ticks_per_second() noexcept nogil
    wf_native_wave *wf_native_wave_constant(double) noexcept nogil
    wf_native_wave *wf_native_wave_gaussian(double) noexcept nogil
    wf_native_wave *wf_native_wave_cos(double, double) noexcept nogil
    wf_native_wave *wf_native_wave_sin(double, double) noexcept nogil
    wf_native_wave *wf_native_wave_square(double) noexcept nogil
    wf_native_wave *wf_native_wave_from_bytes(const uint8_t *, size_t) noexcept nogil
    wf_native_wave *wf_native_wave_add_affine(
        const wf_native_wave *, int64_t, double,
        const wf_native_wave *, int64_t, double) noexcept nogil
    wf_native_wave *wf_native_wave_mul_affine(
        const wf_native_wave *, int64_t, double,
        const wf_native_wave *, int64_t, double) noexcept nogil
    wf_native_wave *wf_native_wave_materialize(
        const wf_native_wave *, int64_t, double) noexcept nogil
    void wf_native_wave_retain(wf_native_wave *) noexcept nogil
    void wf_native_wave_release(wf_native_wave *) noexcept nogil
    const uint8_t *wf_native_wave_bytes(const wf_native_wave *, size_t *) noexcept nogil
    uint64_t wf_native_wave_hash(const wf_native_wave *) noexcept nogil
    int wf_native_wave_equal(const wf_native_wave *, const wf_native_wave *) noexcept nogil
    int64_t wf_native_wave_lower_tick(const wf_native_wave *) noexcept nogil
    int64_t wf_native_wave_upper_tick(const wf_native_wave *) noexcept nogil
    uint32_t wf_native_wave_node_count(const wf_native_wave *) noexcept nogil
    int wf_native_wave_evaluate(
        const wf_native_wave *, const double *, size_t, int64_t, double,
        double, double, double *) noexcept nogil
    int wf_native_wave_sample(
        const wf_native_wave *, int64_t, size_t, int64_t, int64_t, int64_t,
        double, double, double, int, double, void *) noexcept nogil

    wf_native_stack *wf_native_stack_create(
        wf_native_wave *const *, const uint32_t *, const int64_t *,
        const double *, size_t, size_t) noexcept nogil
    wf_native_stack *wf_native_stack_from_bytes(
        const uint8_t *, size_t) noexcept nogil
    wf_native_stack *wf_native_stack_materialize(
        const wf_native_stack *, int64_t, double) noexcept nogil
    void wf_native_stack_retain(wf_native_stack *) noexcept nogil
    void wf_native_stack_release(wf_native_stack *) noexcept nogil
    const uint8_t *wf_native_stack_bytes(
        const wf_native_stack *, size_t *) noexcept nogil
    uint64_t wf_native_stack_hash(const wf_native_stack *) noexcept nogil
    size_t wf_native_stack_event_count(const wf_native_stack *) noexcept nogil
    size_t wf_native_stack_template_count(const wf_native_stack *) noexcept nogil
    int wf_native_stack_evaluate(
        const wf_native_stack *, const double *, size_t, int64_t, double,
        double *) noexcept nogil
    int wf_native_stack_sample(
        const wf_native_stack *, int64_t, size_t, int64_t, int64_t, int64_t,
        double, int, double, void *) noexcept nogil
    wf_native_wave *wf_native_stack_simplify(
        const wf_native_stack *, int64_t, double) noexcept nogil
    const char *wf_native_format_description() noexcept nogil


cdef NativeCore _wrap_wave(wf_native_wave *pointer):
    if pointer == NULL:
        raise MemoryError("native waveform construction failed")
    cdef NativeCore result = NativeCore.__new__(NativeCore)
    result._pointer = pointer
    result._serialized = None
    return result


cdef NativeStackCore _wrap_stack(wf_native_stack *pointer):
    if pointer == NULL:
        raise MemoryError("native stack construction failed")
    cdef NativeStackCore result = NativeStackCore.__new__(NativeStackCore)
    result._pointer = pointer
    result._serialized = None
    return result


cdef class NativeCore:
    cdef wf_native_wave *_pointer
    cdef object _serialized

    def __cinit__(self):
        self._pointer = NULL
        self._serialized = None

    def __dealloc__(self):
        if self._pointer != NULL:
            wf_native_wave_release(self._pointer)

    @staticmethod
    def constant(double value):
        return _wrap_wave(wf_native_wave_constant(value))

    @staticmethod
    def gaussian(double width):
        return _wrap_wave(wf_native_wave_gaussian(width))

    @staticmethod
    def cos(double angular_frequency, double phase=0.0):
        return _wrap_wave(wf_native_wave_cos(angular_frequency, phase))

    @staticmethod
    def sin(double angular_frequency, double phase=0.0):
        return _wrap_wave(wf_native_wave_sin(angular_frequency, phase))

    @staticmethod
    def square(double width):
        return _wrap_wave(wf_native_wave_square(width))

    @classmethod
    def from_bytes(cls, data):
        cdef bytes block = bytes(data)
        cdef size_t size = len(block)
        cdef const uint8_t *pointer = <const uint8_t *>PyBytes_AS_STRING(block)
        cdef wf_native_wave *wave
        cdef NativeCore result
        with nogil:
            wave = wf_native_wave_from_bytes(pointer, size)
        if wave == NULL:
            raise ValueError("invalid WNF4 waveform block")
        result = _wrap_wave(wave)
        result._serialized = block
        return result

    def to_bytes(self):
        cdef size_t size = 0
        cdef const uint8_t *data
        if self._serialized is None:
            data = wf_native_wave_bytes(self._pointer, &size)
            self._serialized = PyBytes_FromStringAndSize(
                <const char *>data, size)
        return self._serialized

    def add_affine(self, NativeCore other, int64_t left_delay=0,
                   double left_scale=1.0, int64_t right_delay=0,
                   double right_scale=1.0):
        cdef wf_native_wave *wave
        with nogil:
            wave = wf_native_wave_add_affine(
                self._pointer, left_delay, left_scale,
                other._pointer, right_delay, right_scale,
            )
        return _wrap_wave(wave)

    def mul_affine(self, NativeCore other, int64_t left_delay=0,
                   double left_scale=1.0, int64_t right_delay=0,
                   double right_scale=1.0):
        cdef wf_native_wave *wave
        with nogil:
            wave = wf_native_wave_mul_affine(
                self._pointer, left_delay, left_scale,
                other._pointer, right_delay, right_scale,
            )
        return _wrap_wave(wave)

    def materialize(self, int64_t delay=0, double scale=1.0):
        cdef wf_native_wave *wave
        with nogil:
            wave = wf_native_wave_materialize(self._pointer, delay, scale)
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
            status = wf_native_wave_evaluate(
                self._pointer, &source_view[0], count, delay, scale,
                lower, upper, &target_view[0],
            )
        if status != 0:
            raise RuntimeError(f"native evaluate failed with status {status}")
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
            status = wf_native_wave_sample(
                self._pointer, start_tick, count, step_numerator,
                step_denominator, delay, scale, lower, upper,
                bits, full_scale, target_pointer,
            )
        if status != 0:
            raise RuntimeError(f"native sample failed with status {status}")
        return target

    @property
    def lower_tick(self):
        return wf_native_wave_lower_tick(self._pointer)

    @property
    def upper_tick(self):
        return wf_native_wave_upper_tick(self._pointer)

    @property
    def node_count(self):
        return wf_native_wave_node_count(self._pointer)

    @property
    def hash64(self):
        return wf_native_wave_hash(self._pointer)

    def __eq__(self, other):
        return isinstance(other, NativeCore) and wf_native_wave_equal(
            self._pointer, (<NativeCore>other)._pointer
        )

    def __hash__(self):
        return hash(self.to_bytes())

    def __reduce__(self):
        return (type(self).from_bytes, (self.to_bytes(),))


cdef class NativeStackCore:
    cdef wf_native_stack *_pointer
    cdef object _serialized

    def __cinit__(self):
        self._pointer = NULL
        self._serialized = None

    def __dealloc__(self):
        if self._pointer != NULL:
            wf_native_stack_release(self._pointer)

    cdef void _ensure_pointer(self) except *:
        cdef bytes block
        cdef const uint8_t *data
        cdef size_t size
        cdef wf_native_stack *stack
        if self._pointer != NULL:
            return
        block = self._serialized
        size = len(block)
        data = <const uint8_t *>PyBytes_AS_STRING(block)
        with nogil:
            stack = wf_native_stack_from_bytes(data, size)
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
        cdef wf_native_wave **wave_pointers = NULL
        cdef wf_native_stack *stack
        cdef Py_ssize_t index
        if delays_view.shape[0] != event_count or scales_view.shape[0] != event_count:
            raise ValueError("event arrays must have equal lengths")
        if template_count:
            wave_pointers = <wf_native_wave **>malloc(
                template_count * sizeof(wf_native_wave *)
            )
            if wave_pointers == NULL:
                raise MemoryError()
        try:
            for index in range(template_count):
                if not isinstance(templates[index], NativeCore):
                    raise TypeError("templates must contain NativeCore objects")
                wave_pointers[index] = (<NativeCore>templates[index])._pointer
            with nogil:
                stack = wf_native_stack_create(
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
        cdef NativeStackCore result
        if size < 16 or block[:8] != b"WNS4\x01\x00\x00\x00":
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
            data = wf_native_stack_bytes(self._pointer, &size)
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
            status = wf_native_stack_evaluate(
                self._pointer, &source_view[0], count, shift, offset,
                &target_view[0],
            )
        if status != 0:
            raise RuntimeError(f"native stack evaluate failed with status {status}")
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
            status = wf_native_stack_sample(
                self._pointer, start_tick, count, step_numerator,
                step_denominator, shift, offset, bits, full_scale,
                target_pointer,
            )
        if status != 0:
            raise RuntimeError(f"native stack sample failed with status {status}")
        return target

    def simplify(self, int64_t shift=0, double offset=0.0):
        cdef wf_native_wave *wave
        self._ensure_pointer()
        with nogil:
            wave = wf_native_stack_simplify(self._pointer, shift, offset)
        return _wrap_wave(wave)

    def materialize(self, int64_t shift=0, double offset=0.0):
        cdef wf_native_stack *stack
        self._ensure_pointer()
        with nogil:
            stack = wf_native_stack_materialize(self._pointer, shift, offset)
        return _wrap_stack(stack)

    @property
    def event_count(self):
        self._ensure_pointer()
        return wf_native_stack_event_count(self._pointer)

    @property
    def template_count(self):
        self._ensure_pointer()
        return wf_native_stack_template_count(self._pointer)

    @property
    def hash64(self):
        self._ensure_pointer()
        return wf_native_stack_hash(self._pointer)

    def __hash__(self):
        return hash(self.to_bytes())

    def __reduce__(self):
        return (type(self).from_bytes, (self.to_bytes(),))


def native_format_description():
    return (<bytes>wf_native_format_description()).decode("ascii")


TICKS_PER_SECOND = wf_native_ticks_per_second()
