# cython: language_level=3
"""Real-only packed binary core for :mod:`waveforms.waveform`.

The binary block is the authoritative representation; a cached normalized
expression is decoded only when sampling or symbolic algebra needs it. WFM3
stores int64 boundary ticks, expression-template ids, int64 per-segment shifts,
and deduplicated real expression bytecode. WVS3 stores deduplicated WFM3
templates followed by struct-of-arrays event ids, delay ticks, and real scales.
"""

import struct
from bisect import bisect_left
from functools import lru_cache
from itertools import chain, product
from math import comb, factorial

import numpy as np
import scipy.special as special
from numpy import inf, pi

NDIGITS = 15

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

_zero = ((), ())
_one = ((((), ()),), (1.0,))

_MAGIC = b"WFM3"
_STACK_MAGIC = b"WVS3"
_VERSION = 3
_HEADER = struct.Struct("<4sHHII")
_STACK_HEADER = struct.Struct("<4sHHIII")

_TIME_RESOLUTION = 1e-12
_TIME_LOCKED = False
_INF_TICK = np.iinfo(np.int64).max


def set_time_resolution(value):
    """Set the process-wide tick duration before creating any packed object."""
    global _TIME_RESOLUTION, _TIME_LOCKED
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("time resolution must be a finite positive number")
    if _TIME_LOCKED and value != _TIME_RESOLUTION:
        raise RuntimeError("time resolution is locked by existing waveform objects")
    _TIME_RESOLUTION = value


def get_time_resolution():
    return _TIME_RESOLUTION


def quantize_time(value):
    if value == inf:
        return inf
    if value == -inf:
        return -inf
    return _tick_to_time(_time_to_tick(value))


def _lock_time_resolution():
    global _TIME_LOCKED
    _TIME_LOCKED = True


def _time_to_tick(value):
    if value == inf:
        return int(_INF_TICK)
    if value == -inf:
        return -int(_INF_TICK)
    tick = int(round(float(value) / _TIME_RESOLUTION))
    if tick <= -int(_INF_TICK) or tick >= int(_INF_TICK):
        raise OverflowError("time is outside the waveform tick range")
    return tick


def _tick_to_time(tick):
    if tick == int(_INF_TICK):
        return inf
    if tick == -int(_INF_TICK):
        return -inf
    return int(tick) * _TIME_RESOLUTION


def _LINEAR(t):
    return t


def _GAUSSIAN(t, std_sq2):
    return np.exp(-(t / std_sq2) ** 2)


def _D_GAUSSIAN(t, std_sq2, n):
    x = t / std_sq2
    return ((-1) ** n / std_sq2 ** n * special.eval_hermite(n, x)
            * np.exp(-(x ** 2)))


def _ERF(t, std_sq2):
    return special.erf(t / std_sq2)


def _COS(t, w):
    return np.cos(w * t)


def _SINC(t, bw):
    return np.sinc(bw * t)


def _EXP(t, alpha):
    return np.exp(alpha * t)


@lru_cache(maxsize=256)
def _interp_grid(start, stop, size):
    return np.linspace(start, stop, size)


def _INTERP(t, start, stop, points):
    return np.interp(t, _interp_grid(start, stop, len(points)), points)


def _LINEARCHIRP(t, f0, f1, duration, phi0):
    return np.sin(phi0 + 2 * pi * ((f1 - f0) / (2 * duration) * t ** 2
                                  + f0 * t))


def _EXPONENTIALCHIRP(t, f0, alpha, phi0):
    return np.sin(phi0 + 2 * pi * f0 * (np.exp(alpha * t) - 1) / alpha)


def _HYPERBOLICCHIRP(t, f0, k, phi0):
    return np.sin(phi0 + 2 * pi * f0 / k * np.log(1 + k * t))


def _COSH(t, w):
    return np.cosh(w * t)


def _SINH(t, w):
    return np.sinh(w * t)


def _drag(t, t0, freq, width, delta, block_freq, phase):
    omega = pi / width
    omega_x = np.sin(omega * (t - t0)) ** 2
    wt = 2 * pi * (freq + delta) * t - (2 * pi * delta * t0 + phase)
    if block_freq is None or block_freq - delta == 0:
        return omega_x * np.cos(wt)
    b = 1 / pi / 2 / (block_freq - delta)
    omega_y = -b * omega * np.sin(2 * omega * (t - t0))
    return omega_x * np.cos(wt) + omega_y * np.sin(wt)


def _b_series_matrix(block_coefficients):
    matrix = np.zeros((len(block_coefficients) + 1, 2, 2))
    matrix[0] = np.identity(2)
    for coefficient in block_coefficients:
        rotation = np.array([[0, coefficient], [-coefficient, 0]])
        matrix[1:] += matrix[:-1] @ rotation
    return matrix


def _sin_power_derivatives(int power, int order, double angular_frequency):
    matrix = np.zeros((order + 1, power + 1))
    matrix[0, power] = 1
    for derivative in range(1, order + 1):
        if derivative % 2:
            matrix[derivative, :-1] = (
                matrix[derivative - 1, 1:] * np.arange(1, power + 1)
                * angular_frequency
            )
        else:
            matrix[derivative, :-2] = (
                matrix[derivative - 2, 2:] * np.arange(1, power)
                * np.arange(2, power + 1)
            )
            matrix[derivative] -= (
                matrix[derivative - 2] * np.arange(power + 1) ** 2
            )
            matrix[derivative] *= angular_frequency ** 2
    return matrix


def _edge_polynomial(derivatives, double x):
    values = np.asarray(derivatives, dtype=float).copy()
    values[0] -= 1
    size = values.shape[0]
    matrix = np.zeros((size, size))
    for derivative in range(size):
        for coefficient in range(size):
            matrix[derivative, coefficient] = (
                x ** (size + coefficient - derivative)
                * factorial(size + coefficient)
                / factorial(size + coefficient - derivative)
            )
    coefficients = np.linalg.solve(matrix, values)
    return np.poly1d((*np.flip(coefficients), *np.zeros_like(values[:-1]), 1))


@lru_cache(maxsize=128)
def _drag_sin_plan(double width, double delta, tuple block_freq):
    frequencies = np.asarray(block_freq, dtype=float)
    block_coefficients = (
        np.empty(0) if frequencies.size == 0
        else 1 / (2 * pi * (frequencies - delta))
    )
    power = max(((len(block_coefficients) + 2) >> 1) << 1, 2)
    transform = _b_series_matrix(block_coefficients)
    derivatives = _sin_power_derivatives(
        power, len(block_coefficients), pi / width
    )
    peak_basis = np.ones(power + 1)
    peak_basis[1::2] = 0
    peak_derivatives = derivatives @ peak_basis
    transformed_peak = np.einsum(
        "ijk,ki->j",
        transform,
        np.array([peak_derivatives, np.zeros_like(peak_derivatives)]),
    )
    normalization = np.sqrt(np.sum(np.abs(transformed_peak) ** 2))
    return transform, derivatives, np.arange(power + 1), normalization


def _drag_sin_derivatives(t, double t0, double width, double plateau,
                          derivatives, powers):
    midpoint = t0 + width / 2
    plateau_stop = midpoint + plateau
    plateau_mask = (t > midpoint) & (t < plateau_stop)
    adjusted = np.where(t >= plateau_stop, t - plateau, t)
    angle = pi / width * (adjusted - t0)
    sine = np.sin(angle)
    cosine = np.cos(angle)
    if np.any(plateau_mask):
        sine = sine.copy()
        cosine = cosine.copy()
        sine[plateau_mask] = 0
        cosine[plateau_mask] = 0
    basis = sine ** powers.reshape((-1, 1))
    basis[1::2] *= cosine
    values = derivatives @ basis
    if np.any(plateau_mask):
        values[0, plateau_mask] = 1
    return values


def _drag_omega_sin(t, double t0, double width, double delta,
                    tuple block_freq, double plateau):
    transform, derivatives, powers, normalization = _drag_sin_plan(
        width, delta, block_freq
    )
    values = _drag_sin_derivatives(
        t, t0, width, plateau, derivatives, powers
    )
    components = np.array([values, np.zeros_like(values)])
    return np.einsum("ijk,kim->jm", transform, components) / normalization


@lru_cache(maxsize=128)
def _drag_sinx_plan(double width, double delta, tuple block_freq, double tab):
    transform, derivatives, powers, _ = _drag_sin_plan(
        width, delta, block_freq
    )
    angular_frequency = pi / width

    left_basis = np.sin(angular_frequency * (1 - tab) * width / 2) ** powers
    left_basis[1::2] *= np.cos(angular_frequency * (1 - tab) * width / 2)
    left_polynomial = _edge_polynomial(
        derivatives @ left_basis, -tab * width / 2
    )

    right_basis = np.sin(angular_frequency * (1 + tab) * width / 2) ** powers
    right_basis[1::2] *= np.cos(angular_frequency * (1 + tab) * width / 2)
    right_polynomial = _edge_polynomial(
        derivatives @ right_basis, tab * width / 2
    )

    order = len(block_freq) + 1
    left_derivatives = tuple(
        np.polyder(left_polynomial, derivative) for derivative in range(order)
    )
    right_derivatives = tuple(
        np.polyder(right_polynomial, derivative) for derivative in range(order)
    )
    return (transform, derivatives, powers, left_derivatives,
            right_derivatives)


def _drag_omega_sinx(t, double t0, double width, double delta,
                     tuple block_freq, double plateau, double tab):
    (transform, derivatives, powers, left_derivatives,
     right_derivatives) = _drag_sinx_plan(width, delta, block_freq, tab)
    values = _drag_sin_derivatives(
        t, t0, width, plateau, derivatives, powers
    )
    midpoint = t0 + width / 2
    plateau_stop = midpoint + plateau
    left_mask = (t >= midpoint - tab * width / 2) & (t <= midpoint)
    right_mask = ((t >= plateau_stop)
                  & (t <= plateau_stop + tab * width / 2))
    for derivative in range(len(block_freq) + 1):
        values[derivative, left_mask] = left_derivatives[derivative](
            t[left_mask] - midpoint
        )
        values[derivative, right_mask] = right_derivatives[derivative](
            t[right_mask] - plateau_stop
        )
    components = np.array([values, np.zeros_like(values)])
    return np.einsum("ijk,kim->jm", transform, components)


def _drag_sin(t, t0, freq, width, delta, block_freq, phase, plateau):
    omega_x, omega_y = _drag_omega_sin(
        t, t0, width, delta, block_freq, plateau
    )
    wt = 2 * pi * (freq + delta) * t - (2 * pi * delta * t0 + phase)
    return omega_x * np.cos(wt) + omega_y * np.sin(wt)


def _drag_sinx(t, t0, freq, width, delta, block_freq, phase, plateau, tab):
    omega_x, omega_y = _drag_omega_sinx(
        t, t0, width, delta, block_freq, plateau, tab
    )
    wt = 2 * pi * (freq + delta) * t - (2 * pi * delta * t0 + phase)
    return omega_x * np.cos(wt) + omega_y * np.sin(wt)


@lru_cache(maxsize=64)
def _mollifier_poly(d):
    polynomial = np.poly1d([-2, 0])
    for n in range(1, d):
        polynomial = (np.poly1d([1, 0, -2, 0, 1]) * polynomial.deriv()
                      + np.poly1d([-4 * n, 0, 4 * n - 2, 0]) * polynomial)
    return polynomial


def _MOLLIFIER(t, radius, derivative):
    x = t / radius
    xx_1 = x * x - 1
    if derivative == 0:
        return np.where(xx_1 >= 0, 0, np.exp(1 / xx_1 + 1))
    polynomial = _mollifier_poly(derivative)
    return (np.where(xx_1 >= 0, 0,
                     np.exp(1 / xx_1 + 1) / (-xx_1) ** (2 * derivative))
            * polynomial(x) / radius ** derivative)


_base_functions = {
    LINEAR: _LINEAR,
    GAUSSIAN: _GAUSSIAN,
    ERF: _ERF,
    COS: _COS,
    SINC: _SINC,
    EXP: _EXP,
    INTERP: _INTERP,
    LINEARCHIRP: _LINEARCHIRP,
    EXPONENTIALCHIRP: _EXPONENTIALCHIRP,
    HYPERBOLICCHIRP: _HYPERBOLICCHIRP,
    COSH: _COSH,
    SINH: _SINH,
    DRAG: _drag,
    MOLLIFIER: _MOLLIFIER,
    D_GAUSSIAN: _D_GAUSSIAN,
    DRAG_SIN: _drag_sin,
    DRAG_SINX: _drag_sinx,
}


cdef object _calc_expr(object expression, object x):
    cdef dict cache = {}
    cdef object total = None
    cdef object product_value
    cdef object value
    cdef object function
    cdef object args
    cdef object shift_value
    cdef object term_value
    cdef Py_ssize_t i, j
    terms, coefficients = expression
    for i in range(len(terms)):
        functions, powers = terms[i]
        product_value = None
        for j in range(len(functions)):
            function = functions[j]
            value = cache.get(function)
            if value is None:
                args = function[1:-1]
                shift_value = function[-1]
                value = _base_functions[function[0]](
                    x if shift_value == 0 else x - shift_value, *args)
                cache[function] = value
            if powers[j] != 1:
                value = value ** powers[j]
            product_value = value if product_value is None else product_value * value
        if product_value is None:
            term_value = coefficients[i]
        elif coefficients[i] == 1:
            term_value = product_value
        else:
            term_value = coefficients[i] * product_value
        total = term_value if total is None else total + term_value
    return 0 if total is None else total


def _normal_number(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _real_number(value):
    value = _normal_number(value)
    if np.iscomplexobj(value):
        value = complex(value)
        if value.imag != 0:
            raise TypeError("packed waveform coefficients must be real")
        value = value.real
    return float(value)


def _encode_builtin_args(bytearray out, int opcode, args):
    args = tuple(_normal_number(value) for value in args)
    if opcode == LINEAR:
        out.extend(struct.pack("<q", _time_to_tick(args[0])))
    elif opcode in (GAUSSIAN, ERF):
        out.extend(struct.pack("<2q", _time_to_tick(args[0]),
                               _time_to_tick(args[1])))
    elif opcode in (COS, SINC, EXP, COSH, SINH):
        out.extend(struct.pack("<dq", float(args[0]), _time_to_tick(args[1])))
    elif opcode == INTERP:
        start, stop, points, shift = args
        points = np.asarray(points, dtype="<f8").reshape(-1)
        out.extend(struct.pack("<3qI", _time_to_tick(start),
                               _time_to_tick(stop), _time_to_tick(shift),
                               len(points)))
        out.extend(points.tobytes())
    elif opcode == LINEARCHIRP:
        out.extend(struct.pack("<ddqdq", float(args[0]), float(args[1]),
                               _time_to_tick(args[2]), float(args[3]),
                               _time_to_tick(args[4])))
    elif opcode in (EXPONENTIALCHIRP, HYPERBOLICCHIRP):
        out.extend(struct.pack("<3dq", *(float(value) for value in args[:-1]),
                               _time_to_tick(args[-1])))
    elif opcode == DRAG:
        values = list(args)
        values[4] = np.nan if values[4] is None else values[4]
        out.extend(struct.pack("<qdqdddq", _time_to_tick(values[0]),
                               float(values[1]), _time_to_tick(values[2]),
                               float(values[3]), float(values[4]),
                               float(values[5]), _time_to_tick(values[6])))
    elif opcode in (DRAG_SIN, DRAG_SINX):
        t0, freq, width, delta, block_freq, phase, plateau = args[:7]
        block_freq = tuple(float(value) for value in block_freq)
        out.extend(struct.pack("<qdqddq", _time_to_tick(t0), float(freq),
                               _time_to_tick(width), float(delta),
                               float(phase), _time_to_tick(plateau)))
        if opcode == DRAG_SINX:
            out.extend(struct.pack("<d", float(args[7])))
        out.extend(struct.pack("<I", len(block_freq)))
        if block_freq:
            out.extend(struct.pack(f"<{len(block_freq)}d", *block_freq))
        out.extend(struct.pack("<q", _time_to_tick(args[-1])))
    elif opcode in (MOLLIFIER, D_GAUSSIAN):
        out.extend(struct.pack("<qiq", _time_to_tick(args[0]), int(args[1]),
                               _time_to_tick(args[2])))
    else:
        raise ValueError(f"unsupported waveform opcode {opcode}")


def _decode_builtin_args(bytes data, Py_ssize_t pos, int opcode):
    if opcode == LINEAR:
        shift = struct.unpack_from("<q", data, pos)[0]
        return (_tick_to_time(shift),), pos + 8
    if opcode in (GAUSSIAN, ERF):
        value, shift = struct.unpack_from("<2q", data, pos)
        return (_tick_to_time(value), _tick_to_time(shift)), pos + 16
    if opcode in (COS, SINC, EXP, COSH, SINH):
        value, shift = struct.unpack_from("<dq", data, pos)
        return (value, _tick_to_time(shift)), pos + 16
    if opcode == INTERP:
        start, stop, shift, count = struct.unpack_from("<3qI", data, pos)
        pos += 28
        points = struct.unpack_from(f"<{count}d", data, pos)
        return (_tick_to_time(start), _tick_to_time(stop), points,
                _tick_to_time(shift)), pos + 8 * count
    if opcode == LINEARCHIRP:
        f0, f1, duration, phi0, shift = struct.unpack_from("<ddqdq", data, pos)
        return (f0, f1, _tick_to_time(duration), phi0,
                _tick_to_time(shift)), pos + 40
    if opcode in (EXPONENTIALCHIRP, HYPERBOLICCHIRP):
        a, b, c, shift = struct.unpack_from("<3dq", data, pos)
        return (a, b, c, _tick_to_time(shift)), pos + 32
    if opcode == DRAG:
        values = list(struct.unpack_from("<qdqdddq", data, pos))
        values[0] = _tick_to_time(values[0])
        values[2] = _tick_to_time(values[2])
        if np.isnan(values[4]):
            values[4] = None
        values[-1] = _tick_to_time(values[-1])
        return tuple(values), pos + 56
    if opcode in (DRAG_SIN, DRAG_SINX):
        t0, freq, width, delta, phase, plateau = struct.unpack_from(
            "<qdqddq", data, pos
        )
        pos += 48
        tab = None
        if opcode == DRAG_SINX:
            tab = struct.unpack_from("<d", data, pos)[0]
            pos += 8
        count = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        block_freq = struct.unpack_from(f"<{count}d", data, pos)
        pos += 8 * count
        shift = struct.unpack_from("<q", data, pos)[0]
        pos += 8
        values = (_tick_to_time(t0), freq, _tick_to_time(width), delta,
                  block_freq, phase, _tick_to_time(plateau))
        if opcode == DRAG_SINX:
            values += (tab,)
        return values + (_tick_to_time(shift),), pos
    if opcode in (MOLLIFIER, D_GAUSSIAN):
        first, order, shift = struct.unpack_from("<qiq", data, pos)
        return (_tick_to_time(first), order, _tick_to_time(shift)), pos + 20
    raise ValueError(f"unsupported waveform opcode {opcode}")


def _encode_expr(expr):
    terms, coefficients = expr
    out = bytearray(struct.pack("<I", len(coefficients)))
    for (functions, powers), coefficient in zip(terms, coefficients):
        out.extend(struct.pack("<dI", _real_number(coefficient),
                               len(functions)))
        for function, power in zip(functions, powers):
            opcode, *args = function
            out.extend(struct.pack("<Hi", int(opcode), int(power)))
            _encode_builtin_args(out, int(opcode), args)
    return bytes(out)


def _decode_expr(bytes data):
    cdef Py_ssize_t pos = 0
    count = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    terms = []
    coefficients = []
    for _ in range(count):
        coefficient, factor_count = struct.unpack_from("<dI", data, pos)
        pos += 12
        coefficients.append(coefficient)
        functions = []
        powers = []
        for _ in range(factor_count):
            opcode, power = struct.unpack_from("<Hi", data, pos)
            pos += 6
            args, pos = _decode_builtin_args(data, pos, opcode)
            functions.append((opcode, *args))
            powers.append(power)
        terms.append((tuple(functions), tuple(powers)))
    if pos != len(data):
        raise ValueError("trailing data in packed expression")
    return tuple(terms), tuple(coefficients)


def _normalize_expression(expr, outer_shift=0.0):
    inner_shift = 0.0
    has_function = False
    for term in expr[0]:
        if term[0]:
            has_function = True
            inner_shift = quantize_time(term[0][0][-1])
            break
    if not has_function:
        return expr, 0.0
    if inner_shift:
        expr = _expr_shift(expr, -inner_shift)
    return expr, quantize_time(outer_shift + inner_shift)


def _pack_normalized(bounds, seq, expr_shifts):
    _lock_time_resolution()
    bound_ticks = tuple(_time_to_tick(bound) for bound in bounds)
    seq = tuple(seq)
    expr_shifts = tuple(expr_shifts)
    if (len(bound_ticks) != len(seq) or not bound_ticks
            or bound_ticks[-1] != int(_INF_TICK)):
        raise ValueError("bounds and seq must have equal non-zero lengths ending at +inf")
    if len(expr_shifts) != len(seq):
        raise ValueError("expression shifts and seq must have equal lengths")
    if any(bound_ticks[i] >= bound_ticks[i + 1]
           for i in range(len(bound_ticks) - 1)):
        raise ValueError("bounds must be strictly increasing")

    expr_map = {}
    expr_blobs = []
    expr_ids = []
    shift_ticks = []
    for expr, expression_shift in zip(seq, expr_shifts):
        expr_id = expr_map.get(expr)
        if expr_id is None and expr not in expr_map:
            expr_id = len(expr_blobs)
            expr_map[expr] = expr_id
            expr_blobs.append(_encode_expr(expr))
        expr_ids.append(expr_id)
        if expr == _zero or expr[0] == (((), ()),):
            expression_shift = 0.0
        shift_ticks.append(_time_to_tick(expression_shift))

    out = bytearray(_HEADER.pack(_MAGIC, _VERSION, 0, len(bound_ticks),
                                 len(expr_blobs)))
    out.extend(np.asarray(bound_ticks, dtype="<i8").tobytes())
    out.extend(np.asarray(expr_ids, dtype="<u4").tobytes())
    out.extend(np.asarray(shift_ticks, dtype="<i8").tobytes())
    offsets = [0]
    for blob in expr_blobs:
        offsets.append(offsets[-1] + len(blob))
    out.extend(np.asarray(offsets, dtype="<u4").tobytes())
    for blob in expr_blobs:
        out.extend(blob)
    return bytes(out)


def _pack_block(bounds, seq):
    normalized = []
    shifts = []
    for expr in seq:
        expr, shift = _normalize_expression(expr)
        normalized.append(expr)
        shifts.append(shift)
    return _pack_normalized(bounds, normalized, shifts)


def _block_layout(bytes data):
    if len(data) < _HEADER.size:
        raise ValueError("truncated packed waveform")
    magic, version, flags, segment_count, expr_count = _HEADER.unpack_from(data)
    if magic != _MAGIC or version != _VERSION or flags != 0:
        raise ValueError("unsupported packed waveform")
    bounds_offset = _HEADER.size
    ids_offset = bounds_offset + 8 * segment_count
    shifts_offset = ids_offset + 4 * segment_count
    offsets_offset = shifts_offset + 8 * segment_count
    expr_data_offset = offsets_offset + 4 * (expr_count + 1)
    if expr_data_offset > len(data):
        raise ValueError("truncated packed waveform sections")
    offsets = np.frombuffer(data, dtype="<u4", count=expr_count + 1,
                            offset=offsets_offset)
    if offsets[0] != 0 or expr_data_offset + int(offsets[-1]) != len(data):
        raise ValueError("invalid packed waveform expression offsets")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("invalid packed waveform expression ordering")
    ids = np.frombuffer(data, dtype="<u4", count=segment_count,
                        offset=ids_offset)
    if len(ids) and int(ids.max()) >= expr_count:
        raise ValueError("invalid packed waveform expression id")
    return (segment_count, expr_count, bounds_offset, ids_offset,
            shifts_offset, offsets_offset, expr_data_offset)


cdef class PackedWaveform:
    cdef bytes _data
    cdef object _decoded
    cdef object _normalized

    def __cinit__(self, data=None):
        self._decoded = None
        self._normalized = None
        if data is not None:
            _lock_time_resolution()
            self._data = data if isinstance(data, bytes) else bytes(data)
            _block_layout(self._data)

    @classmethod
    def from_legacy(cls, bounds, seq):
        return cls(_pack_block(bounds, seq))

    @classmethod
    def from_bytes(cls, data):
        return cls(data)

    def to_bytes(self):
        return self._data

    def __bytes__(self):
        return self._data

    def __len__(self):
        return len(self._data)

    def __hash__(self):
        return hash(self._data)

    def __eq__(self, other):
        return isinstance(other, PackedWaveform) and self._data == other._data

    def __reduce__(self):
        return (type(self).from_bytes, (self._data,))

    def get_bounds(self):
        segment_count, _, bounds_offset, _, _, _, _ = _block_layout(self._data)
        ticks = np.frombuffer(self._data, dtype="<i8", count=segment_count,
                              offset=bounds_offset)
        view = ticks.astype(np.float64) * _TIME_RESOLUTION
        if len(view) and ticks[-1] == _INF_TICK:
            view[-1] = inf
        view.flags.writeable = False
        return view

    def get_bound_ticks(self):
        segment_count, _, bounds_offset, _, _, _, _ = _block_layout(self._data)
        view = np.frombuffer(self._data, dtype="<i8", count=segment_count,
                             offset=bounds_offset)
        view.flags.writeable = False
        return view

    cdef object _decode_normalized(self):
        if self._normalized is not None:
            return self._normalized
        segment_count, expr_count, bounds_offset, ids_offset, shifts_offset, offsets_offset, expr_data_offset = _block_layout(self._data)
        ticks = np.frombuffer(self._data, dtype="<i8", count=segment_count,
                              offset=bounds_offset)
        bounds = tuple(_tick_to_time(tick) for tick in ticks)
        ids = np.frombuffer(self._data, dtype="<u4", count=segment_count,
                            offset=ids_offset)
        shift_ticks = np.frombuffer(self._data, dtype="<i8", count=segment_count,
                                    offset=shifts_offset)
        shifts = tuple(_tick_to_time(tick) for tick in shift_ticks)
        offsets = np.frombuffer(self._data, dtype="<u4", count=expr_count + 1,
                                offset=offsets_offset)
        exprs = []
        for i in range(expr_count):
            start = expr_data_offset + int(offsets[i])
            stop = expr_data_offset + int(offsets[i + 1])
            exprs.append(_decode_expr(self._data[start:stop]))
        seq = tuple(exprs[int(i)] for i in ids)
        self._normalized = bounds, seq, shifts
        return self._normalized

    def to_legacy(self):
        if self._decoded is not None:
            return self._decoded
        bounds, seq, shifts = self._decode_normalized()
        seq = tuple(_expr_shift(expr, shift) if shift else expr
                    for expr, shift in zip(seq, shifts))
        self._decoded = bounds, seq
        return self._decoded

    def shifted(self, time):
        bounds, seq, shifts = self._decode_normalized()
        return PackedWaveform(_pack_normalized(
            tuple(round(bound + time, NDIGITS) for bound in bounds), seq,
            tuple(quantize_time(shift + time) for shift in shifts)))

    def scaled(self, value):
        bounds, seq, shifts = self._decode_normalized()
        constant = _const_expr(_real_number(value))
        scaled_seq = tuple(_expr_mul(expr, constant) for expr in seq)
        scaled_shifts = tuple(0 if expr == _zero else shift
                              for expr, shift in zip(scaled_seq, shifts))
        return PackedWaveform(_pack_normalized(bounds, scaled_seq,
                                               scaled_shifts))

    def add(self, PackedWaveform other):
        return _merge_blocks(self, other, False)

    def mul(self, PackedWaveform other):
        return _merge_blocks(self, other, True)

    def power(self, n):
        bounds, seq, shifts = self._decode_normalized()
        powered = []
        powered_shifts = []
        cache = {}
        for expr, shift in zip(seq, shifts):
            result = cache.get(expr)
            if result is None:
                result = _expr_pow(expr, n)
                cache[expr] = result
            result, shift = _normalize_expression(result, shift)
            powered.append(result)
            powered_shifts.append(shift)
        return PackedWaveform(_pack_normalized(bounds, powered,
                                               powered_shifts))

    def simplify(self, eps=1e-15):
        bounds, seq, shifts = self._decode_normalized()
        new_bounds = []
        new_seq = []
        new_shifts = []
        cache = {}
        for bound, expr, shift in zip(bounds, seq, shifts):
            simplified = cache.get(expr)
            if simplified is None:
                simplified = _simplify_expr(expr, eps)
                cache[expr] = simplified
            simplified, shift = _normalize_expression(simplified, shift)
            if (new_seq and simplified == new_seq[-1]
                    and shift == new_shifts[-1]):
                new_bounds[-1] = bound
            else:
                new_bounds.append(bound)
                new_seq.append(simplified)
                new_shifts.append(shift)
        return PackedWaveform(_pack_normalized(tuple(new_bounds), new_seq,
                                               new_shifts))

    def filtered(self, low=0, high=inf, eps=1e-15):
        bounds, seq, shifts = self._decode_normalized()
        filtered_seq = []
        filtered_shifts = []
        cache = {}
        for expr, shift in zip(seq, shifts):
            result = cache.get(expr)
            if result is None:
                result = _filter_expr(expr, low, high, eps)
                cache[expr] = result
            result, shift = _normalize_expression(result, shift)
            filtered_seq.append(result)
            filtered_shifts.append(shift)
        return PackedWaveform(_pack_normalized(bounds, filtered_seq,
                                               filtered_shifts))

    def derivative(self, order=1):
        if order < 0 or not isinstance(order, int):
            raise ValueError("order must be a non-negative integer")
        bounds, seq, shifts = self._decode_normalized()
        derived_seq = []
        derived_shifts = []
        cache = {}
        for expr, shift in zip(seq, shifts):
            result = cache.get(expr)
            if result is None:
                result = expr
                for _ in range(order):
                    result = _derivative_expr(result)
                cache[expr] = result
            result, shift = _normalize_expression(result, shift)
            derived_seq.append(result)
            derived_shifts.append(shift)
        return PackedWaveform(_pack_normalized(bounds, derived_seq,
                                               derived_shifts))

    def evaluate(self, x, lower=-inf, upper=inf):
        parts, _ = self.parts_shifted(x, 0.0, lower, upper)
        out = np.zeros_like(x, dtype=float)
        for start, stop, part in parts:
            out[start:stop] += part
        return out

    def parts(self, x, lower=-inf, upper=inf):
        return self.parts_shifted(x, 0.0, lower, upper)

    def parts_shifted(self, x, delay, lower=-inf, upper=inf):
        bounds, _, _ = self._decode_normalized()
        if delay:
            bounds = tuple(round(bound + delay, NDIGITS) for bound in bounds)
        return self.parts_with_bounds(x, bounds, delay, lower, upper)

    def parts_with_bounds(self, x, bounds, delay, lower=-inf, upper=inf):
        if np.iscomplexobj(x):
            raise TypeError("waveform sample positions must be real")
        _, seq, shifts = self._decode_normalized()
        ranges = np.searchsorted(x, bounds)
        parts = []
        start = 0
        should_clip = lower != -inf or upper != inf
        for i, stop in enumerate(ranges):
            stop = int(stop)
            if start < stop and seq[i] != _zero:
                total_shift = delay + shifts[i]
                values = _calc_expr(
                    seq[i], x[start:stop] if total_shift == 0
                    else x[start:stop] - total_shift)
                if should_clip:
                    values = np.clip(values, lower, upper)
                parts.append((start, stop, values))
            start = stop
        return parts, float

    def evaluate_shifted(self, x, delay, lower=-inf, upper=inf):
        parts, _ = self.parts_shifted(x, delay, lower, upper)
        out = np.zeros_like(x, dtype=float)
        for start, stop, part in parts:
            out[start:stop] += part
        return out


def _const_expr(value):
    value = _real_number(value)
    if value == 0:
        return _zero
    return ((((), ()),), (value,))


def _insert_pair(t_list, v_list, term, value, lo, hi):
    i = bisect_left(t_list, term, lo, hi)
    if i < hi and t_list[i] == term:
        value += v_list[i]
        if value == 0:
            t_list.pop(i)
            v_list.pop(i)
            return i, hi - 1
        v_list[i] = value
        return i, hi
    t_list.insert(i, term)
    v_list.insert(i, value)
    return i, hi + 1


def _expr_add(x, y):
    t_list, v_list = list(x[0]), list(x[1])
    lo, hi = 0, len(t_list)
    for term, value in zip(*y):
        lo, hi = _insert_pair(t_list, v_list, term, value, lo, hi)
    return tuple(t_list), tuple(v_list)


def _expr_mul(x, y):
    t_list, v_list = [], []
    lo = hi = 0
    for (t1, t2), (v1, v2) in zip(product(x[0], y[0]),
                                  product(x[1], y[1])):
        value = v1 * v2
        if value == 0:
            continue
        term = _expr_add(t1, t2)
        lo, hi = _insert_pair(t_list, v_list, term, value, lo, hi)
    return tuple(t_list), tuple(v_list)


def _expr_shift(expr, time):
    if expr == _zero or expr[0] == (((), ()),):
        return expr
    terms = []
    for functions, powers in expr[0]:
        terms.append((tuple((*function[:-1], function[-1] + time)
                            for function in functions), powers))
    return tuple(terms), expr[1]


def _expr_pow(expr, n):
    if expr == _zero:
        return _zero
    if n == 0:
        return _one
    if expr[0] == (((), ()),):
        return _const_expr(expr[1][0] ** n)
    if len(expr[0]) == 1:
        terms = []
        values = []
        for (functions, powers), value in zip(*expr):
            terms.append((functions, tuple(n * power for power in powers)))
            values.append(value ** n)
        return tuple(terms), tuple(values)
    if not isinstance(n, int) or n < 0:
        raise ValueError("non-constant waveform powers must be non-negative integers")
    result = _one
    for _ in range(n):
        result = _expr_mul(result, expr)
    return result


def _derivative_base(function):
    opcode, *args, shift_value = function
    if opcode == LINEAR:
        return _one
    if opcode == GAUSSIAN:
        return (((((LINEAR, shift_value), (GAUSSIAN, *args, shift_value)),
                   (1, 1)),), (-2 / args[0] ** 2,))
    if opcode == ERF:
        return (((((GAUSSIAN, *args, shift_value),), (1,)),),
                (2 / args[0] / np.sqrt(pi),))
    if opcode == COS:
        return (((((COS, args[0], shift_value - pi / args[0] / 2),),
                   (1,)),), (args[0],))
    if opcode == SINC:
        frequency = pi * args[0]
        return (((((LINEAR, shift_value),
                    (COS, frequency, shift_value)), (-1, 1)),
                 (((LINEAR, shift_value),
                    (COS, frequency, shift_value + pi / (2 * frequency))),
                  (-2, 1))),
                (1.0, -1 / frequency))
    if opcode == EXP:
        return (((((EXP, *args, shift_value),), (1,)),), (args[0],))
    if opcode == INTERP:
        start, stop, points = args
        gradient = tuple(np.gradient(np.asarray(points)))
        return (((((INTERP, start, stop, gradient, shift_value),), (1,)),),
                ((len(points) - 1) / (stop - start),))
    if opcode == COSH:
        return (((((SINH, *args, shift_value),), (1,)),), (args[0],))
    if opcode == SINH:
        return (((((COSH, *args, shift_value),), (1,)),), (args[0],))
    if opcode == LINEARCHIRP:
        f0, f1, duration, phi0 = args
        terms = (
            (((LINEARCHIRP, f0, f1, duration, phi0 + pi / 2, shift_value),),
             (1,)),
            (((LINEAR, shift_value),
              (LINEARCHIRP, f0, f1, duration, phi0 + pi / 2, shift_value)),
             (1, 1)),
        )
        values = (2 * pi * f0, 2 * pi * (f1 - f0) / duration)
        return (terms[1:], values[1:]) if f0 == 0 else (terms, values)
    if opcode == EXPONENTIALCHIRP:
        f0, alpha, phi0 = args
        return (((((EXP, alpha, shift_value),
                    (EXPONENTIALCHIRP, f0, alpha, phi0 + pi / 2,
                     shift_value)), (1, 1)),), (2 * pi * f0,))
    if opcode == HYPERBOLICCHIRP:
        f0, k, phi0 = args
        return (((((LINEAR, shift_value - 1 / k),
                    (HYPERBOLICCHIRP, f0, k, phi0 + pi / 2,
                     shift_value)), (-1, 1)),), (2 * pi * f0,))
    if opcode == MOLLIFIER:
        radius, derivative = args
        return (((((MOLLIFIER, radius, derivative + 1, shift_value),),
                   (1,)),), (1.0,))
    if opcode == D_GAUSSIAN:
        std_sq2, derivative = args
        return (((((D_GAUSSIAN, std_sq2, derivative + 1, shift_value),),
                   (1,)),), (1.0,))
    raise ValueError(f"derivative is not registered for opcode {opcode}")


def _derivative_expr(expr):
    if expr == _zero or expr[0] == (((), ()),):
        return _zero
    terms, values = expr
    result = _zero
    for (functions, powers), coefficient in zip(terms, values):
        for index, (function, power) in enumerate(zip(functions, powers)):
            remaining_functions = list(functions)
            remaining_powers = list(powers)
            if power == 1:
                remaining_functions.pop(index)
                remaining_powers.pop(index)
            else:
                remaining_powers[index] = power - 1
            remaining = (((tuple(remaining_functions),
                           tuple(remaining_powers)),),
                         (coefficient * power,))
            result = _expr_add(result,
                               _expr_mul(remaining,
                                         _derivative_base(function)))
    return result


def _cos_power_n(function, n):
    _, frequency, shift_value = function
    result = _zero
    for k in range(0, n // 2 + 1):
        if n == 2 * k:
            result = _expr_add(result, _const_expr(comb(n, k) / 2 ** n))
        else:
            expr = (((((COS, (n - 2 * k) * frequency, shift_value),),
                       (1,)),), (comb(n, k) / 2 ** (n - 1),))
            result = _expr_add(result, expr)
    return result


def _trig_mul_pair(left, right, value):
    _, w1, t1 = left
    _, w2, t2 = right
    if w2 > w1:
        t1, t2 = t2, t1
        w1, w2 = w2, w1
    sum_function = (COS, w1 + w2, (w1 * t1 + w2 * t2) / (w1 + w2))
    if w1 == w2:
        constant_value = value * np.cos(w1 * t1 - w2 * t2) / 2
        if constant_value == 0:
            return ((((sum_function,), (1,)),), (0.5 * value,))
        return ((((), ()), ((sum_function,), (1,))),
                (constant_value, 0.5 * value))
    difference_function = (COS, w1 - w2,
                           (w1 * t1 - w2 * t2) / (w1 - w2))
    if difference_function[1] > sum_function[1]:
        difference_function, sum_function = sum_function, difference_function
    return ((((difference_function,), (1,)), ((sum_function,), (1,))),
            (0.5 * value, 0.5 * value))


def _trig_mul(left, right):
    if left == _zero or right == _zero:
        return _zero
    if left[0] == (((), ()),) or right[0] == (((), ()),):
        return _expr_mul(left, right)
    result = _zero
    for (term1, term2), (value1, value2) in zip(
            product(left[0], right[0]), product(left[1], right[1])):
        value = value1 * value2
        non_trig = _one
        trig = []
        for function, power in zip(chain(term1[0], term2[0]),
                                   chain(term1[1], term2[1])):
            if function[0] == COS:
                trig.extend([function] * power)
            else:
                non_trig = _expr_mul(
                    non_trig, ((((function,), (power,)),), (1.0,)))
        if len(trig) == 1:
            expr = ((((trig[0],), (1,)),), (value,))
        elif len(trig) == 2:
            expr = _trig_mul_pair(trig[0], trig[1], value)
        else:
            expr = _const_expr(value)
            for function in trig:
                expr = _expr_mul(expr, ((((function,), (1,)),), (1.0,)))
        result = _expr_add(result, _expr_mul(non_trig, expr))
    return result


def _exp_trig_reduce(term, value):
    trig = _one
    alpha = 0
    exp_shift = 0
    functions = []
    powers = []
    for function, power in zip(*term):
        if function[0] == COS:
            trig = _trig_mul(trig, _cos_power_n(function, power))
        elif function[0] == EXP:
            new_weight = alpha * exp_shift + power * function[1] * function[-1]
            alpha += power * function[1]
            exp_shift = 0 if alpha == 0 else new_weight / alpha
        elif function[0] == GAUSSIAN and power != 1:
            functions.append((function[0], function[1] / np.sqrt(power),
                              function[2]))
            powers.append(1)
        else:
            functions.append(function)
            powers.append(power)
    result = (((tuple(functions), tuple(powers)),), (value,))
    if alpha != 0:
        result = _expr_mul(
            result, (((((EXP, alpha, exp_shift),), (1,)),), (1.0,)))
    return _expr_mul(result, trig)


def _frequency_parts(term):
    functions = []
    powers = []
    frequency = 0
    shift_value = 0
    for function, power in zip(*term):
        if function[0] == COS:
            if frequency != 0:
                raise ValueError("trigonometric expression was not reduced")
            frequency = function[1]
            shift_value = function[-1]
        else:
            functions.append(function)
            powers.append(power)
    return frequency, shift_value, (tuple(functions), tuple(powers))


def _simplify_expr(expr, eps):
    groups = {}
    for term, value in zip(*expr):
        for reduced_term, reduced_value in zip(*_exp_trig_reduce(term, value)):
            frequency, shift_value, base_term = _frequency_parts(reduced_term)
            value = _real_number(reduced_value)
            phase_shift = shift_value
            if (base_term, frequency) in groups:
                old_value, old_shift = groups[(base_term, frequency)]
                if frequency == 0:
                    value += old_value
                else:
                    a = (old_value * np.cos(frequency * old_shift)
                         + value * np.cos(frequency * phase_shift))
                    b = (old_value * np.sin(frequency * old_shift)
                         + value * np.sin(frequency * phase_shift))
                    phase_shift = np.arctan2(b, a) / frequency
                    value = np.hypot(a, b)
            groups[(base_term, frequency)] = value, phase_shift

    result = _zero
    for (base_term, frequency), (value, phase_shift) in groups.items():
        if frequency == 0:
            if abs(value) >= eps:
                result = _expr_add(result, ((base_term,), (value,)))
            continue
        if abs(value) >= eps:
            result = _expr_add(
                result,
                _expr_mul(((base_term,), (1.0,)),
                          (((((COS, frequency, phase_shift),), (1,)),),
                           (value,))))
    return result


def _filter_expr(expr, low, high, eps):
    expr = _simplify_expr(expr, eps)
    result = _zero
    for term, value in zip(*expr):
        for function, power in zip(*term):
            if function[0] == COS:
                if low <= function[1] < high:
                    result = _expr_add(result, ((term,), (value,)))
                break
        else:
            if low <= 0:
                result = _expr_add(result, ((term,), (value,)))
    return result


def _merge_blocks(PackedWaveform left, PackedWaveform right, bint multiply):
    b1, s1 = left.to_legacy()
    b2, s2 = right.to_legacy()
    bounds = []
    seq = []
    i = j = 0
    while i < len(b1) and j < len(b2):
        bound = min(b1[i], b2[j])
        expr = _expr_mul(s1[i], s2[j]) if multiply else _expr_add(s1[i], s2[j])
        if seq and expr == seq[-1]:
            bounds[-1] = bound
        else:
            bounds.append(bound)
            seq.append(expr)
        if bound == b1[i]:
            i += 1
        if bound == b2[j]:
            j += 1
    return PackedWaveform.from_legacy(tuple(bounds), tuple(seq))


def constant(value):
    return PackedWaveform.from_legacy((inf,), (_const_expr(value),))


def basic(opcode, args=(), shift=0.0):
    expr = (((((int(opcode), *tuple(args), float(shift)),), (1,)),), (1.0,))
    return PackedWaveform.from_legacy((inf,), (expr,))


def piecewise(bounds, expressions):
    seq = []
    for expression in expressions:
        if isinstance(expression, PackedWaveform):
            eb, es = expression.to_legacy()
            if len(eb) != 1:
                raise ValueError("piecewise expressions must be scalar cores")
            seq.append(es[0])
        elif isinstance(expression, (int, float, complex, np.number)):
            seq.append(_const_expr(expression))
        else:
            seq.append(expression)
    return PackedWaveform.from_legacy(bounds, tuple(seq))


def sum_cores(cores):
    cores = list(cores)
    if not cores:
        return constant(0)
    accumulator = {}
    events = {}
    for core in cores:
        bounds, seq = core.to_legacy()
        for term, value in zip(*seq[0]):
            accumulator[term] = accumulator.get(term, 0) + value
            if accumulator[term] == 0:
                del accumulator[term]
        for i, boundary in enumerate(bounds[:-1]):
            events.setdefault(boundary, []).append((seq[i], seq[i + 1]))

    def snapshot():
        if not accumulator:
            return _zero
        items = sorted(accumulator.items())
        return tuple(item[0] for item in items), tuple(item[1] for item in items)

    output_bounds = []
    output_seq = [snapshot()]
    for boundary in sorted(events):
        for old, new in events[boundary]:
            for term, value in zip(*old):
                accumulator[term] = accumulator.get(term, 0) - value
                if accumulator[term] == 0:
                    del accumulator[term]
            for term, value in zip(*new):
                accumulator[term] = accumulator.get(term, 0) + value
                if accumulator[term] == 0:
                    del accumulator[term]
        expr = snapshot()
        if expr != output_seq[-1]:
            output_bounds.append(boundary)
            output_seq.append(expr)
    output_bounds.append(inf)
    return PackedWaveform.from_legacy(tuple(output_bounds), tuple(output_seq))


def _sum_events(events, global_shift=0, offset=0):
    cdef PackedWaveform packed_core
    accumulator = {}
    boundary_events = {}
    for core, delay, scale in events:
        packed_core = core
        bounds, seq, shifts = packed_core._decode_normalized()
        delay = quantize_time(delay + global_shift)
        first_key = (seq[0], quantize_time(shifts[0] + delay))
        if seq[0] != _zero:
            accumulator[first_key] = accumulator.get(first_key, 0) + scale
            if accumulator[first_key] == 0:
                del accumulator[first_key]
        for i, boundary in enumerate(bounds[:-1]):
            boundary = quantize_time(boundary + delay)
            old_key = (seq[i], quantize_time(shifts[i] + delay))
            new_key = (seq[i + 1], quantize_time(shifts[i + 1] + delay))
            boundary_events.setdefault(boundary, []).append(
                (old_key, new_key, scale))

    if offset != 0:
        accumulator[(_one, 0.0)] = offset

    def snapshot():
        if not accumulator:
            return _zero, 0.0
        if len(accumulator) == 1:
            (expr, shift), scale = next(iter(accumulator.items()))
            if scale == 1:
                return expr, shift
            result = _expr_mul(expr, _const_expr(scale))
            return _normalize_expression(result, shift)
        origin = 0.0
        for (expr, shift), scale in accumulator.items():
            if scale != 0 and expr != _zero and expr[0] != (((), ()),):
                origin = shift
                break
        result = _zero
        for (expr, shift), scale in accumulator.items():
            if expr == _zero or scale == 0:
                continue
            term = expr if scale == 1 else _expr_mul(expr, _const_expr(scale))
            if shift != origin:
                term = _expr_shift(term, shift - origin)
            result = _expr_add(result, term)
        return _normalize_expression(result, origin)

    output_bounds = []
    first_expr, first_shift = snapshot()
    output_seq = [first_expr]
    output_shifts = [first_shift]
    for boundary in sorted(boundary_events):
        for old_key, new_key, scale in boundary_events[boundary]:
            if old_key[0] != _zero:
                accumulator[old_key] = accumulator.get(old_key, 0) - scale
                if accumulator[old_key] == 0:
                    del accumulator[old_key]
            if new_key[0] != _zero:
                accumulator[new_key] = accumulator.get(new_key, 0) + scale
                if accumulator[new_key] == 0:
                    del accumulator[new_key]
        expr, shift = snapshot()
        if expr != output_seq[-1] or shift != output_shifts[-1]:
            output_bounds.append(boundary)
            output_seq.append(expr)
            output_shifts.append(shift)
    output_bounds.append(inf)
    return PackedWaveform(_pack_normalized(tuple(output_bounds), output_seq,
                                           output_shifts))


def _pack_stack(events):
    _lock_time_resolution()
    templates = []
    template_map = {}
    ids = []
    delays = []
    scales = []
    for core, delay, scale in events:
        data = core.to_bytes()
        template_id = template_map.get(data)
        if template_id is None:
            template_id = len(templates)
            template_map[data] = template_id
            templates.append(data)
        ids.append(template_id)
        delays.append(_time_to_tick(delay))
        scales.append(_real_number(scale))

    offsets_size = 4 * (len(templates) + 1)
    template_start = _STACK_HEADER.size + offsets_size
    offsets = [template_start]
    for data in templates:
        offsets.append(offsets[-1] + len(data))
    event_offset = offsets[-1]
    out = bytearray(_STACK_HEADER.pack(_STACK_MAGIC, _VERSION, 0,
                                       len(templates), len(ids), event_offset))
    out.extend(np.asarray(offsets, dtype="<u4").tobytes())
    for data in templates:
        out.extend(data)
    out.extend(np.asarray(ids, dtype="<u4").tobytes())
    out.extend(np.asarray(delays, dtype="<i8").tobytes())
    out.extend(np.asarray(scales, dtype="<f8").tobytes())
    return bytes(out)


def _stack_layout(bytes data):
    if len(data) < _STACK_HEADER.size:
        raise ValueError("truncated packed waveform stack")
    magic, version, flags, template_count, event_count, event_offset = _STACK_HEADER.unpack_from(data)
    if magic != _STACK_MAGIC or version != _VERSION or flags != 0:
        raise ValueError("unsupported packed waveform stack")
    offsets_end = _STACK_HEADER.size + 4 * (template_count + 1)
    event_end = event_offset + 20 * event_count
    if offsets_end > len(data) or event_end != len(data):
        raise ValueError("invalid packed waveform stack layout")
    offsets = np.frombuffer(data, dtype="<u4", count=template_count + 1,
                            offset=_STACK_HEADER.size)
    if offsets[0] != offsets_end or offsets[-1] != event_offset or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("invalid packed waveform template offsets")
    return template_count, event_count, event_offset, offsets


cdef class PackedStack:
    cdef bytes _data
    cdef object _events
    cdef object _compiled

    def __cinit__(self, data=None):
        self._events = None
        self._compiled = None
        if data is None:
            self._data = _pack_stack(())
        elif isinstance(data, bytes):
            _lock_time_resolution()
            self._data = data
            _stack_layout(data)
        else:
            self._data = _pack_stack(data)

    @classmethod
    def from_bytes(cls, data):
        return cls(data if isinstance(data, bytes) else bytes(data))

    @classmethod
    def from_events(cls, events):
        return cls(events)

    def to_bytes(self):
        return self._data

    def __bytes__(self):
        return self._data

    def __len__(self):
        return len(self._data)

    def __hash__(self):
        return hash(self._data)

    def __eq__(self, other):
        return isinstance(other, PackedStack) and self._data == other._data

    def __reduce__(self):
        return (type(self).from_bytes, (self._data,))

    def events(self):
        if self._events is not None:
            return self._events
        template_count, event_count, event_offset, offsets = _stack_layout(self._data)
        templates = []
        for i in range(template_count):
            templates.append(PackedWaveform(self._data[int(offsets[i]):int(offsets[i + 1])]))
        ids = np.frombuffer(self._data, dtype="<u4", count=event_count,
                            offset=event_offset)
        delay_offset = event_offset + 4 * event_count
        delays = np.frombuffer(self._data, dtype="<i8", count=event_count,
                               offset=delay_offset)
        scale_offset = delay_offset + 8 * event_count
        scales = np.frombuffer(self._data, dtype="<f8", count=event_count,
                               offset=scale_offset)
        self._events = tuple((templates[int(ids[i])], _tick_to_time(delays[i]),
                              float(scales[i])) for i in range(event_count))
        return self._events

    def compiled_events(self):
        if self._compiled is None:
            compiled = []
            for core, delay, scale in self.events():
                bounds = np.around(core.get_bounds() + delay, NDIGITS)
                bounds.flags.writeable = False
                compiled.append((core, bounds, delay, scale))
            self._compiled = tuple(compiled)
        return self._compiled

    def evaluate(self, x, offset=0, shift=0):
        if np.iscomplexobj(x):
            raise TypeError("waveform sample positions must be real")
        out = np.full_like(x, _real_number(offset), dtype=np.float64)
        for core, bounds, delay, scale in self.compiled_events():
            if shift:
                shifted_bounds = np.around(bounds + shift, NDIGITS)
            else:
                shifted_bounds = bounds
            parts, _ = core.parts_with_bounds(x, shifted_bounds,
                                               shift + delay)
            for start, stop, values in parts:
                out[start:stop] += scale * values
        return out

    def simplified(self, shift=0, offset=0, eps=1e-15):
        return _sum_events(self.events(), shift, offset).simplify(eps)

    def combined(self, PackedStack other):
        return PackedStack.from_events((*self.events(), *other.events()))

    def scaled(self, value):
        value = _real_number(value)
        return PackedStack.from_events((core, delay, scale * value)
                                       for core, delay, scale in self.events())


def registerBaseFunc(*args, **kwargs):
    raise NotImplementedError("custom waveform base functions are not supported")


def registerDerivative(*args, **kwargs):
    raise NotImplementedError("custom waveform derivatives are not supported")
