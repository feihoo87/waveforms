"""Independent packed-binary waveform implementation.

``Waveform`` and ``WaveVStack`` keep their signal representation in immutable
binary blocks owned by :mod:`waveforms._waveform`. Times in those blocks are
signed 64-bit ticks. The tick duration is process-wide configuration and is
intentionally not repeated in every block.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, cast

import numpy as np
from numpy import e, inf, pi
from scipy.signal import sosfilt

from ._waveform import (
    COS, COSH, D_GAUSSIAN, DRAG, ERF, EXP, EXPONENTIALCHIRP, GAUSSIAN,
    HYPERBOLICCHIRP, INTERP, LINEAR, LINEARCHIRP, MOLLIFIER, SINC, SINH,
    PackedStack, PackedWaveform, basic, constant, get_time_resolution,
    piecewise, quantize_time, registerBaseFunc, registerDerivative,
    set_time_resolution,
)

_ZERO_EXPR = ((), ())
_ONE_EXPR = ((((), ()),), (1.0,))


def _copy_sampling_metadata(source, target):
    target.start = source.start
    target.stop = source.stop
    target.sample_rate = source.sample_rate
    target.filters = source.filters
    target.label = source.label
    return target


def _scalar(opcode, *args, shift=0.0):
    return basic(opcode, args, shift)


def _piecewise(bounds, *expressions):
    return Waveform._from_core(piecewise(tuple(bounds), expressions))


class _SamplingMixin:
    start: float | None
    stop: float | None
    sample_rate: float | None
    filters: tuple[np.ndarray, float] | None

    def sample(self, sample_rate=None, out: np.ndarray | None = None,
               chunk_size=None, function_lib=None,
               filters: tuple[np.ndarray, float] | None = None):
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
            return self._sample_iter(sample_rate, int(chunk_size), out, filters)

        x = np.arange(self.start, self.stop, 1 / sample_rate)
        sig = cast(np.ndarray, self(x, out=out))
        if filters is not None:
            sos, initial = filters
            sos = np.asarray(sos)
            if not sos.flags.writeable:
                sos = sos.copy()
            sig = sosfilt(sos, sig - initial) + initial if initial else sosfilt(sos, sig)
        return cast(np.ndarray, sig)

    def _sample_iter(self, sample_rate, chunk_size, out, filters):
        start = cast(float, self.start)
        stop_limit = cast(float, self.stop)
        output_index = 0
        zi = None
        initial = 0
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if filters is not None:
            sos, initial = filters
            sos = np.asarray(sos)
            if not sos.flags.writeable:
                sos = sos.copy()
            zi = np.zeros((sos.shape[0], 2))

        while start < stop_limit:
            size = min(chunk_size, round((stop_limit - start) * sample_rate))
            if size <= 0:
                break
            stop = start + size / sample_rate
            x = np.linspace(start, stop, size, endpoint=False)
            sig = cast(np.ndarray, self(x))
            if filters is not None:
                if initial:
                    sig = sig - initial
                sig, zi = sosfilt(sos, sig, zi=zi)
                if initial:
                    sig = sig + initial
            if out is not None:
                out[output_index:output_index + size] = sig
                yield out[output_index:output_index + size]
            else:
                yield sig
            start = stop
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
        self._scale = complex(_scale) if isinstance(_scale, complex) else _scale
        self.max = max
        self.min = min
        self.start = None
        self.stop = None
        self.sample_rate = None
        self.filters = None
        self.label = None

    @classmethod
    def _from_core(cls, core, delay=0.0, scale=1.0):
        return cls(_core=core, _delay=delay, _scale=scale)

    def _materialized_core(self):
        core = self._core
        if self._scale != 1:
            core = core.scaled(self._scale)
        if self._delay != 0:
            core = core.shifted(self._delay)
        return core

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
        return cls._from_core(PackedWaveform.from_bytes(data))

    def simplify(self, eps=1e-15):
        return Waveform._from_core(self._materialized_core().simplify(eps))

    def filter(self, low=0, high=inf, eps=1e-15):
        return Waveform._from_core(self._materialized_core().filtered(low, high, eps))

    def __pow__(self, n):
        return Waveform._from_core(self._materialized_core().power(n))

    def __add__(self, other):
        if not isinstance(other, Waveform):
            other = const(other)
        return Waveform._from_core(
            self._materialized_core().add(other._materialized_core())
        )

    def __radd__(self, value):
        return self + value

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, value):
        return value + (-self)

    def __mul__(self, other):
        if isinstance(other, Waveform):
            return Waveform._from_core(
                self._materialized_core().mul(other._materialized_core())
            )
        return Waveform._from_core(self._core, self._delay, self._scale * other)

    def __rmul__(self, value):
        return self * value

    def __truediv__(self, other):
        if isinstance(other, (Waveform, WaveVStack)):
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
        complex_output = False
        should_scale = self._scale != 1
        should_clip = self.min != -inf or self.max != inf
        for start, stop, part in parts:
            if should_scale:
                part = part * self._scale
            if should_clip:
                part = np.clip(part, self.min, self.max)
            complex_output |= np.iscomplexobj(part)
            scaled_parts.append((start, stop, part))
        if frag:
            if out is None:
                return scaled_parts
            if not accumulate:
                out.clear()
            out.extend(scaled_parts)
            return out

        if out is None:
            out = np.zeros_like(values, dtype=complex if complex_output else float)
        elif not accumulate:
            out[...] = 0
        for start, stop, part in scaled_parts:
            out[start:stop] += part
        return out[0] if scalar else out

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, (int, float, complex, np.number)):
            other = const(other)
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


class WaveVStack(_SamplingMixin):
    __slots__ = (
        "_stack", "start", "stop", "sample_rate", "offset", "shift",
        "filters", "label", "function_lib",
    )

    def __init__(self, wlist=()):
        if isinstance(wlist, WaveVStack):
            self._stack = wlist._stack
        else:
            events = []
            for wav in wlist:
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
            result = self._from_stack(self._stack)
            result.offset = self.offset + other
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
        if not isinstance(other, Waveform):
            result = self._from_stack(self._stack.scaled(other))
            result.offset = self.offset * other
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
        if isinstance(other, (Waveform, WaveVStack)):
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


def zero():
    return Waveform._from_core(constant(0))


def one():
    return Waveform._from_core(constant(1.0))


def const(value):
    return Waveform._from_core(constant(value))


def D(wav: Waveform, d: int = 1):
    if not isinstance(wav, Waveform):
        raise TypeError("D expects a Waveform")
    if d < 0 or not isinstance(d, int):
        raise ValueError("d must be a non-negative integer")
    return Waveform._from_core(wav._materialized_core().derivative(d))


def sign():
    return _piecewise((0, inf), -1, 1)


def step(edge, type="erf"):
    if edge == 0:
        return _piecewise((0, inf), 0, 1)
    if type == "cos":
        rise = constant(0.5).add(
            _scalar(COS, pi / edge, shift=0.5 * edge).scaled(0.5)
        )
        return _piecewise((-edge / 2, edge / 2, inf), 0, rise, 1)
    if type == "linear":
        rise = constant(0.5).add(_scalar(LINEAR).scaled(1 / edge))
        return _piecewise((-edge / 2, edge / 2, inf), 0, rise, 1)
    rise = constant(0.5).add(_scalar(ERF, edge / 5).scaled(0.5))
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
        return _scalar(opcode, *args, shift=shift)

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
    return _piecewise((-width / 2, width / 2, inf), 0, _scalar(SINC, bw), 0)


def cosPulse(width, plateau=0.0):
    if quantize_time(plateau / 2) > 0:
        return square(plateau + width / 2, edge=width / 2, type="cos")
    if width <= 0:
        return zero()
    pulse = constant(0.5).add(
        _scalar(COS, 2 * pi / width).scaled(0.5)
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
        return constant(amplitude / (amplitude - 1)).add(
            _scalar(COSH, w, shift=shift).scaled(scale)
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
                          _scalar(MOLLIFIER, width / 2, d), 0)
    return _piecewise(
        (-width / 2 - plateau / 2, -plateau / 2, plateau / 2,
         width / 2 + plateau / 2, inf),
        0, _scalar(MOLLIFIER, width / 2, d, shift=-plateau / 2), 1,
        _scalar(MOLLIFIER, width / 2, d, shift=plateau / 2), 0,
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
            _scalar(DRAG, t0, freq, width, delta, block_freq, phase), 0,
        )
    if width <= 0:
        w = 2 * pi * (freq + delta)
        carrier = _scalar(COS, w, shift=(phase + 2 * pi * delta * t0) / w)
        return _piecewise((t0, t0 + plateau, inf), 0, carrier, 0)
    w = 2 * pi * (freq + delta)
    carrier = _scalar(COS, w, shift=(phase + 2 * pi * delta * t0) / w)
    return _piecewise(
        (t0, t0 + width / 2, t0 + width / 2 + plateau,
         t0 + width + plateau, inf),
        0, _scalar(DRAG, t0, freq, width, delta, block_freq, phase),
        carrier,
        _scalar(DRAG, t0 + plateau, freq, width, delta, block_freq,
                phase - 2 * pi * delta * plateau), 0,
    )


def chirp(f0, f1, T, phi0=0, type="linear"):
    if f0 == f1:
        return sin(f0, phi0)
    if T <= 0:
        raise ValueError("T must be positive")
    if type == "linear":
        core = _scalar(LINEARCHIRP, f0, f1, T, phi0)
    elif type in ("exp", "exponential", "geometric"):
        if f0 == 0:
            raise ValueError("f0 must be non-zero")
        core = _scalar(EXPONENTIALCHIRP, f0, np.log(f1 / f0) / T, phi0)
    elif type in ("hyperbolic", "hyp"):
        if f0 * f1 == 0:
            return const(np.sin(phi0))
        core = _scalar(HYPERBOLICCHIRP, f0, (f0 - f1) / (f1 * T), phi0)
    else:
        raise ValueError(f"unknown type {type}")
    return _piecewise((0, T, inf), 0, core, 0)


def interp(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) == 0:
        raise ValueError("x and y must be non-empty one-dimensional arrays of equal size")
    bounds = [x[0]]
    expressions = [0]
    for x1, x2, y1, y2 in zip(x[:-1], x[1:], y[:-1], y[1:]):
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        expressions.append(
            _scalar(LINEAR, shift=x1).scaled(slope).add(constant(y1))
        )
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
    core = _scalar(INTERP, start, stop, tuple(points))
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
def wave_eval(expr: str) -> Waveform:
    """Parse an expression directly against the packed waveform backend."""
    import sys

    from .waveform_parser import WaveformParseError, parse_waveform_expression

    try:
        return parse_waveform_expression(expr, backend=sys.modules[__name__],
                                         extra_modules=())
    except WaveformParseError as exc:
        raise SyntaxError(f"Failed to parse expression {expr!r}: {exc}") from exc


__all__ = [
    "D", "Waveform", "WaveVStack", "chirp", "const", "cos", "cosh",
    "coshPulse", "cosPulse", "cut", "drag", "exp", "function",
    "gaussian", "general_cosine", "get_time_resolution", "hanning",
    "interp", "mixing", "mollifier", "one", "play", "poly",
    "registerBaseFunc", "registerDerivative", "samplingPoints",
    "set_time_resolution", "sign", "sin", "sinc", "sinh", "slepian",
    "square", "step", "t", "wave_eval", "zero", "e", "inf", "pi",
]
