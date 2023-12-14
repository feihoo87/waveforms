from fractions import Fraction

import numpy as np
from numpy import e, inf, pi
from scipy.signal import sosfilt

from ._waveform import (_D, COS, COSH, DRAG, ERF, EXP, EXPONENTIALCHIRP,
                        GAUSSIAN, HYPERBOLICCHIRP, INTERP, LINEAR, LINEARCHIRP,
                        NDIGITS, SINC, SINH, _baseFunc, _baseFunc_latex,
                        _const, _half, _one, _zero, add, basic_wave,
                        calc_parts, filter, is_const, merge_waveform, mul, pow,
                        registerBaseFunc, registerBaseFuncLatex,
                        registerDerivative, shift, simplify, wave_sum)


def _test_spec_num(num, spec):
    x = Fraction(num / spec).limit_denominator(1000000000)
    if x.denominator <= 24:
        return True, x, 1
    x = Fraction(spec * num).limit_denominator(1000000000)
    if x.denominator <= 24:
        return True, x, -1
    return False, x, 0


def _spec_num_latex(num):
    for spec, spec_latex in [(1, ''), (np.sqrt(2), '\\sqrt{2}'),
                             (np.sqrt(3), '\\sqrt{3}'),
                             (np.sqrt(5), '\\sqrt{5}'),
                             (np.log(2), '\\log{2}'), (np.log(3), '\\log{3}'),
                             (np.log(5), '\\log{5}'), (np.e, 'e'),
                             (np.pi, '\\pi'), (np.pi**2, '\\pi^2'),
                             (np.sqrt(np.pi), '\\sqrt{\\pi}')]:
        flag, x, sign = _test_spec_num(num, spec)
        if flag:
            if sign < 0:
                spec_latex = f"\\frac{{{1}}}{{{spec_latex}}}"
            if x.denominator == 1:
                if x.numerator == 1:
                    return f"{spec_latex}"
                else:
                    return f"{x.numerator:g}{spec_latex}"
            else:
                if x.numerator < 0:
                    return f"-\\frac{{{-x.numerator}}}{{{x.denominator}}}{spec_latex}"
                else:
                    return f"\\frac{{{x.numerator}}}{{{x.denominator}}}{spec_latex}"
    return f"{num:g}"


def _num_latex(num):
    if num == -np.inf:
        return r"-\infty"
    elif num == np.inf:
        return r"\infty"
    if num.imag > 0:
        return f"\\left({_num_latex(num.real)}+{_num_latex(num.imag)}j\\right)"
    elif num.imag < 0:
        return f"\\left({_num_latex(num.real)}-{_num_latex(-num.imag)}j\\right)"
    s = _spec_num_latex(num.real)
    if s == '' and round(num.real) == 1:
        return '1'
    if "e" in s:
        a, n = s.split("e")
        n = float(n)
        s = f"{a} \\times 10^{{{n:g}}}"
    return s


def _fun_latex(fun):
    funID, *args, shift = fun
    if _baseFunc_latex[funID] is None:
        shift = _num_latex(shift)
        if shift == "0":
            shift = ""
        elif shift[0] != '-':
            shift = "+" + shift
        return r"\mathrm{Func}" + f"{funID}(t{shift}, ...)"
    return _baseFunc_latex[funID](shift, *args)


def _wav_latex(wav):

    if wav == _zero:
        return "0"
    elif is_const(wav):
        return f"{wav[1][0]}"

    sum_expr = []
    for mul, amp in zip(*wav):
        if mul == ((), ()):
            sum_expr.append(_num_latex(amp))
            continue
        mul_expr = []
        amp = _num_latex(amp)
        if amp != "1":
            mul_expr.append(amp)
        for fun, n in zip(*mul):
            fun_expr = _fun_latex(fun)
            if n != 1:
                mul_expr.append(fun_expr + "^{" + f"{n}" + "}")
            else:
                mul_expr.append(fun_expr)
        sum_expr.append(''.join(mul_expr))

    ret = sum_expr[0]
    for expr in sum_expr[1:]:
        if expr[0] == '-':
            ret += expr
        else:
            ret += "+" + expr
    return ret


class Waveform:
    __slots__ = ('bounds', 'seq', 'max', 'min', 'start', 'stop', 'sample_rate',
                 'filters', 'label')

    def __init__(self, bounds=(+inf, ), seq=(_zero, ), min=-inf, max=inf):
        self.bounds = bounds
        self.seq = seq
        self.max = max
        self.min = min
        self.start = None
        self.stop = None
        self.sample_rate = None
        self.filters = None
        self.label = None

    def _begin(self):
        for i, s in enumerate(self.seq):
            if s is not _zero:
                if i == 0:
                    return -inf
                return self.bounds[i - 1]
        return inf

    def _end(self):
        N = len(self.bounds)
        for i, s in enumerate(self.seq[::-1]):
            if s is not _zero:
                if i == 0:
                    return inf
                return self.bounds[N - i - 1]
        return -inf

    @property
    def begin(self):
        if self.start is None:
            return self._begin()
        else:
            return max(self.start, self._begin())

    @property
    def end(self):
        if self.stop is None:
            return self._end()
        else:
            return min(self.stop, self._end())

    def sample(self,
               sample_rate=None,
               out=None,
               chunk_size=None,
               function_lib=None,
               filters=None):
        if sample_rate is None:
            sample_rate = self.sample_rate
        if self.start is None or self.stop is None or sample_rate is None:
            raise ValueError(
                f'Waveform is not initialized. {self.start=}, {self.stop=}, {sample_rate=}'
            )
        if filters is None:
            filters = self.filters
        if chunk_size is None:
            x = np.arange(self.start, self.stop, 1 / sample_rate)
            sig = self.__call__(x, out=out, function_lib=function_lib)
            if filters is not None:
                sos, initial = filters
                if initial:
                    sig = sosfilt(sos, sig - initial) + initial
                else:
                    sig = sosfilt(sos, sig)
            return sig
        else:
            return self._sample_iter(sample_rate, chunk_size, out,
                                     function_lib, filters)

    def _sample_iter(self, sample_rate, chunk_size, out, function_lib,
                     filters):
        start = self.start
        start_n = 0
        if filters is not None:
            sos, initial = filters
            # zi = sosfilt_zi(sos)
            zi = np.zeros((sos.shape[0], 2))
        length = chunk_size / sample_rate
        while start < self.stop:
            if start + length > self.stop:
                length = self.stop - start
                stop = self.stop
                size = round((stop - start) * sample_rate)
            else:
                stop = start + length
                size = chunk_size
            x = np.linspace(start, stop, size, endpoint=False)

            if filters is None:
                if out is not None:
                    yield self.__call__(x,
                                        out=out[start_n:],
                                        function_lib=function_lib)
                else:
                    yield self.__call__(x, function_lib=function_lib)
            else:
                sig = self.__call__(x, function_lib=function_lib)
                if initial:
                    sig -= initial
                sig, zi = sosfilt(sos, sig, zi=zi)
                if initial:
                    sig += initial
                if out is not None:
                    out[start_n:start_n + size] = sig
                yield sig

            start = stop
            start_n += chunk_size

    @staticmethod
    def _tolist(bounds, seq, ret=None):
        if ret is None:
            ret = []
        ret.append(len(bounds))
        for seq, b in zip(seq, bounds):
            ret.append(b)
            tlist, amplist = seq
            ret.append(len(amplist))
            for t, amp in zip(tlist, amplist):
                ret.append(amp)
                mtlist, nlist = t
                ret.append(len(nlist))
                for fun, n in zip(mtlist, nlist):
                    ret.append(n)
                    ret.append(len(fun))
                    ret.extend(fun)
        return ret

    @staticmethod
    def _fromlist(l, pos=0):

        def _read(l, pos, size):
            try:
                return tuple(l[pos:pos + size]), pos + size
            except:
                raise ValueError('Invalid waveform format')

        (nseg, ), pos = _read(l, pos, 1)
        bounds = []
        seq = []
        for _ in range(nseg):
            (b, nsum), pos = _read(l, pos, 2)
            bounds.append(b)
            amp = []
            t = []
            for _ in range(nsum):
                (a, nmul), pos = _read(l, pos, 2)
                amp.append(a)
                nlst = []
                mt = []
                for _ in range(nmul):
                    (n, nfun), pos = _read(l, pos, 2)
                    nlst.append(n)
                    fun, pos = _read(l, pos, nfun)
                    mt.append(fun)
                t.append((tuple(mt), tuple(nlst)))
            seq.append((tuple(t), tuple(amp)))

        return tuple(bounds), tuple(seq), pos

    def tolist(self):
        l = [self.max, self.min, self.start, self.stop, self.sample_rate]
        if self.filters is None:
            l.append(None)
        else:
            sos, initial = self.filters
            sos = list(sos.reshape(-1))
            l.append(len(sos))
            l.extend(sos)
            l.append(initial)

        return self._tolist(self.bounds, self.seq, l)

    @classmethod
    def fromlist(cls, l):
        w = cls()
        pos = 6
        (w.max, w.min, w.start, w.stop, w.sample_rate, sos_size) = l[:pos]
        if sos_size is not None:
            sos = np.array(l[pos:pos + sos_size]).reshape(-1, 6)
            pos += sos_size
            initial = l[pos]
            pos += 1
            w.filters = sos, initial

        w.bounds, w.seq, pos = cls._fromlist(l, pos)
        return w

    def totree(self):
        if self.filters is None:
            header = (self.max, self.min, self.start, self.stop,
                      self.sample_rate, None)
        else:
            header = (self.max, self.min, self.start, self.stop,
                      self.sample_rate, self.filters)
        body = []

        for seq, b in zip(self.seq, self.bounds):
            tlist, amplist = seq
            new_seq = []
            for t, amp in zip(tlist, amplist):
                mtlist, nlist = t
                new_t = []
                for fun, n in zip(mtlist, nlist):
                    new_t.append((n, fun))
                new_seq.append((amp, tuple(new_t)))
            body.append((b, tuple(new_seq)))
        return header, tuple(body)

    @staticmethod
    def fromtree(tree):
        w = Waveform()
        header, body = tree

        (w.max, w.min, w.start, w.stop, w.sample_rate, w.filters) = header
        bounds = []
        seqs = []
        for b, seq in body:
            bounds.append(b)
            amp_list = []
            t_list = []
            for amp, t in seq:
                amp_list.append(amp)
                n_list = []
                mt_list = []
                for n, mt in t:
                    n_list.append(n)
                    mt_list.append(mt)
                t_list.append((tuple(mt_list), tuple(n_list)))
            seqs.append((tuple(t_list), tuple(amp_list)))
        w.bounds = tuple(bounds)
        w.seq = tuple(seqs)
        return w

    def simplify(self, eps=1e-15):
        seq = [simplify(self.seq[0], eps)]
        bounds = [self.bounds[0]]
        for expr, b in zip(self.seq[1:], self.bounds[1:]):
            expr = simplify(expr, eps)
            if expr == seq[-1]:
                seq.pop()
                bounds.pop()
            seq.append(expr)
            bounds.append(b)
        return Waveform(tuple(bounds), tuple(seq))

    def filter(self, low=0, high=inf, eps=1e-15):
        seq = []
        for expr in self.seq:
            seq.append(filter(expr, low, high, eps))
        return Waveform(self.bounds, tuple(seq))

    def _comb(self, other, oper):
        return Waveform(*merge_waveform(self.bounds, self.seq, other.bounds,
                                        other.seq, oper))

    def __pow__(self, n):
        return Waveform(self.bounds, tuple(pow(w, n) for w in self.seq))

    def __add__(self, other):
        if isinstance(other, Waveform):
            return self._comb(other, add)
        else:
            return self + const(other)

    def __radd__(self, v):
        return const(v) + self

    def __ior__(self, other):
        return self | other

    def __or__(self, other):
        if isinstance(other, (int, float, complex)):
            other = const(other)
        w = self.marker + other.marker

        def _or(a, b):
            if a != _zero or b != _zero:
                return _one
            else:
                return _zero

        return self._comb(other, _or)

    def __iand__(self, other):
        return self & other

    def __and__(self, other):
        if isinstance(other, (int, float, complex)):
            other = const(other)
        w = self.marker + other.marker

        def _and(a, b):
            if a != _zero and b != _zero:
                return _one
            else:
                return _zero

        return self._comb(other, _and)

    @property
    def marker(self):
        w = self.simplify()
        return Waveform(w.bounds,
                        tuple(_zero if s == _zero else _one for s in w.seq))

    def mask(self, edge=0):
        w = self.marker
        in_wave = w.seq[0] == _zero
        bounds = []
        seq = []

        if w.seq[0] == _zero:
            in_wave = False
            b = w.bounds[0] - edge
            bounds.append(b)
            seq.append(_zero)

        for b, s in zip(w.bounds[1:], w.seq[1:]):
            if not in_wave and s != _zero:
                in_wave = True
                bounds.append(b + edge)
                seq.append(_one)
            elif in_wave and s == _zero:
                in_wave = False
                b = b - edge
                if b > bounds[-1]:
                    bounds.append(b)
                    seq.append(_zero)
                else:
                    bounds.pop()
                    bounds.append(b)
        return Waveform(tuple(bounds), tuple(seq))

    def __mul__(self, other):
        if isinstance(other, Waveform):
            return self._comb(other, mul)
        else:
            return self * const(other)

    def __rmul__(self, v):
        return const(v) * self

    def __truediv__(self, other):
        if isinstance(other, Waveform):
            raise TypeError('division by waveform')
        else:
            return self * const(1 / other)

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, v):
        return v + (-self)

    def __rshift__(self, time):
        return Waveform(
            tuple(round(bound + time, NDIGITS) for bound in self.bounds),
            tuple(shift(expr, time) for expr in self.seq))

    def __lshift__(self, time):
        return self >> (-time)

    @staticmethod
    def _merge_parts(
        parts: list[tuple[int, int, np.ndarray | int | float | complex]],
        out: list[tuple[int, int, np.ndarray | int | float | complex]]
    ) -> list[tuple[int, int, np.ndarray | int | float | complex]]:
        # TODO: merge parts
        raise NotImplementedError

    @staticmethod
    def _fill_parts(parts, out):
        for start, stop, part in parts:
            out[start:stop] += part

    def __call__(self,
                 x,
                 frag=False,
                 out=None,
                 accumulate=False,
                 function_lib=None):
        if function_lib is None:
            function_lib = _baseFunc
        if isinstance(x, (int, float, complex)):
            return self.__call__(np.array([x]), function_lib=function_lib)[0]
        parts, dtype = calc_parts(self.bounds, self.seq, x, function_lib,
                                  self.min, self.max)
        if not frag:
            if out is None:
                out = np.zeros_like(x, dtype=dtype)
            elif not accumulate:
                out *= 0
            self._fill_parts(parts, out)
        else:
            if out is None:
                return parts
            else:
                if not accumulate:
                    out.clear()
                    out.extend(parts)
                else:
                    self._merge_parts(parts, out)
        return out

    def __hash__(self):
        return hash((self.max, self.min, self.start, self.stop,
                     self.sample_rate, self.bounds, self.seq))

    def __eq__(self, o: object) -> bool:
        if isinstance(o, (int, float, complex)):
            return self == const(o)
        elif isinstance(o, Waveform):
            a = self.simplify()
            b = o.simplify()
            return a.seq == b.seq and a.bounds == b.bounds and (
                a.max, a.min, a.start, a.stop) == (b.max, b.min, b.start,
                                                   b.stop)
        else:
            return False

    def _repr_latex_(self):
        parts = []
        start = -np.inf
        for end, wav in zip(self.bounds, self.seq):
            e_str = _wav_latex(wav)
            start_str = _num_latex(start)
            end_str = _num_latex(end)
            parts.append(e_str + r",~~&t\in" + f"({start_str},{end_str}" +
                         (']' if end < np.inf else ')'))
            start = end
        if len(parts) == 1:
            expr = ''.join(['f(t)=', *parts[0].split('&')])
        else:
            expr = '\n'.join([
                r"f(t)=\begin{cases}", (r"\\" + '\n').join(parts),
                r"\end{cases}"
            ])
        return "$$\n{}\n$$".format(expr)

    def _play(self, time_unit, volume=1.0):
        import pyaudio

        CHUNK = 1024
        RATE = 48000

        dynamic_volume = 1.0
        amp = 2**15 * 0.999 * volume * dynamic_volume

        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=RATE,
                            output=True)
            try:
                for data in self.sample(sample_rate=RATE / time_unit,
                                        chunk_size=CHUNK):
                    lim = np.abs(data).max()
                    if lim > 0 and dynamic_volume > 1.0 / lim:
                        dynamic_volume = 1.0 / lim
                        amp = 2**15 * 0.99 * volume * dynamic_volume
                    data = (amp * data).astype(np.int16)
                    stream.write(bytes(data.data))
            finally:
                stream.stop_stream()
                stream.close()
        finally:
            p.terminate()

    def play(self, time_unit=1, volume=1.0):
        import multiprocessing as mp
        p = mp.Process(target=self._play,
                       args=(time_unit, volume),
                       daemon=True)
        p.start()


class WaveVStack(Waveform):

    def __init__(self, wlist: list[Waveform] = []):
        self.wlist = [(w.bounds, w.seq) for w in wlist]
        self.start = None
        self.stop = None
        self.sample_rate = None
        self.offset = 0
        self.shift = 0
        self.filters = None
        self.label = None
        self.function_lib = None

    def __call__(self, x, frag=False, out=None, function_lib=None):
        assert frag is False, 'WaveVStack does not support frag mode'
        out = np.full_like(x, self.offset, dtype=complex)
        if self.shift != 0:
            x = x - self.shift
        if function_lib is None:
            if self.function_lib is None:
                function_lib = _baseFunc
            else:
                function_lib = self.function_lib
        for bounds, seq in self.wlist:
            parts, dtype = calc_parts(bounds, seq, x, function_lib)
            self._fill_parts(parts, out)
        return out.real

    def tolist(self):
        l = [
            self.start,
            self.stop,
            self.offset,
            self.shift,
            self.sample_rate,
        ]
        if self.filters is None:
            l.append(None)
        else:
            sos, initial = self.filters
            sos = list(sos.reshape(-1))
            l.append(len(sos))
            l.extend(sos)
            l.append(initial)
        l.append(len(self.wlist))
        for bounds, seq in self.wlist:
            self._tolist(bounds, seq, l)
        return l

    @classmethod
    def fromlist(cls, l):
        w = cls()
        pos = 6
        w.start, w.stop, w.offset, w.shift, w.sample_rate, sos_size = l[:pos]
        if sos_size is not None:
            sos = np.array(l[pos:pos + sos_size]).reshape(-1, 6)
            pos += sos_size
            initial = l[pos]
            pos += 1
            w.filters = sos, initial
        n = l[pos]
        pos += 1
        for _ in range(n):
            bounds, seq, pos = cls._fromlist(l, pos)
            w.wlist.append((bounds, seq))
        return w

    def simplify(self, eps=1e-15):
        if not self.wlist:
            return zero()
        bounds, seq = wave_sum(self.wlist)
        wav = Waveform(bounds=bounds, seq=seq)
        if self.offset != 0:
            wav += self.offset
        if self.shift != 0:
            wav >>= self.shift
        wav = wav.simplify(eps)
        wav.start = self.start
        wav.stop = self.stop
        wav.sample_rate = self.sample_rate
        return wav

    @staticmethod
    def _rshift(wlist, time):
        if time == 0:
            return wlist
        return [(tuple(round(bound + time, NDIGITS) for bound in bounds),
                 tuple(shift(expr, time) for expr in seq))
                for bounds, seq in wlist]

    def __rshift__(self, time):
        ret = WaveVStack()
        ret.wlist = self.wlist
        ret.sample_rate = self.sample_rate
        ret.start = self.start
        ret.stop = self.stop
        ret.shift = self.shift + time
        ret.offset = self.offset
        return ret

    def __add__(self, other):
        ret = WaveVStack()
        ret.wlist.extend(self.wlist)
        if isinstance(other, WaveVStack):
            if other.shift != self.shift:
                ret.wlist = self._rshift(ret.wlist, self.shift)
                ret.wlist.extend(self._rshift(other.wlist, other.shift))
            else:
                ret.wlist.extend(other.wlist)
            ret.offset = self.offset + other.offset
        elif isinstance(other, Waveform):
            other <<= self.shift
            ret.wlist.append((other.bounds, other.seq))
        else:
            # ret.wlist.append(((+inf, ), (_const(1.0 * other), )))
            ret.offset += other
        return ret

    def __radd__(self, v):
        return self + v

    def __mul__(self, other):
        if isinstance(other, Waveform):
            other = other.simplify() << self.shift
            ret = WaveVStack([Waveform(*w) * other for w in self.wlist])
            if self.offset != 0:
                w = other * self.offset
                ret.wlist.append((w.bounds, w.seq))
            return ret
        else:
            ret = WaveVStack([Waveform(*w) * other for w in self.wlist])
            ret.offset = self.offset * other
            return ret

    def __rmul__(self, v):
        return self * v

    def __eq__(self, other):
        if self.wlist:
            return False
        else:
            return zero() == other

    def _repr_latex_(self):
        return r"\sum_{i=1}^{" + f"{len(self.wlist)}" + r"}" + r"f_i(t)"

    def __getstate__(self) -> tuple:
        function_lib = self.function_lib
        if function_lib:
            try:
                import dill
                function_lib = dill.dumps(function_lib)
            except:
                function_lib = None
        return (self.wlist, self.start, self.stop, self.sample_rate,
                self.offset, self.shift, self.filters, self.label,
                function_lib)

    def __setstate__(self, state: tuple) -> None:
        (self.wlist, self.start, self.stop, self.sample_rate, self.offset,
         self.shift, self.filters, self.label, function_lib) = state
        if function_lib:
            try:
                import dill
                function_lib = dill.loads(function_lib)
            except:
                function_lib = None
        self.function_lib = function_lib


def play(data, rate=48000):
    import io

    import pyaudio

    CHUNK = 1024

    max_amp = np.max(np.abs(data))

    if max_amp > 1:
        data /= max_amp

    data = np.array(2**15 * 0.999 * data, dtype=np.int16)
    buff = io.BytesIO(data.data)
    p = pyaudio.PyAudio()

    try:
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=rate,
                        output=True)
        try:
            while True:
                data = buff.read(CHUNK)
                if data:
                    stream.write(data)
                else:
                    break
        finally:
            stream.stop_stream()
            stream.close()
    finally:
        p.terminate()


_zero_waveform = Waveform()
_one_waveform = Waveform(seq=(_one, ))


def zero():
    return _zero_waveform


def one():
    return _one_waveform


def const(c):
    return Waveform(seq=(_const(1.0 * c), ))


# register base function
def _format_LINEAR(shift, *args):
    if shift != 0:
        shift = _num_latex(-shift)
        if shift[0] == '-':
            return f"(t{shift})"
        else:
            return f"(t+{shift})"
    else:
        return 't'


def _format_GAUSSIAN(shift, *args):
    sigma = _num_latex(args[0] / np.sqrt(2))
    shift = _num_latex(-shift)
    if shift != '0':
        if shift[0] != '-':
            shift = '+' + shift
        if sigma == '1':
            return ('\\exp\\left[-\\frac{\\left(t' + shift +
                    '\\right)^2}{2}\\right]')
        else:
            return ('\\exp\\left[-\\frac{1}{2}\\left(\\frac{t' + shift + '}{' +
                    sigma + '}\\right)^2\\right]')
    else:
        if sigma == '1':
            return ('\\exp\\left(-\\frac{t^2}{2}\\right)')
        else:
            return ('\\exp\\left[-\\frac{1}{2}\\left(\\frac{t}{' + sigma +
                    '}\\right)^2\\right]')


def _format_SINC(shift, *args):
    shift = _num_latex(-shift)
    bw = _num_latex(args[0])
    if shift != '0':
        if shift[0] != '-':
            shift = '+' + shift
        if bw == '1':
            return '\\mathrm{sinc}(t' + shift + ')'
        else:
            return '\\mathrm{sinc}[' + bw + '(t' + shift + ')]'
    else:
        if bw == '1':
            return '\\mathrm{sinc}(t)'
        else:
            return '\\mathrm{sinc}(' + bw + 't)'


def _format_COSINE(shift, *args):
    freq = args[0] / 2 / np.pi
    phase = -shift * freq
    freq = _num_latex(freq)
    if freq == '1':
        freq = ''
    phase = _num_latex(phase)
    if phase == '0':
        phase = ''
    elif phase[0] != '-':
        phase = '+' + phase
    if phase != '':
        return f'\\cos\\left[2\\pi\\left({freq}t{phase}\\right)\\right]'
    elif freq != '':
        return f'\\cos\\left(2\\pi\\times {freq}t\\right)'
    else:
        return '\\cos\\left(2\\pi t\\right)'


def _format_ERF(shift, *args):
    if shift > 0:
        return '\\mathrm{erf}(\\frac{t-' + f"{_num_latex(shift)}" + '}{' + f'{args[0]:g}' + '})'
    elif shift < 0:
        return '\\mathrm{erf}(\\frac{t+' + f"{_num_latex(-shift)}" + '}{' + f'{args[0]:g}' + '})'
    else:
        return '\\mathrm{erf}(\\frac{t}{' + f'{args[0]:g}' + '})'


def _format_COSH(shift, *args):
    if shift > 0:
        return '\\cosh(\\frac{t-' + f"{_num_latex(shift)}" + '}{' + f'{1/args[0]:g}' + '})'
    elif shift < 0:
        return '\\cosh(\\frac{t+' + f"{_num_latex(-shift)}" + '}{' + f'{1/args[0]:g}' + '})'
    else:
        return '\\cosh(\\frac{t}{' + f'{1/args[0]:g}' + '})'


def _format_SINH(shift, *args):
    if shift > 0:
        return '\\sinh(\\frac{t-' + f"{_num_latex(shift)}" + '}{' + f'{args[0]:g}' + '})'
    elif shift < 0:
        return '\\sinh(\\frac{t+' + f"{_num_latex(-shift)}" + '}{' + f'{args[0]:g}' + '})'
    else:
        return '\\sinh(\\frac{t}{' + f'{args[0]:g}' + '})'


def _format_EXP(shift, *args):
    if _num_latex(shift) and shift > 0:
        return '\\exp\\left(-' + f'{args[0]:g}' + '\\left(t-' + f"{_num_latex(shift)}" + '\\right)\\right)'
    elif _num_latex(-shift) and shift < 0:
        return '\\exp\\left(-' + f'{args[0]:g}' + '\\left(t+' + f"{_num_latex(-shift)}" + '\\right)\\right)'
    else:
        return '\\exp\\left(-' + f'{args[0]:g}' + 't\\right)'


def _format_DRAG(shift, *args):
    return f"DRAG(...)"


registerBaseFuncLatex(LINEAR, _format_LINEAR)
registerBaseFuncLatex(GAUSSIAN, _format_GAUSSIAN)
registerBaseFuncLatex(ERF, _format_ERF)
registerBaseFuncLatex(COS, _format_COSINE)
registerBaseFuncLatex(SINC, _format_SINC)
registerBaseFuncLatex(EXP, _format_EXP)
registerBaseFuncLatex(COSH, _format_COSH)
registerBaseFuncLatex(SINH, _format_SINH)
registerBaseFuncLatex(DRAG, _format_DRAG)


def D(wav):
    """derivative
    """
    return Waveform(bounds=wav.bounds, seq=tuple(_D(x) for x in wav.seq))


def convolve(a, b):
    pass


def sign():
    return Waveform(bounds=(0, +inf), seq=(_const(-1), _one))


def step(edge, type='erf'):
    """
    type: "erf", "cos", "linear"
    """
    if edge == 0:
        return Waveform(bounds=(0, +inf), seq=(_zero, _one))
    if type == 'cos':
        rise = add(_half,
                   mul(_half, basic_wave(COS, pi / edge, shift=0.5 * edge)))
        return Waveform(bounds=(round(-edge / 2,
                                      NDIGITS), round(edge / 2,
                                                      NDIGITS), +inf),
                        seq=(_zero, rise, _one))
    elif type == 'linear':
        rise = add(_half, mul(_const(1 / edge), basic_wave(LINEAR)))
        return Waveform(bounds=(round(-edge / 2,
                                      NDIGITS), round(edge / 2,
                                                      NDIGITS), +inf),
                        seq=(_zero, rise, _one))
    else:
        std_sq2 = edge / 5
        # rise = add(_half, mul(_half, basic_wave(ERF, std_sq2)))
        rise = ((((), ()), (((ERF, std_sq2, 0), ), (1, ))), (0.5, 0.5))
        return Waveform(bounds=(-round(edge, NDIGITS), round(edge,
                                                             NDIGITS), +inf),
                        seq=(_zero, rise, _one))


def square(width, edge=0, type='erf'):
    if width <= 0:
        return zero()
    if edge == 0:
        return Waveform(bounds=(round(-0.5 * width,
                                      NDIGITS), round(0.5 * width,
                                                      NDIGITS), +inf),
                        seq=(_zero, _one, _zero))
    else:
        return ((step(edge, type=type) << width / 2) -
                (step(edge, type=type) >> width / 2))


def gaussian(width, plateau=0.0):
    if width <= 0 and plateau <= 0.0:
        return zero()
    # width is two times FWHM
    # std_sq2 = width / (4 * np.sqrt(np.log(2)))
    std_sq2 = width / 3.3302184446307908
    # std is set to give total pulse area same as a square
    # std_sq2 = width/np.sqrt(np.pi)
    if round(0.5 * plateau, NDIGITS) <= 0.0:
        return Waveform(bounds=(round(-0.75 * width,
                                      NDIGITS), round(0.75 * width,
                                                      NDIGITS), +inf),
                        seq=(_zero, basic_wave(GAUSSIAN, std_sq2), _zero))
    else:
        return Waveform(bounds=(round(-0.75 * width - 0.5 * plateau,
                                      NDIGITS), round(-0.5 * plateau, NDIGITS),
                                round(0.5 * plateau, NDIGITS),
                                round(0.75 * width + 0.5 * plateau,
                                      NDIGITS), +inf),
                        seq=(_zero,
                             basic_wave(GAUSSIAN,
                                        std_sq2,
                                        shift=-0.5 * plateau), _one,
                             basic_wave(GAUSSIAN, std_sq2,
                                        shift=0.5 * plateau), _zero))


def cos(w, phi=0):
    if w == 0:
        return const(np.cos(phi))
    if w < 0:
        phi = -phi
        w = -w
    return Waveform(seq=(basic_wave(COS, w, shift=-phi / w), ))


def sin(w, phi=0):
    if w == 0:
        return const(np.sin(phi))
    if w < 0:
        phi = -phi + pi
        w = -w
    return Waveform(seq=(basic_wave(COS, w, shift=(pi / 2 - phi) / w), ))


def exp(alpha):
    if isinstance(alpha, complex):
        if alpha.real == 0:
            return cos(alpha.imag) + 1j * sin(alpha.imag)
        else:
            return exp(alpha.real) * (cos(alpha.imag) + 1j * sin(alpha.imag))
    else:
        return Waveform(seq=(basic_wave(EXP, alpha), ))


def sinc(bw):
    if bw <= 0:
        return zero()
    width = 100 / bw
    return Waveform(bounds=(round(-0.5 * width,
                                  NDIGITS), round(0.5 * width, NDIGITS), +inf),
                    seq=(_zero, basic_wave(SINC, bw), _zero))


def cosPulse(width, plateau=0.0):
    # cos = basic_wave(COS, 2*np.pi/width)
    # pulse = mul(add(cos, _one), _half)
    if round(0.5 * plateau, NDIGITS) > 0:
        return square(plateau + 0.5 * width, edge=0.5 * width, type='cos')
    if width <= 0:
        return zero()
    pulse = ((((), ()), (((COS, 6.283185307179586 / width, 0), ), (1, ))),
             (0.5, 0.5))
    return Waveform(bounds=(round(-0.5 * width,
                                  NDIGITS), round(0.5 * width, NDIGITS), +inf),
                    seq=(_zero, pulse, _zero))


def hanning(width, plateau=0.0):
    return cosPulse(width, plateau=plateau)


def cosh(w):
    return Waveform(seq=(basic_wave(COSH, w), ))


def sinh(w):
    return Waveform(seq=(basic_wave(SINH, w), ))


def coshPulse(width, eps=1.0, plateau=0.0):
    """Cosine hyperbolic pulse with the following im

    pulse edge shape:
            cosh(eps / 2) - cosh(eps * t / T)
    f(t) = -----------------------------------
                  cosh(eps / 2) - 1
    where T is the pulse width and eps is the pulse edge steepness.
    The pulse is defined for t in [-T/2, T/2].

    In case of plateau > 0, the pulse is defined as:
           | f(t + plateau/2)   if t in [-T/2 - plateau/2, - plateau/2]
    g(t) = | 1                  if t in [-plateau/2, plateau/2]
           | f(t - plateau/2)   if t in [plateau/2, T/2 + plateau/2]

    Parameters
    ----------
    width : float
        Pulse width.
    eps : float
        Pulse edge steepness.
    plateau : float
        Pulse plateau.
    """
    if width <= 0 and plateau <= 0:
        return zero()
    w = eps / width
    A = np.cosh(eps / 2)

    if plateau == 0.0 or round(-0.5 * plateau, NDIGITS) == round(
            0.5 * plateau, NDIGITS):
        pulse = ((((), ()), (((COSH, w, 0), ), (1, ))), (A / (A - 1),
                                                         -1 / (A - 1)))
        return Waveform(bounds=(round(-0.5 * width,
                                      NDIGITS), round(0.5 * width,
                                                      NDIGITS), +inf),
                        seq=(_zero, pulse, _zero))
    else:
        raising = ((((), ()), (((COSH, w, -0.5 * plateau), ), (1, ))),
                   (A / (A - 1), -1 / (A - 1)))
        falling = ((((), ()), (((COSH, w, 0.5 * plateau), ), (1, ))),
                   (A / (A - 1), -1 / (A - 1)))
        return Waveform(bounds=(round(-0.5 * width - 0.5 * plateau,
                                      NDIGITS), round(-0.5 * plateau, NDIGITS),
                                round(0.5 * plateau, NDIGITS),
                                round(0.5 * width + 0.5 * plateau,
                                      NDIGITS), +inf),
                        seq=(_zero, raising, _one, falling, _zero))


def general_cosine(duration, *arg):
    wav = zero()
    arg = np.asarray(arg)
    arg /= arg[::2].sum()
    for i, a in enumerate(arg, start=1):
        wav += a / 2 * (1 - (-1)**i * cos(i * 2 * pi / duration))
    return wav * square(duration)


def slepian(duration, *arg):
    wav = zero()
    arg = np.asarray(arg)
    arg /= arg[::2].sum()
    for i, a in enumerate(arg, start=1):
        wav += a / 2 * (1 - (-1)**i * cos(i * 2 * pi / duration))
    return wav * square(duration)


def _poly(*a):
    """
    a[0] + a[1] * t + a[2] * t**2 + ...
    """
    t = []
    amp = []
    if a[0] != 0:
        t.append(((), ()))
        amp.append(a[0])
    for n, a_ in enumerate(a[1:], start=1):
        if a_ != 0:
            t.append((((LINEAR, 0), ), (n, )))
            amp.append(a_)
    return tuple(t), tuple(a)


def poly(a):
    """
    a[0] + a[1] * t + a[2] * t**2 + ...
    """
    return Waveform(seq=(_poly(*a), ))


def t():
    return Waveform(seq=((((LINEAR, 0), ), (1, )), (1, )))


def drag(freq, width, plateau=0, delta=0, block_freq=None, phase=0, t0=0):
    phase += pi * delta * (width + plateau)
    if plateau <= 0:
        return Waveform(seq=(_zero,
                             basic_wave(DRAG, t0, freq, width, delta,
                                        block_freq, phase), _zero),
                        bounds=(round(t0, NDIGITS), round(t0 + width,
                                                          NDIGITS), +inf))
    elif width <= 0:
        w = 2 * pi * (freq + delta)
        return Waveform(
            seq=(_zero,
                 basic_wave(COS, w,
                            shift=(phase + 2 * pi * delta * t0) / w), _zero),
            bounds=(round(t0, NDIGITS), round(t0 + plateau, NDIGITS), +inf))
    else:
        w = 2 * pi * (freq + delta)
        return Waveform(
            seq=(_zero,
                 basic_wave(DRAG, t0, freq, width, delta, block_freq, phase),
                 basic_wave(COS, w, shift=(phase + 2 * pi * delta * t0) / w),
                 basic_wave(DRAG, t0 + plateau, freq, width, delta, block_freq,
                            phase - 2 * pi * delta * plateau), _zero),
            bounds=(round(t0, NDIGITS), round(t0 + width / 2, NDIGITS),
                    round(t0 + width / 2 + plateau,
                          NDIGITS), round(t0 + width + plateau,
                                          NDIGITS), +inf))


def chirp(f0, f1, T, phi0=0, type='linear'):
    """
    A chirp is a signal in which the frequency increases (up-chirp)
    or decreases (down-chirp) with time. In some sources, the term
    chirp is used interchangeably with sweep signal.

    type: "linear", "exponential", "hyperbolic"
    """
    if f0 == f1:
        return sin(f0, phi0)
    if T <= 0:
        raise ValueError('T must be positive')

    if type == 'linear':
        # f(t) = f1 * (t/T) + f0 * (1 - t/T)
        return Waveform(bounds=(0, round(T, NDIGITS), +inf),
                        seq=(_zero, basic_wave(LINEARCHIRP, f0, f1, T,
                                               phi0), _zero))
    elif type in ['exp', 'exponential', 'geometric']:
        # f(t) = f0 * (f1/f0) ** (t/T)
        if f0 == 0:
            raise ValueError('f0 must be non-zero')
        alpha = np.log(f1 / f0) / T
        return Waveform(bounds=(0, round(T, NDIGITS), +inf),
                        seq=(_zero,
                             basic_wave(EXPONENTIALCHIRP, f0, alpha,
                                        phi0), _zero))
    elif type in ['hyperbolic', 'hyp']:
        # f(t) = f0 * f1 / (f0 * (t/T) + f1 * (1-t/T))
        if f0 * f1 == 0:
            return const(np.sin(phi0))
        k = (f0 - f1) / (f1 * T)
        return Waveform(bounds=(0, round(T, NDIGITS), +inf),
                        seq=(_zero, basic_wave(HYPERBOLICCHIRP, f0, k,
                                               phi0), _zero))
    else:
        raise ValueError(f'unknown type {type}')


def interp(x, y):
    seq, bounds = [_zero], [x[0]]
    for x1, x2, y1, y2 in zip(x[:-1], x[1:], y[:-1], y[1:]):
        if x2 == x1:
            continue
        seq.append(
            add(
                mul(_const((y2 - y1) / (x2 - x1)), basic_wave(LINEAR,
                                                              shift=x1)),
                _const(y1)))
        bounds.append(x2)
    bounds.append(inf)
    seq.append(_zero)
    return Waveform(seq=tuple(seq),
                    bounds=tuple(round(b, NDIGITS)
                                 for b in bounds)).simplify()


def cut(wav, start=None, stop=None, head=None, tail=None, min=None, max=None):
    offset = 0
    if start is not None and head is not None:
        offset = head - wav(np.array([1.0 * start]))[0]
    elif stop is not None and tail is not None:
        offset = tail - wav(np.array([1.0 * stop]))[0]
    wav = wav + offset

    if start is not None:
        wav = wav * (step(0) >> start)
    if stop is not None:
        wav = wav * ((1 - step(0)) >> stop)
    if min is not None:
        wav.min = min
    if max is not None:
        wav.max = max
    return wav


def function(fun, *args, start=None, stop=None):
    TYPEID = registerBaseFunc(fun)
    seq = (basic_wave(TYPEID, *args), )
    wav = Waveform(seq=seq)
    if start is not None:
        wav = wav * (step(0) >> start)
    if stop is not None:
        wav = wav * ((1 - step(0)) >> stop)
    return wav


def samplingPoints(start, stop, points):
    return Waveform(bounds=(round(start, NDIGITS), round(stop, NDIGITS), inf),
                    seq=(_zero, basic_wave(INTERP, start, stop,
                                           tuple(points)), _zero))


def mixing(I,
           Q=None,
           *,
           phase=0.0,
           freq=0.0,
           ratioIQ=1.0,
           phaseDiff=0.0,
           block_freq=None,
           DRAGScaling=None):
    """SSB or envelope mixing
    """
    if Q is None:
        I = I
        Q = zero()

    w = 2 * pi * freq
    if freq != 0.0:
        # SSB mixing
        Iout = I * cos(w, -phase) + Q * sin(w, -phase)
        Qout = -I * sin(w, -phase + phaseDiff) + Q * cos(w, -phase + phaseDiff)
    else:
        # envelope mixing
        Iout = I * np.cos(-phase) + Q * np.sin(-phase)
        Qout = -I * np.sin(-phase) + Q * np.cos(-phase)

    # apply DRAG
    if block_freq is not None and block_freq != freq:
        a = block_freq / (block_freq - freq)
        b = 1 / (block_freq - freq)
        I = a * Iout + b / (2 * pi) * D(Qout)
        Q = a * Qout - b / (2 * pi) * D(Iout)
        Iout, Qout = I, Q
    elif DRAGScaling is not None and DRAGScaling != 0:
        # 2 * pi * scaling * (freq - block_freq) = 1
        I = (1 - w * DRAGScaling) * Iout - DRAGScaling * D(Qout)
        Q = (1 - w * DRAGScaling) * Qout + DRAGScaling * D(Iout)
        Iout, Qout = I, Q

    Qout = ratioIQ * Qout

    return Iout, Qout


__all__ = [
    'D', 'Waveform', 'chirp', 'const', 'cos', 'cosh', 'coshPulse', 'cosPulse',
    'cut', 'drag', 'exp', 'function', 'gaussian', 'general_cosine', 'hanning',
    'interp', 'mixing', 'one', 'poly', 'registerBaseFunc',
    'registerDerivative', 'samplingPoints', 'sign', 'sin', 'sinc', 'sinh',
    'square', 'step', 't', 'zero'
]
