import pickle
from bisect import bisect_left
from fractions import Fraction
from itertools import chain, product
from math import comb

import numpy as np
import scipy.special as special
from numpy import e, inf, pi
from scipy.signal import sosfilt

NDIGITS = 15

_zero = ((), ())


def _const(c):
    if c == 0:
        return _zero
    return (((), ()), ), (c, )


_one = _const(1.0)
_half = _const(1 / 2)
_two = _const(2.0)
_pi = _const(pi)
_two_pi = _const(2 * pi)
_half_pi = _const(pi / 2)


def _is_const(x):
    return x == _zero or x[0] == (((), ()), )


def _basic_wave(Type, *args, shift=0):
    return ((((Type, *args, shift), ), (1, )), ), (1.0, )


def _insert_type_value_pair(t_list, v_list, t, v, lo, hi):
    i = bisect_left(t_list, t, lo, hi)
    if i < hi and t_list[i] == t:
        v += v_list[i]
        if v == 0:
            t_list.pop(i)
            v_list.pop(i)
            return i, hi - 1
        else:
            v_list[i] = v
            return i, hi
    else:
        t_list.insert(i, t)
        v_list.insert(i, v)
        return i, hi + 1


def _mul(x, y):
    t_list, v_list = [], []
    xt_list, xv_list = x
    yt_list, yv_list = y
    lo, hi = 0, 0
    for (t1, t2), (v1, v2) in zip(product(xt_list, yt_list),
                                  product(xv_list, yv_list)):
        if v1 * v2 == 0:
            continue
        t = _add(t1, t2)
        lo, hi = _insert_type_value_pair(t_list, v_list, t, v1 * v2, lo, hi)
    return tuple(t_list), tuple(v_list)


def _add(x, y):
    # x, y = (x, y) if len(x[0]) >= len(y[0]) else (y, x)
    t_list, v_list = list(x[0]), list(x[1])
    lo, hi = 0, len(t_list)
    for t, v in zip(*y):
        lo, hi = _insert_type_value_pair(t_list, v_list, t, v, lo, hi)
    return tuple(t_list), tuple(v_list)


def _shift(x, time):
    if _is_const(x):
        return x

    t_list = []

    for pre_mtlist, nlist in x[0]:
        mtlist = []
        for Type, *args, shift in pre_mtlist:
            mtlist.append((Type, *args, shift + time))
        t_list.append((tuple(mtlist), nlist))
    return tuple(t_list), x[1]


def _pow(x, n):
    if x == _zero:
        return _zero
    if n == 0:
        return _one
    if _is_const(x):
        return _const(x[1][0]**n)

    if len(x[0]) == 1:
        t_list, v_list = [], []
        for (mtlist, pre_nlist), v in zip(*x):
            nlist = []
            for m in pre_nlist:
                nlist.append(n * m)
            t_list.append((mtlist, tuple(nlist)))
            v_list.append(v**n)
        return tuple(t_list), tuple(v_list)
    else:
        assert isinstance(n, int) and n > 0
        ret = _one
        for i in range(n):
            ret = _mul(ret, x)
        return ret


def _cos_power_n(x, n):
    _, w, shift = x
    ret = _zero
    for k in range(0, n // 2 + 1):
        if n == 2 * k:
            a = _const(comb(n, k) / 2**n)
            ret = _add(ret, a)
        else:
            expr = (((((COS, (n - 2 * k) * w, shift), ), (1, )), ),
                    (comb(n, k) / 2**(n - 1), ))
            ret = _add(ret, expr)
    return ret


def _trigMul_t(x, y, v):
    """cos(a)cos(b) = 0.5*cos(a+b)+0.5*cos(a-b)"""
    _, w1, t1 = x
    _, w2, t2 = y
    if w2 > w1:
        t1, t2 = t2, t1
        w1, w2 = w2, w1
    exp1 = (COS, w1 + w2, (w1 * t1 + w2 * t2) / (w1 + w2))
    if w1 == w2:
        c = v * np.cos(w1 * t1 - w2 * t2) / 2
        if c == 0:
            return (((exp1, ), (1, )), ), (0.5 * v, )
        else:
            return (((), ()), ((exp1, ), (1, ))), (c, 0.5 * v)
    else:
        exp2 = (COS, w1 - w2, (w1 * t1 - w2 * t2) / (w1 - w2))
        if exp2[1] > exp1[1]:
            exp2, exp1 = exp1, exp2
        return (((exp2, ), (1, )), ((exp1, ), (1, ))), (0.5 * v, 0.5 * v)


def _trigMul(x, y):
    if _is_const(x) or _is_const(y):
        return _mul(x, y)
    ret = _zero
    for (t1, t2), (v1, v2) in zip(product(x[0], y[0]), product(x[1], y[1])):
        v = v1 * v2
        tmp = _one
        trig = []
        for mt, n in zip(chain(t1[0], t2[0]), chain(t1[1], t2[1])):
            if mt[0] == COS:
                trig.append(mt)
            else:
                tmp = _mul(tmp, ((((mt, ), (n, )), ), (1, )))
        if len(trig) == 1:
            x = ((((trig[0], ), (1, )), ), (v, ))
            expr = _mul(tmp, x)
        elif len(trig) == 2:
            expr = _trigMul_t(trig[0], trig[1], v)
            expr = _mul(tmp, expr)
        else:
            expr = _mul(tmp, _const(v))
        ret = _add(ret, expr)
    return ret


def _exp_trig_Reduce(mtlist, v):
    trig = _one
    alpha = 0
    shift = 0
    ml, nl = [], []
    for mt, n in zip(*mtlist):
        if mt[0] == COS:
            trig = _trigMul(trig, _cos_power_n(mt, n))
        elif mt[0] == EXP:
            x = alpha * shift + n * mt[1] * mt[-1]
            alpha += n * mt[1]
            if alpha == 0:
                shift = 0
            else:
                shift = x / alpha
        elif mt[0] == GAUSSIAN and n != 1:
            ml.append((mt[0], mt[1] / np.sqrt(n), mt[2]))
            nl.append(1)
        else:
            ml.append(mt)
            nl.append(n)
    ret = (((tuple(ml), tuple(nl)), ), (v, ))

    if alpha != 0:
        ret = _mul(ret, _basic_wave(EXP, alpha, shift=shift))

    return _mul(ret, trig)


def _get_freq(t):
    t2 = [[], []]
    freq, shift = 0, 0
    for mt, n in zip(*t):
        if mt[0] == COS:
            if freq != 0:
                raise ValueError("run _exp_trig_Reduce first")
            freq = mt[1]
            shift = mt[-1]
        else:
            t2[0].append(mt)
            t2[1].append(n)
    t2 = (tuple(t2[0]), tuple(t2[1]))
    return freq, shift, t2


def _simplify(expr, eps):
    d = {}
    for t, v in zip(*expr):
        for t, v in zip(*_exp_trig_Reduce(t, v)):
            freq, shift, t = _get_freq(t)
            v_r, v_i, shift_r, shift_i = v.real, v.imag, shift, shift
            if (t, freq) in d:
                v0_r, shift0_r, v0_i, shift0_i = d[(t, freq)]
                if freq == 0:
                    v_r, v_i = v.real + v0_r, v.imag + v0_i
                else:
                    a = v0_r * np.cos(freq * shift0_r) + v_r * np.cos(
                        freq * shift_r)
                    b = v0_r * np.sin(freq * shift0_r) + v_r * np.sin(
                        freq * shift_r)
                    shift_r = np.arctan2(b, a) / freq
                    v_r = np.sqrt(a**2 + b**2)

                    a = v0_i * np.cos(freq * shift0_i) + v_i * np.cos(
                        freq * shift_i)
                    b = v0_i * np.sin(freq * shift0_i) + v_i * np.sin(
                        freq * shift_i)
                    shift_i = np.arctan2(b, a) / freq
                    v_i = np.sqrt(a**2 + b**2)
            d[(t, freq)] = v_r, shift_r, v_i, shift_i
    ret = _zero
    for (t, freq), (v_r, shift_r, v_i, shift_i) in d.items():
        if freq == 0 and abs(v) >= eps:
            if v_i == 0:
                ret = _add(ret, ((t, ), (v_r, )))
            else:
                ret = _add(ret, ((t, ), (v_r + 1j * v_i, )))
        else:
            if abs(v_i) < eps and abs(v_r) < eps:
                continue
            elif abs(v_i) < eps and abs(v_r) >= eps:
                expr = (((((COS, freq, shift_r), ), (1, )), ), (v_r, ))
            elif abs(v_i) >= eps and abs(v_r) < eps:
                expr = (((((COS, freq, shift_i), ), (1, )), ), (v_i * 1j, ))
            elif abs(v_i) >= eps and abs(v_r) >= eps:
                expr = (((((COS, freq, shift_r), ), (1, )),
                         (((COS, freq, shift_i), ), (1, ))), (v_r, v_i * 1j))
            else:
                pass  # Never reach here

            expr = _mul(((t, ), (1, )), expr)
            ret = _add(ret, expr)
    return ret


def _filter(expr, low, high, eps):
    expr = _simplify(expr, eps)
    ret = _zero
    for t, v in zip(*expr):
        for i, (mt, n) in enumerate(zip(*t)):
            if mt[0] == COS:
                if low <= mt[1] < high:
                    ret = _add(ret, ((t, ), (v, )))
                break
            elif mt[0] == SINC and n == 1:
                pass
            elif mt[0] == GAUSSIAN and n == 1:
                pass
        else:
            if low <= 0:
                ret = _add(ret, ((t, ), (v, )))
    return ret


def _apply(function_lib, func_id, x, shift, *args):
    return function_lib[func_id](x - shift, *args)


def _calc(wav, x, function_lib):
    lru_cache = {}

    def _calc_m(t, x):
        ret = 1
        for mt, n in zip(*t):
            if mt not in lru_cache:
                func_id, *args, shift = mt
                lru_cache[mt] = _apply(function_lib, func_id, x, shift, *args)
            if n == 1:
                ret = ret * lru_cache[mt]
            else:
                ret = ret * lru_cache[mt]**n
        return ret

    ret = 0
    for t, v in zip(*wav):
        ret = ret + v * _calc_m(t, x)
    return ret


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
    from waveforms.waveform import _is_const, _zero

    if wav == _zero:
        return "0"
    elif _is_const(wav):
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
            if out is not None:
                if filters is None:
                    yield self.__call__(x,
                                        out=out[start_n:],
                                        function_lib=function_lib)
                else:
                    if initial:
                        sig -= initial
                    sig, zi = sosfilt(sos,
                                      self.__call__(x,
                                                    function_lib=function_lib),
                                      zi=zi)
                    if initial:
                        sig += initial
                    out[start_n:start_n + size] = sig
                    yield sig
            else:
                if filters is None:
                    yield self.__call__(x, function_lib=function_lib)
                else:
                    if initial:
                        sig -= initial
                    sig, zi = sosfilt(sos,
                                      self.__call__(x,
                                                    function_lib=function_lib),
                                      zi=zi)
                    if initial:
                        sig += initial
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
        seq = [_simplify(self.seq[0], eps)]
        bounds = [self.bounds[0]]
        for expr, b in zip(self.seq[1:], self.bounds[1:]):
            expr = _simplify(expr, eps)
            if expr == seq[-1]:
                seq.pop()
                bounds.pop()
            seq.append(expr)
            bounds.append(b)
        return Waveform(tuple(bounds), tuple(seq))

    def filter(self, low=0, high=inf, eps=1e-15):
        seq = []
        for expr in self.seq:
            seq.append(_filter(expr, low, high, eps))
        return Waveform(self.bounds, tuple(seq))

    def _comb(self, other, oper):
        bounds = []
        seq = []
        i1, i2 = 0, 0
        h1, h2 = len(self.bounds), len(other.bounds)
        while i1 < h1 or i2 < h2:
            s = oper(self.seq[i1], other.seq[i2])
            b = min(self.bounds[i1], other.bounds[i2])
            if seq and s == seq[-1]:
                bounds[-1] = b
            else:
                bounds.append(b)
                seq.append(s)
            if b == self.bounds[i1]:
                i1 += 1
            if b == other.bounds[i2]:
                i2 += 1
        return Waveform(tuple(bounds), tuple(seq))

    def __pow__(self, n):
        return Waveform(self.bounds, tuple(_pow(w, n) for w in self.seq))

    def __add__(self, other):
        if isinstance(other, Waveform):
            return self._comb(other, _add)
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
            return self._comb(other, _mul)
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
            tuple(_shift(expr, time) for expr in self.seq))

    def __lshift__(self, time):
        return self >> (-time)

    @staticmethod
    def _calc_parts(bounds, seq, x, function_lib, min=-inf, max=inf):
        range_list = np.searchsorted(x, bounds)
        parts = []
        start, stop = 0, 0
        dtype = float
        for i, stop in enumerate(range_list):
            if start < stop and seq[i] != _zero:
                part = np.clip(_calc(seq[i], x[start:stop], function_lib), min,
                               max)
                if (isinstance(part, complex) or isinstance(part, np.ndarray)
                        and isinstance(part[0], complex)):
                    dtype = complex
                parts.append((start, stop, part))
            start = stop
        return parts, dtype

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
        parts, dtype = self._calc_parts(self.bounds, self.seq, x, function_lib,
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
            parts, dtype = self._calc_parts(bounds, seq, x, function_lib)
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
        wav = wave_sum(self.wlist)
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
                 tuple(_shift(expr, time) for expr in seq))
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


def wave_sum(waves):
    if not waves:
        return zero()

    bounds, seq = waves[0]
    if not waves[1:]:
        return Waveform(bounds, seq)
    bounds, seq = list(bounds), list(seq)

    for bounds_, seq_ in waves[1:]:
        if len(bounds_) == 1:
            for i, s in enumerate(seq):
                seq[i] = _add(s, seq_[0])
        elif len(bounds) == 1:
            bounds = list(bounds_)
            seq = [_add(seq[0], s) for s in seq_]
        else:
            lo = 0
            for b, s in zip(bounds_, seq_):
                i = bisect_left(bounds, b, lo=lo)
                if bounds[i] > b:
                    bounds.insert(i, b)
                    if i == 0:
                        seq.insert(i, s)
                    else:
                        seq.insert(i, _add(s, seq[i]))
                    up = i - 1
                else:
                    up = i
                for j in range(lo + 1, up + 1):
                    seq[j] = _add(seq[j], s)
                lo = i

    i = 0
    while i < len(bounds) - 1:
        if seq[i] == seq[i + 1]:
            del seq[i + 1]
            del bounds[i + 1]
        else:
            i += 1

    return Waveform(tuple(bounds), tuple(seq))


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


__TypeIndex = 1
_baseFunc = {}
_derivativeBaseFunc = {}
_baseFunc_latex = {}


def registerBaseFunc(func, latex=None):
    global __TypeIndex
    Type = __TypeIndex
    __TypeIndex += 1

    _baseFunc[Type] = func
    _baseFunc_latex[Type] = latex

    return Type


def packBaseFunc():
    return pickle.dumps(_baseFunc)


def updateBaseFunc(buf):
    _baseFunc.update(pickle.loads(buf))


def registerDerivative(Type, dFunc):
    _derivativeBaseFunc[Type] = dFunc


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


LINEAR = registerBaseFunc(lambda t: t, _format_LINEAR)
GAUSSIAN = registerBaseFunc(lambda t, std_sq2: np.exp(-(t / std_sq2)**2),
                            _format_GAUSSIAN)
ERF = registerBaseFunc(lambda t, std_sq2: special.erf(t / std_sq2),
                       _format_ERF)
COS = registerBaseFunc(lambda t, w: np.cos(w * t), _format_COSINE)
SINC = registerBaseFunc(lambda t, bw: np.sinc(bw * t), _format_SINC)
EXP = registerBaseFunc(lambda t, alpha: np.exp(alpha * t), _format_EXP)
INTERP = registerBaseFunc(lambda t, start, stop, points: np.interp(
    t, np.linspace(start, stop, len(points)), points))
LINEARCHIRP = registerBaseFunc(lambda t, f0, f1, T, phi0: np.sin(
    phi0 + 2 * np.pi * ((f1 - f0) / (2 * T) * t**2 + f0 * t)))
EXPONENTIALCHIRP = registerBaseFunc(lambda t, f0, alpha, phi0: np.sin(
    phi0 + 2 * pi * f0 * (np.exp(alpha * t) - 1) / alpha))
HYPERBOLICCHIRP = registerBaseFunc(lambda t, f0, k, phi0: np.sin(
    phi0 + 2 * np.pi * f0 / k * np.log(1 + k * t)))
COSH = registerBaseFunc(lambda t, w: np.cosh(w * t), _format_COSH)
SINH = registerBaseFunc(lambda t, w: np.sinh(w * t), _format_SINH)


def _drag(t: np.ndarray, t0: float, freq: float, width: float, delta: float,
          block_freq: float, phase: float):

    o = np.pi / width
    Omega_x = np.sin(o * (t - t0))**2
    wt = 2 * np.pi * (freq + delta) * t - (2 * np.pi * delta * t0 + phase)

    if block_freq is None or block_freq - delta == 0:
        return Omega_x * np.cos(wt)

    b = 1 / np.pi / 2 / (block_freq - delta)
    Omega_y = -b * o * np.sin(2 * o * (t - t0))

    return Omega_x * np.cos(wt) + Omega_y * np.sin(wt)


def _format_DRAG(shift, *args):
    return f"DRAG(...)"


DRAG = registerBaseFunc(_drag, _format_DRAG)

# register derivative
registerDerivative(LINEAR, lambda shift, *args: _one)

registerDerivative(
    GAUSSIAN, lambda shift, *args: (((((LINEAR, shift),
                                       (GAUSSIAN, *args, shift)), (1, 1)), ),
                                    (-2 / args[0]**2, )))

registerDerivative(
    ERF, lambda shift, *args: (((((GAUSSIAN, *args, shift), ), (1, )), ),
                               (2 / args[0] / np.sqrt(pi), )))

registerDerivative(
    COS, lambda shift, *args: (((((COS, args[0], shift - pi / args[0] / 2), ),
                                 (1, )), ), (args[0], )))

registerDerivative(
    SINC, lambda shift, *args:
    (((((LINEAR, shift), (COS, *args, shift)), (-1, 1)),
      (((LINEAR, shift), (COS, args[0], args[1] - pi / 2, shift)), (-2, 1))),
     (1, -1 / args[0])))

registerDerivative(
    EXP, lambda shift, *args: (((((EXP, *args, shift), ), (1, )), ),
                               (args[0], )))

registerDerivative(
    INTERP, lambda shift, start, stop, points:
    (((((INTERP, start, stop, tuple(np.gradient(np.asarray(points))), shift),
        ), (1, )), ), ((len(points) - 1) / (stop - start), )))

registerDerivative(
    COSH, lambda shift, *args: (((((SINH, *args, shift), ), (1, )), ),
                                (args[0], )))

registerDerivative(
    SINH, lambda shift, *args: (((((COSH, *args, shift), ), (1, )), ),
                                (args[0], )))


def _d_LINEARCHIRP(shift, f0, f1, T, phi0):
    tlist = (
        (((LINEARCHIRP, f0, f1, T, phi0 + pi / 2, shift), ), (1, )),
        (((LINEAR, shift), (LINEARCHIRP, f0, f1, T, phi0 + pi / 2, shift)),
         (1, 1)),
    )
    alist = (2 * pi * f0, 2 * pi * (f1 - f0) / T)

    if f0 == 0:
        return tlist[1:], alist[1:]
    else:
        return tlist, alist


registerDerivative(LINEARCHIRP, _d_LINEARCHIRP)
registerDerivative(
    EXPONENTIALCHIRP, lambda shift, f0, alpha, phi0:
    (((((EXP, alpha, shift),
        (EXPONENTIALCHIRP, f0, alpha, phi0 + pi / 2, shift)), (1, 1)), ),
     (2 * pi * f0, )))
registerDerivative(
    HYPERBOLICCHIRP, lambda shift, f0, k, phi0:
    (((((LINEAR, shift - 1 / k),
        (HYPERBOLICCHIRP, f0, k, phi0 + pi / 2, shift)), (-1, 1)), ),
     (2 * pi * f0, )))


def _D_base(m):
    Type, *args, shift = m
    return _derivativeBaseFunc[Type](shift, *args)


def _D(x):
    if _is_const(x):
        return _zero
    t_list, v_list = x
    if len(v_list) == 1:
        (m_list, n_list), v = t_list[0], v_list[0]
        if len(m_list) == 1:
            m, n = m_list[0], n_list[0]
            if n == 1:
                return _mul(_D_base(m), _const(v))
            else:
                return _mul(((((m, ), (n - 1, )), ), (n * v, )),
                            _D(((((m, ), (1, )), ), (1, ))))
        else:
            a = (((m_list[:1], n_list[:1]), ), (v, ))
            b = (((m_list[1:], n_list[1:]), ), (1, ))
            return _add(_mul(a, _D(b)), _mul(_D(a), b))
    else:
        return _add(_D((t_list[:1], v_list[:1])), _D((t_list[1:], v_list[1:])))


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
        rise = _add(_half,
                    _mul(_half, _basic_wave(COS, pi / edge, shift=0.5 * edge)))
        return Waveform(bounds=(round(-edge / 2,
                                      NDIGITS), round(edge / 2,
                                                      NDIGITS), +inf),
                        seq=(_zero, rise, _one))
    elif type == 'linear':
        rise = _add(_half, _mul(_const(1 / edge), _basic_wave(LINEAR)))
        return Waveform(bounds=(round(-edge / 2,
                                      NDIGITS), round(edge / 2,
                                                      NDIGITS), +inf),
                        seq=(_zero, rise, _one))
    else:
        std_sq2 = edge / 5
        # rise = _add(_half, _mul(_half, _basic_wave(ERF, std_sq2)))
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
                        seq=(_zero, _basic_wave(GAUSSIAN, std_sq2), _zero))
    else:
        return Waveform(bounds=(round(-0.75 * width - 0.5 * plateau,
                                      NDIGITS), round(-0.5 * plateau, NDIGITS),
                                round(0.5 * plateau, NDIGITS),
                                round(0.75 * width + 0.5 * plateau,
                                      NDIGITS), +inf),
                        seq=(_zero,
                             _basic_wave(GAUSSIAN,
                                         std_sq2,
                                         shift=-0.5 * plateau), _one,
                             _basic_wave(GAUSSIAN,
                                         std_sq2,
                                         shift=0.5 * plateau), _zero))


def cos(w, phi=0):
    if w == 0:
        return const(np.cos(phi))
    if w < 0:
        phi = -phi
        w = -w
    return Waveform(seq=(_basic_wave(COS, w, shift=-phi / w), ))


def sin(w, phi=0):
    if w == 0:
        return const(np.sin(phi))
    if w < 0:
        phi = -phi + pi
        w = -w
    return Waveform(seq=(_basic_wave(COS, w, shift=(pi / 2 - phi) / w), ))


def exp(alpha):
    if isinstance(alpha, complex):
        if alpha.real == 0:
            return cos(alpha.imag) + 1j * sin(alpha.imag)
        else:
            return exp(alpha.real) * (cos(alpha.imag) + 1j * sin(alpha.imag))
    else:
        return Waveform(seq=(_basic_wave(EXP, alpha), ))


def sinc(bw):
    if bw <= 0:
        return zero()
    width = 100 / bw
    return Waveform(bounds=(round(-0.5 * width,
                                  NDIGITS), round(0.5 * width, NDIGITS), +inf),
                    seq=(_zero, _basic_wave(SINC, bw), _zero))


def cosPulse(width, plateau=0.0):
    # cos = _basic_wave(COS, 2*np.pi/width)
    # pulse = _mul(_add(cos, _one), _half)
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
    return Waveform(seq=(_basic_wave(COSH, w), ))


def sinh(w):
    return Waveform(seq=(_basic_wave(SINH, w), ))


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
                             _basic_wave(DRAG, t0, freq, width, delta,
                                         block_freq, phase), _zero),
                        bounds=(round(t0, NDIGITS), round(t0 + width,
                                                          NDIGITS), +inf))
    elif width <= 0:
        w = 2 * pi * (freq + delta)
        return Waveform(
            seq=(_zero,
                 _basic_wave(COS, w,
                             shift=(phase + 2 * pi * delta * t0) / w), _zero),
            bounds=(round(t0, NDIGITS), round(t0 + plateau, NDIGITS), +inf))
    else:
        w = 2 * pi * (freq + delta)
        return Waveform(
            seq=(_zero,
                 _basic_wave(DRAG, t0, freq, width, delta, block_freq, phase),
                 _basic_wave(COS, w, shift=(phase + 2 * pi * delta * t0) / w),
                 _basic_wave(DRAG, t0 + plateau, freq, width, delta,
                             block_freq,
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
                        seq=(_zero, _basic_wave(LINEARCHIRP, f0, f1, T,
                                                phi0), _zero))
    elif type in ['exp', 'exponential', 'geometric']:
        # f(t) = f0 * (f1/f0) ** (t/T)
        if f0 == 0:
            raise ValueError('f0 must be non-zero')
        alpha = np.log(f1 / f0) / T
        return Waveform(bounds=(0, round(T, NDIGITS), +inf),
                        seq=(_zero,
                             _basic_wave(EXPONENTIALCHIRP, f0, alpha,
                                         phi0), _zero))
    elif type in ['hyperbolic', 'hyp']:
        # f(t) = f0 * f1 / (f0 * (t/T) + f1 * (1-t/T))
        if f0 * f1 == 0:
            return const(np.sin(phi0))
        k = (f0 - f1) / (f1 * T)
        return Waveform(bounds=(0, round(T, NDIGITS), +inf),
                        seq=(_zero, _basic_wave(HYPERBOLICCHIRP, f0, k,
                                                phi0), _zero))
    else:
        raise ValueError(f'unknown type {type}')


def interp(x, y):
    seq, bounds = [_zero], [x[0]]
    for x1, x2, y1, y2 in zip(x[:-1], x[1:], y[:-1], y[1:]):
        if x2 == x1:
            continue
        seq.append(
            _add(
                _mul(_const((y2 - y1) / (x2 - x1)),
                     _basic_wave(LINEAR, shift=x1)), _const(y1)))
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
    seq = (_basic_wave(TYPEID, *args), )
    wav = Waveform(seq=seq)
    if start is not None:
        wav = wav * (step(0) >> start)
    if stop is not None:
        wav = wav * ((1 - step(0)) >> stop)
    return wav


def samplingPoints(start, stop, points):
    return Waveform(bounds=(round(start, NDIGITS), round(stop, NDIGITS), inf),
                    seq=(_zero, _basic_wave(INTERP, start, stop,
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
