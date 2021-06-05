import math
from bisect import bisect_left
from itertools import chain, product

import numpy as np
import scipy.special as special

_zero = ((), ())


def _const(c):
    return (((), ()), ), (c, )


_one = _const(1)
_half = _const(1 / 2)
_two = _const(2)
_pi = _const(np.pi)
_two_pi = _const(2 * np.pi)
_half_pi = _const(np.pi / 2)


def _is_const(x):
    return x == _zero or len(x[0]) == 1 and x[0][0][0] == ()


def _basic_wave(Type, *args, shift=0):
    return ((((Type, shift, *args), ), (1, )), ), (1, )


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
    if x == _zero:
        return y
    if y == _zero:
        return x
    x, y = (x, y) if len(x[0]) >= len(y[0]) else (y, x)
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
        for mt in pre_mtlist:
            mtlist.append((mt[0], mt[1] + time, *mt[2:]))
        t_list.append((tuple(mtlist), nlist))
    return tuple(t_list), x[1]


def _pow(x, n):
    if x == _zero:
        return _zero
    if n == 0:
        return _one
    if _is_const(x):
        return _const(x[1][0]**n)

    t_list, v_list = [], []

    for (mtlist, pre_nlist), v in zip(*x):
        nlist = []
        for m in pre_nlist:
            nlist.append(n * m)
        t_list.append((mtlist, tuple(nlist)))
        v_list.append(v**n)
    return tuple(t_list), tuple(v_list)


def _cos_power_n(x, n):
    _, shift, w = x
    ret = _zero
    for k in range(0, n // 2 + 1):
        if n == 2 * k:
            a = _const(math.comb(n, k) / 2**n)
            ret = _add(ret, a)
        else:
            expr = (((((COS, shift, (n - 2 * k) * w), ), (1, )), ),
                    (math.comb(n, k) / 2**(n - 1), ))
            ret = _add(ret, expr)
    return ret


def _trigMul_t(x, y, v):
    """cos(a)cos(b) = 0.5*cos(a+b)+0.5*cos(a-b)"""
    _, t1, w1 = x
    _, t2, w2 = y
    if w2 > w1:
        t1, t2 = t2, t1
        w1, w2 = w2, w1
    exp1 = (COS, (w1 * t1 + w2 * t2) / (w1 + w2), w1 + w2)
    if w1 == w2:
        return (((exp1, ), (1, )), ), (0.5 * v, )
    else:
        exp2 = (COS, (w1 * t1 - w2 * t2) / (w1 - w2), w1 - w2)
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
            expr = _mul(tmp, (((trig[0], ), (1, )), (v, )))
        elif len(trig) == 2:
            expr = _trigMul_t(trig[0], trig[1], v)
            expr = _mul(tmp, expr)
        else:
            expr = _mul(tmp, _const(v))
        ret = _add(ret, expr)
    return ret


def _trigReduce(mtlist, v):
    trig = _one
    ml, nl = [], []
    for mt, n in zip(*mtlist):
        if mt[0] == COS:
            trig = _trigMul(trig, _cos_power_n(mt, n))
        else:
            ml.append(mt)
            nl.append(n)
    ret = (((tuple(ml), tuple(nl)), ), (v, ))

    return _mul(ret, trig)


def _simplify(expr):
    ret = _zero
    for t, v in zip(*expr):
        y = _trigReduce(t, v)
        ret = _add(ret, y)
    return ret


def _filter(expr, low, high):
    expr = _simplify(expr)
    ret = _zero
    for t, v in zip(*expr):
        for i, (mt, n) in enumerate(zip(*t)):
            if mt[0] == COS:
                if low <= mt[2] < high:
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


def _apply(x, Type, shift, *args):
    return _baseFunc[Type](x - shift, *args)


def _calc(wav, x):
    lru_cache = {}

    def _calc_m(t, x):
        ret = 1
        for mt, n in zip(*t):
            if mt not in lru_cache:
                lru_cache[mt] = _apply(x, *mt)
            ret = ret * lru_cache[mt]**n
        return ret

    ret = 0
    for t, v in zip(*wav):
        ret = ret + v * _calc_m(t, x)
    return ret


class Waveform:
    __slots__ = ('bounds', 'seq')

    def __init__(self, bounds=(+np.inf, ), seq=(_zero, )):
        self.bounds = bounds
        self.seq = seq

    def simplify(self):
        seq = []
        for expr in self.seq:
            seq.append(_simplify(expr))
        return Waveform(self.bounds, tuple(seq))

    def filter(self, low=0, high=np.inf):
        seq = []
        for expr in self.seq:
            seq.append(_filter(expr, low, high))
        return Waveform(self.bounds, tuple(seq))

    def _comb(self, other, oper):
        bounds = []
        seq = []
        i1, i2 = 0, 0
        h1, h2 = len(self.bounds), len(other.bounds)
        while i1 < h1 or i2 < h2:
            s = oper(self.seq[i1], other.seq[i2])
            b = min(self.bounds[i1], other.bounds[i2])
            # if s == seq[-1]:
            #     bounds[-1] = b
            # else:
            #     bounds.append(b)
            #     seq.append(s)
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

    def append(self, other):
        if not isinstance(other, Waveform):
            raise TypeError('connect Waveform by other type')
        if len(self.bounds) == 1:
            self.bounds = other.bounds
            self.seq = self.seq + other.seq[1:]
            return

        assert self.bounds[-2] <= other.bounds[
            0], f"connect waveforms with overlaped domain {self.bounds}, {other.bounds}"
        if self.bounds[-2] < other.bounds[0]:
            self.bounds = self.bounds[:-1] + other.bounds
            self.seq = self.seq + other.seq[1:]
        else:
            self.bounds = self.bounds[:-2] + other.bounds
            self.seq = self.seq[:-1] + other.seq[1:]

    def __ior__(self, other):
        self.append(other)
        return self

    def __or__(self, other):
        w = Waveform(self.bounds, self.seq)
        w.append(other)
        return w

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
        return Waveform(tuple(bound + time for bound in self.bounds),
                        tuple(_shift(expr, time) for expr in self.seq))

    def __lshift__(self, time):
        return self >> (-time)

    def __call__(self, x, frag=False):
        range_list = np.searchsorted(x, self.bounds)
        #ret = np.zeros_like(x)
        ret = []
        start, stop = 0, 0
        for i, stop in enumerate(range_list):
            if start < stop and self.seq[i] != _zero:
                #ret[start:stop] = _calc(self.seq[i], x[start:stop])
                ret.append((start, stop, _calc(self.seq[i], x[start:stop])))
            start = stop
        if not frag:
            y = np.zeros_like(x)
            for start, stop, part in ret:
                y[start:stop] = part
            return y
        else:
            return ret

    def __hash__(self):
        return hash((self.bounds, self.seq))

    def __eq__(self, o: object) -> bool:
        if isinstance(o, (int, float, complex)):
            return self == const(o)
        elif isinstance(o, Waveform):
            a = self.simplify()
            b = o.simplify()
            return a.seq == b.seq and a.bounds == b.bounds
        else:
            return False


_zero_waveform = Waveform()
_one_waveform = Waveform(seq=(_one, ))


def zero():
    return _zero_waveform


def one():
    return _one_waveform


def const(c):
    return Waveform(seq=(_const(c), ))


__TypeIndex = 1
_baseFunc = {}
_derivativeBaseFunc = {}


def registerBaseFunc(func):
    global __TypeIndex
    Type = __TypeIndex
    __TypeIndex += 1

    _baseFunc[Type] = func

    return Type


def registerDerivative(Type, dFunc):
    _derivativeBaseFunc[Type] = dFunc


# register base function
LINEAR = registerBaseFunc(lambda t: t)
GAUSSIAN = registerBaseFunc(lambda t, std_sq2: np.exp(-(t / std_sq2)**2))
ERF = registerBaseFunc(lambda t, std_sq2: special.erf(t / std_sq2))
COS = registerBaseFunc(lambda t, w: np.cos(w * t))
SINC = registerBaseFunc(lambda t, bw: np.sinc(bw * t))
EXP = registerBaseFunc(lambda t, alpha: np.exp(alpha * t))

# register derivative
registerDerivative(LINEAR, lambda shift, *args: _one)

registerDerivative(
    GAUSSIAN, lambda shift, *args: (((((LINEAR, shift),
                                       (GAUSSIAN, shift, *args)), (1, 1)), ),
                                    (-2 / args[0]**2, )))

registerDerivative(
    ERF, lambda shift, *args: (((((GAUSSIAN, shift, *args), ), (1, )), ),
                               (2 / args[0] / np.sqrt(np.pi), )))

registerDerivative(
    COS, lambda shift, *args:
    (((((COS, shift - np.pi / args[0] / 2, args[0]), ), (1, )), ),
     (args[0], )))

registerDerivative(
    SINC, lambda shift, *args:
    (((((LINEAR, shift), (COS, shift, *args)), (-1, 1)),
      (((LINEAR, shift), (COS, shift, args[0], args[1] - np.pi / 2)),
       (-2, 1))), (1, -1 / args[0])))

registerDerivative(
    EXP, lambda shift, *args: (((((EXP, shift, *args), ), (1, )), ),
                               (args[0], )))


def _D_base(m):
    Type, shift, *args = m
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
    return Waveform(bounds=(0, +np.inf), seq=(_const(-1), _one))


def step(edge, type='erf'):
    """
    type: "erf", "cos", "linear"
    """
    if type == 'cos':
        rise = _add(
            _half, _mul(_half, _basic_wave(COS, np.pi / edge,
                                           shift=0.5 * edge)))
        return Waveform(bounds=(-edge / 2, edge / 2, +np.inf),
                        seq=(_zero, rise, _one))
    elif type == 'linear':
        rise = _add(_half, _mul(_const(1 / edge), _basic_wave(LINEAR)))
        return Waveform(bounds=(-edge / 2, edge / 2, +np.inf),
                        seq=(_zero, rise, _one))
    else:
        std_sq2 = edge / 5
        # rise = _add(_half, _mul(_half, _basic_wave(ERF, std_sq2)))
        rise = ((((), ()), (((ERF, 0, std_sq2), ), (1, ))), (0.5, 0.5))
        return Waveform(bounds=(-edge, edge, +np.inf), seq=(_zero, rise, _one))


def square(width, edge=0, type='erf'):
    if width <= 0:
        return zero()
    if edge == 0:
        return Waveform(bounds=(-0.5 * width, 0.5 * width, +np.inf),
                        seq=(_zero, _one, _zero))
    else:
        return ((step(edge, type=type) << width / 2) -
                (step(edge, type=type) >> width / 2))


def gaussian(width):
    if width <= 0:
        return zero()
    # width is two times FWHM
    # std_sq2 = width / (4 * np.sqrt(np.log(2)))
    std_sq2 = width / 3.3302184446307908
    # std is set to give total pulse area same as a square
    #std_sq2 = width/np.sqrt(np.pi)
    return Waveform(bounds=(-0.75 * width, 0.75 * width, +np.inf),
                    seq=(_zero, _basic_wave(GAUSSIAN, std_sq2), _zero))


def cos(w, phi=0):
    if w == 0:
        return const(np.cos(phi))
    if w < 0:
        phi -= np.pi
        w = -w
    return Waveform(seq=(_basic_wave(COS, w, shift=-phi / w), ))


def sin(w, phi=0):
    if w == 0:
        return const(np.sin(phi))
    if w < 0:
        phi -= np.pi
        w = -w
    return Waveform(seq=(_basic_wave(COS, w, shift=(np.pi / 2 - phi) / w), ))


def exp(alpha):
    return Waveform(seq=(_basic_wave(EXP, alpha), ))


def sinc(bw):
    if bw <= 0:
        return zero()
    width = 100 / bw
    return Waveform(bounds=(-0.5 * width, 0.5 * width, +np.inf),
                    seq=(_zero, _basic_wave(SINC, bw), _zero))


def cosPulse(width):
    # cos = _basic_wave(COS, 2*np.pi/width)
    # pulse = _mul(_add(cos, _one), _half)
    if width <= 0:
        return zero()
    pulse = ((((), ()), (((COS, 0, 6.283185307179586 / width), ), (1, ))),
             (0.5, 0.5))
    return Waveform(bounds=(-0.5 * width, 0.5 * width, +np.inf),
                    seq=(_zero, pulse, _zero))


def _poly(*a):
    """
    a[0] + a[1] * t + a[2] * t**2 + ...
    """
    t = (((), ()), *[(((LINEAR, 0), ), (n, ))
                     for n, _ in enumerate(a[1:], start=1)])
    return t, a


def poly(a):
    """
    a[0] + a[1] * t + a[2] * t**2 + ...
    """
    return Waveform(seq=(_poly(*a), ))


def mixing(I,
           Q=None,
           *,
           phase=0.0,
           freq=0.0,
           ratioIQ=1.0,
           phaseDiff=0.0,
           DRAGScaling=None):
    """SSB or envelope mixing
    """
    if Q is None:
        I = I
        Q = zero()

    w = 2 * np.pi * freq
    if freq != 0.0:
        # SSB mixing
        Iout = I * cos(w, -phase) + Q * sin(w, -phase)
        Qout = -I * sin(w, -phase + phaseDiff) + Q * cos(w, -phase + phaseDiff)
    else:
        # envelope mixing
        Iout = I * np.cos(-phase) + Q * np.sin(-phase)
        Qout = -I * np.sin(-phase) + Q * np.cos(-phase)

    if DRAGScaling is not None and DRAGScaling != 0:
        # apply DRAG
        I = (1 - w * DRAGScaling) * Iout - DRAGScaling * D(Qout)
        Q = (1 - w * DRAGScaling) * Qout + DRAGScaling * D(Iout)
        Iout, Qout = I, Q

    Qout = ratioIQ * Qout

    return Iout, Qout


__all__ = [
    'D', 'Waveform', 'const', 'cos', 'cosPulse', 'exp', 'gaussian', 'mixing',
    'one', 'poly', 'registerBaseFunc', 'registerDerivative', 'sign', 'sin',
    'square', 'step', 'zero'
]
