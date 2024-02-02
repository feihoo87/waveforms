import pickle
from bisect import bisect_left
from itertools import chain, product

import numpy as np
import scipy.special as special
from numpy import e, inf, pi

NDIGITS = 15
__TypeIndex = 1
_baseFunc = {}
_derivativeBaseFunc = {}
_baseFunc_latex = {}

_zero = ((), ())


cdef int comb(int n, int k):
    if k > n:
        return 0
    if k > n // 2:
        k = n - k
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return c


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


def is_const(x):
    return x == _zero or x[0] == (((), ()), )


def basic_wave(Type, *args, shift=0):
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


def mul(x, y):
    t_list, v_list = [], []
    xt_list, xv_list = x
    yt_list, yv_list = y
    lo, hi = 0, 0
    for (t1, t2), (v1, v2) in zip(product(xt_list, yt_list),
                                  product(xv_list, yv_list)):
        if v1 * v2 == 0:
            continue
        t = add(t1, t2)
        lo, hi = _insert_type_value_pair(t_list, v_list, t, v1 * v2, lo, hi)
    return tuple(t_list), tuple(v_list)


def add(x, y):
    # x, y = (x, y) if len(x[0]) >= len(y[0]) else (y, x)
    t_list, v_list = list(x[0]), list(x[1])
    lo, hi = 0, len(t_list)
    for t, v in zip(*y):
        lo, hi = _insert_type_value_pair(t_list, v_list, t, v, lo, hi)
    return tuple(t_list), tuple(v_list)


def shift(x, time):
    if is_const(x):
        return x

    t_list = []

    for pre_mtlist, nlist in x[0]:
        mtlist = []
        for Type, *args, shift in pre_mtlist:
            mtlist.append((Type, *args, shift + time))
        t_list.append((tuple(mtlist), nlist))
    return tuple(t_list), x[1]


def pow(x, n):
    if x == _zero:
        return _zero
    if n == 0:
        return _one
    if is_const(x):
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
            ret = mul(ret, x)
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


def calc_parts(bounds, seq, x, function_lib, min=-inf, max=inf):
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


def wave_sum(waves):
    if not waves:
        return ((+inf, ), (_zero, ))

    bounds, seq = waves[0]
    if not waves[1:]:
        return bounds, seq
    bounds, seq = list(bounds), list(seq)

    for bounds_, seq_ in waves[1:]:
        if len(bounds_) == 1:
            for i, s in enumerate(seq):
                seq[i] = add(s, seq_[0])
        elif len(bounds) == 1:
            bounds = list(bounds_)
            seq = [add(seq[0], s) for s in seq_]
        else:
            lo = 0
            for b, s in zip(bounds_, seq_):
                i = bisect_left(bounds, b, lo=lo)
                if bounds[i] > b:
                    bounds.insert(i, b)
                    if i == 0:
                        seq.insert(i, s)
                    else:
                        seq.insert(i, add(s, seq[i]))
                    up = i - 1
                else:
                    up = i
                for j in range(lo + 1, up + 1):
                    seq[j] = add(seq[j], s)
                lo = i

    i = 0
    while i < len(bounds) - 1:
        if seq[i] == seq[i + 1]:
            del seq[i]
            del bounds[i]
        else:
            i += 1

    return tuple(bounds), tuple(seq)


def merge_waveform(b1, s1, b2, s2, oper):
    bounds = []
    seq = []
    i1 = 0
    i2 = 0
    h1 = len(b1)
    h2 = len(b2)
    while i1 < h1 or i2 < h2:
        s = oper(s1[i1], s2[i2])
        b = min(b1[i1], b2[i2])
        if seq and s == seq[-1]:
            bounds[-1] = b
        else:
            bounds.append(b)
            seq.append(s)
        if b == b1[i1]:
            i1 += 1
        if b == b2[i2]:
            i2 += 1
    return tuple(bounds), tuple(seq)


def _D_base(m):
    Type, *args, shift = m
    return _derivativeBaseFunc[Type](shift, *args)


def _D(x):
    if is_const(x):
        return _zero
    t_list, v_list = x
    if len(v_list) == 1:
        (m_list, n_list), v = t_list[0], v_list[0]
        if len(m_list) == 1:
            m, n = m_list[0], n_list[0]
            if n == 1:
                return mul(_D_base(m), _const(v))
            else:
                return mul(((((m, ), (n - 1, )), ), (n * v, )),
                           _D(((((m, ), (1, )), ), (1, ))))
        else:
            a = (((m_list[:1], n_list[:1]), ), (v, ))
            b = (((m_list[1:], n_list[1:]), ), (1, ))
            return add(mul(a, _D(b)), mul(_D(a), b))
    else:
        return add(_D((t_list[:1], v_list[:1])), _D((t_list[1:], v_list[1:])))


def registerBaseFunc(func):
    global __TypeIndex
    Type = __TypeIndex
    __TypeIndex += 1

    _baseFunc[Type] = func

    return Type


def packBaseFunc():
    return pickle.dumps(_baseFunc)


def updateBaseFunc(buf):
    _baseFunc.update(pickle.loads(buf))


def registerDerivative(Type, dFunc):
    _derivativeBaseFunc[Type] = dFunc


def registerBaseFuncLatex(Type, dFunc):
    _baseFunc_latex[Type] = dFunc


def _LINEAR(t):
    return t


def _GAUSSIAN(t, std_sq2):
    return np.exp(-(t / std_sq2)**2)


def _ERF(t, std_sq2):
    return special.erf(t / std_sq2)


def _COS(t, w):
    return np.cos(w * t)


def _SINC(t, bw):
    return np.sinc(bw * t)


def _EXP(t, alpha):
    return np.exp(alpha * t)


def _INTERP(t, start, stop, points):
    return np.interp(t, np.linspace(start, stop, len(points)), points)


def _LINEARCHIRP(t, f0, f1, T, phi0):
    return np.sin(phi0 + 2 * np.pi * ((f1 - f0) / (2 * T) * t**2 + f0 * t))


def _EXPONENTIALCHIRP(t, f0, alpha, phi0):
    return np.sin(phi0 + 2 * pi * f0 * (np.exp(alpha * t) - 1) / alpha)


def _HYPERBOLICCHIRP(t, f0, k, phi0):
    return np.sin(phi0 + 2 * np.pi * f0 / k * np.log(1 + k * t))


def _COSH(t, w):
    return np.cosh(w * t)


def _SINH(t, w):
    return np.sinh(w * t)


def _drag(t: np.ndarray, t0: float, freq: float, width: float, delta: float,
          block_freq: float | None, phase: float):

    o = np.pi / width
    Omega_x = np.sin(o * (t - t0))**2
    wt = 2 * np.pi * (freq + delta) * t - (2 * np.pi * delta * t0 + phase)

    if block_freq is None or block_freq - delta == 0:
        return Omega_x * np.cos(wt)

    b = 1 / np.pi / 2 / (block_freq - delta)
    Omega_y = -b * o * np.sin(2 * o * (t - t0))

    return Omega_x * np.cos(wt) + Omega_y * np.sin(wt)


LINEAR = registerBaseFunc(_LINEAR)
GAUSSIAN = registerBaseFunc(_GAUSSIAN)
ERF = registerBaseFunc(_ERF)
COS = registerBaseFunc(_COS)
SINC = registerBaseFunc(_SINC)
EXP = registerBaseFunc(_EXP)
INTERP = registerBaseFunc(_INTERP)
LINEARCHIRP = registerBaseFunc(_LINEARCHIRP)
EXPONENTIALCHIRP = registerBaseFunc(_EXPONENTIALCHIRP)
HYPERBOLICCHIRP = registerBaseFunc(_HYPERBOLICCHIRP)
COSH = registerBaseFunc(_COSH)
SINH = registerBaseFunc(_SINH)
DRAG = registerBaseFunc(_drag)


def _d_LINEAR(shift, *args):
    return _one


def _d_GAUSSIAN(shift, *args):
    return (((((LINEAR, shift), (GAUSSIAN, *args, shift)), (1, 1)), ),
            (-2 / args[0]**2, ))


def _d_ERF(shift, *args):
    return (((((GAUSSIAN, *args, shift), ), (1, )), ),
            (2 / args[0] / np.sqrt(pi), ))


def _d_COS(shift, *args):
    return (((((COS, args[0], shift - pi / args[0] / 2), ), (1, )), ),
            (args[0], ))


def _d_SINC(shift, *args):
    return (((((LINEAR, shift), (COS, *args, shift)), (-1, 1)),
             (((LINEAR, shift), (COS, args[0], args[1] - pi / 2, shift)),
              (-2, 1))), (1, -1 / args[0]))


def _d_EXP(shift, *args):
    return (((((EXP, *args, shift), ), (1, )), ), (args[0], ))


def _d_INTERP(shift, start, stop, points):
    return (((((INTERP, start, stop, tuple(np.gradient(np.asarray(points))),
                shift), ), (1, )), ), ((len(points) - 1) / (stop - start), ))


def _d_COSH(shift, *args):
    return (((((SINH, *args, shift), ), (1, )), ), (args[0], ))


def _d_SINH(shift, *args):
    return (((((COSH, *args, shift), ), (1, )), ), (args[0], ))


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


def _d_EXPONENTIALCHIRP(shift, f0, alpha, phi0):
    return (((((EXP, alpha, shift), (EXPONENTIALCHIRP, f0, alpha,
                                     phi0 + pi / 2, shift)), (1, 1)), ),
            (2 * pi * f0, ))


def _d_HYPERBOLICCHIRP(shift, f0, k, phi0):
    return (((((LINEAR, shift - 1 / k), (HYPERBOLICCHIRP, f0, k, phi0 + pi / 2,
                                         shift)), (-1, 1)), ), (2 * pi * f0, ))


# register derivative
registerDerivative(LINEAR, _d_LINEAR)
registerDerivative(GAUSSIAN, _d_GAUSSIAN)
registerDerivative(ERF, _d_ERF)
registerDerivative(COS, _d_COS)
registerDerivative(SINC, _d_SINC)
registerDerivative(EXP, _d_EXP)
registerDerivative(INTERP, _d_INTERP)
registerDerivative(COSH, _d_COSH)
registerDerivative(SINH, _d_SINH)
registerDerivative(LINEARCHIRP, _d_LINEARCHIRP)
registerDerivative(EXPONENTIALCHIRP, _d_EXPONENTIALCHIRP)
registerDerivative(HYPERBOLICCHIRP, _d_HYPERBOLICCHIRP)


def _cos_power_n(x, n):
    _, w, shift = x
    ret = _zero
    for k in range(0, n // 2 + 1):
        if n == 2 * k:
            a = _const(comb(n, k) / 2**n)
            ret = add(ret, a)
        else:
            expr = (((((COS, (n - 2 * k) * w, shift), ), (1, )), ),
                    (comb(n, k) / 2**(n - 1), ))
            ret = add(ret, expr)
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
    if is_const(x) or is_const(y):
        return mul(x, y)
    ret = _zero
    for (t1, t2), (v1, v2) in zip(product(x[0], y[0]), product(x[1], y[1])):
        v = v1 * v2
        tmp = _one
        trig = []
        for mt, n in zip(chain(t1[0], t2[0]), chain(t1[1], t2[1])):
            if mt[0] == COS:
                trig.append(mt)
            else:
                tmp = mul(tmp, ((((mt, ), (n, )), ), (1, )))
        if len(trig) == 1:
            x = ((((trig[0], ), (1, )), ), (v, ))
            expr = mul(tmp, x)
        elif len(trig) == 2:
            expr = _trigMul_t(trig[0], trig[1], v)
            expr = mul(tmp, expr)
        else:
            expr = mul(tmp, _const(v))
        ret = add(ret, expr)
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
        ret = mul(ret, basic_wave(EXP, alpha, shift=shift))

    return mul(ret, trig)


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


def simplify(expr, eps):
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
                ret = add(ret, ((t, ), (v_r, )))
            else:
                ret = add(ret, ((t, ), (v_r + 1j * v_i, )))
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

            expr = mul(((t, ), (1, )), expr)
            ret = add(ret, expr)
    return ret


def filter(expr, low, high, eps):
    expr = simplify(expr, eps)
    ret = _zero
    for t, v in zip(*expr):
        for i, (mt, n) in enumerate(zip(*t)):
            if mt[0] == COS:
                if low <= mt[1] < high:
                    ret = add(ret, ((t, ), (v, )))
                break
            elif mt[0] == SINC and n == 1:
                pass
            elif mt[0] == GAUSSIAN and n == 1:
                pass
        else:
            if low <= 0:
                ret = add(ret, ((t, ), (v, )))
    return ret
