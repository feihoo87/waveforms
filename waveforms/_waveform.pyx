import pickle
from bisect import bisect_left
from itertools import product

import numpy as np
from numpy import e, inf, pi

NDIGITS = 15
__TypeIndex = 1
_baseFunc = {}
_derivativeBaseFunc = {}
_baseFunc_latex = {}

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

    return tuple(bounds), tuple(seq)


def _merge_waveform(b1, s1, b2, s2, oper):
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
    
