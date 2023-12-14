import inspect
from concurrent.futures import Future
from types import MappingProxyType

import numpy as np
import scipy.sparse as sp


def call_func_with_kwds(func, args, kwds, log=None):
    funcname = getattr(func, '__name__', repr(func))
    sig = inspect.signature(func)
    for p in sig.parameters.values():
        if p.kind == p.VAR_KEYWORD:
            return func(*args, **kwds)
    kw = {
        k: v
        for k, v in kwds.items()
        if k in list(sig.parameters.keys())[len(args):]
    }
    try:
        args = [
            arg.result() if isinstance(arg, Future) else arg for arg in args
        ]
        kw = {
            k: v.result() if isinstance(v, Future) else v
            for k, v in kw.items()
        }
        return func(*args, **kw)
    except:
        if log:
            log.exception(f'Call {funcname} with {args} and {kw}')
        raise
    finally:
        if log:
            log.debug(f'Call {funcname} with {args} and {kw}')


def try_to_call(x, args, kwds, log=None):
    if callable(x):
        return call_func_with_kwds(x, args, kwds, log)
    return x


def freeze(x):
    """Freeze a mutable object.
    """
    if isinstance(x, (int, float, complex, str, bytes, type(None))):
        pass
    elif isinstance(x, (list, tuple)):
        return tuple([freeze(y) for y in x])
    elif isinstance(x, dict):
        return MappingProxyType({k: freeze(v) for k, v in x.items()})
    elif isinstance(x, set):
        return frozenset([freeze(y) for y in x])
    elif isinstance(x, (np.ndarray, np.matrix)):
        x.flags.writeable = False
    elif isinstance(x, sp.spmatrix):
        x.data.flags.writeable = False
        if x.format in {'csr', 'csc', 'bsr'}:
            x.indices.flags.writeable = False
            x.indptr.flags.writeable = False
        elif x.format == 'coo':
            x.row.flags.writeable = False
            x.col.flags.writeable = False
    elif isinstance(x, bytearray):
        x = bytes(x)
    return x
