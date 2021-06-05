from collections import defaultdict
from functools import wraps
from inspect import signature
from typing import Optional

from waveforms.waveform import zero

from .config import Config


def gate(qnum: int = 1, name: Optional[str] = None, scope: dict = None):
    def decorator(func: callable, name: str = name):
        if name is None:
            name = func.__name__
        sig = signature(func)
        anum = len(sig.parameters) - 1
        if 'scope' in sig.parameters:
            anum -= 1

        @wraps(func)
        def wrapper(qubits, *args, scope: dict = scope, **kwds):
            if isinstance(qubits, int):
                if qnum != 1:
                    raise TypeError(
                        f"gate {name} should be {qnum}-qubit gate, but 1 were given."
                    )
            elif len(qubits) != qnum:
                raise TypeError(
                    f"gate {name} should be {qnum}-qubit gate, but {len(qubits)} were given."
                )
            if len(args) + len(kwds) != anum:
                raise TypeError(
                    f"gate {name} should be {anum}-arguments gate, but {len(args)+len(kwds)} were given."
                )
            if 'scope' in sig.parameters:
                kwds['scope'] = scope
            return func(qubits, *args, **kwds)

        scope[name] = wrapper
        return wrapper

    return decorator


class _ChannelGetter():
    def __init__(self, ctx):
        self.ctx = ctx

    def __getitem__(self, key):
        return self.ctx.raw_waveforms.__getitem__(key)

    def __setitem__(self, key, wav):
        self.ctx.raw_waveforms.__setitem__(key, wav)


class Context():
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.time = defaultdict(lambda: 0)
        self.waveforms = defaultdict(zero)
        self.raw_waveforms = defaultdict(zero)
        self.measures = defaultdict(list)
        self.phases = defaultdict(lambda: 0)

    @property
    def channel(self):
        return _ChannelGetter(self)


def opaque(name: str, type: str = 'default', scope: dict = None):
    def decorator(func: callable, name: str = name):
        sig = signature(func)

        @wraps(func)
        def wrapper(ctx: Context, qubits, *args, **kwds):
            if 'scope' in sig.parameters:
                kwds['scope'] = scope
            return func(ctx, qubits, *args, **kwds)

        scope[name][type] = wrapper
        return wrapper

    return decorator


class Library():
    def __init__(self):
        self.gates = {}
        self.opaques = defaultdict(dict)
        self.qasmLib = {}

    def gate(self, qnum=1, name=None):
        return gate(qnum=qnum, name=name, scope=self.gates)

    def opaque(self, name: str, type: str = 'default'):
        return opaque(name, type=type, scope=self.opaques)


def libraries(*libs: Library) -> Library:
    ret = Library()
    for lib in libs:
        ret.gates.update(lib.gates)
        ret.opaques.update(lib.opaques)
        ret.qasmLib.update(lib.qasmLib)
    return ret
