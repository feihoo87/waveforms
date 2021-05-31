from collections import defaultdict
from functools import wraps
from inspect import signature


class Scope():
    def __init__(self, parent=None):
        self.ns = {}
        self.parent = parent

    def get(self, name):
        if name in self.ns:
            return self.ns[name]
        elif self.parent is None:
            raise KeyError(f'{name} not defined.')
        else:
            return self.parent.get(name)

    def local(self, name):
        return self.ns[name]

    def set(self, name, value):
        self.ns[name] = value

    def scope(self):
        return Scope(parent=self)

    def __contains__(self, name):
        return (name in self.ns
                or self.parent is not None and self.parent.__contains__(name))


def gate(qnum=1, name=None, scope=None):
    def decorator(func, name=name):
        if name is None:
            name = func.__name__
        sig = signature(func)
        anum = len(sig.parameters) - 1
        if 'scope' in sig.parameters:
            anum -= 1

        @wraps(func)
        def wrapper(qubits, *args, scope=scope, **kwds):
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

        scope.set(name, wrapper)
        return wrapper

    return decorator


def opaque(name, type='default', qnum=1, scope=None):
    def decorator(func, name=name):
        if name is None:
            name = func.__name__
        sig = signature(func)
        anum = len(sig.parameters) - 1
        if 'scope' in sig.parameters:
            anum -= 1

        @wraps(func)
        def wrapper(qubits, *args, scope=scope, **kwds):
            if 'scope' in sig.parameters:
                kwds['scope'] = scope
            if isinstance(qubits, int):
                assert qnum == 1, f"gate {name} should be {qnum}-qubit gate, but 1 were given."
            else:
                assert len(
                    qubits
                ) == qnum, f"gate {name} should be {qnum}-qubit gate, but {len(qubits)} were given."
            assert len(args) + len(
                kwds
            ) == anum, f"gate {name} should be {anum}-arguments gate, but {len(args)+len(kwds)} were given."
            return func(qubits, *args, **kwds)

        if name in scope.ns:
            scope.ns[name][type] = wrapper
        else:
            scope.ns[name] = {type: wrapper}
        return wrapper

    return decorator


class Library():
    def __init__(self):
        self.gates = Scope()
        self.opaques = Scope()
        self.qasmLib = {}

    def gate(self, qnum=1, name=None):
        return gate(qnum=qnum, name=name, scope=self.gates)

    def opaque(self, name, type='default', qnum=1):
        return opaque(name, type=type, qnum=qnum, scope=self.opaques)
