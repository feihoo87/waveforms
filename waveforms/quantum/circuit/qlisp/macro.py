import weakref
from functools import wraps

from numpy import pi


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

    def __contains__(self, name):
        return (name in self.ns
                or self.parent is not None and self.parent.__contains__(name))


nested_scope = Scope()


def macro(name, qnum=1, anum=0, scope=None):
    if scope is None:
        scope = nested_scope

    def decorator(func):
        @wraps(func)
        def wrapper(qubits, *args, scope=scope, **kwds):
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

        scope.set(name, wrapper)
        return wrapper

    return decorator


def gateName(st):
    if isinstance(st[0], str):
        return st[0]
    else:
        return st[0][0]


def call_macro(st, scope):
    func = scope.get(gateName(st))
    qubits = st[1]
    if isinstance(st[0], str):
        args = ()
    else:
        args = st[0][1:]
    return func(qubits, *args)


def extend_macro(qlisp, scope=None):
    if scope is None:
        scope = nested_scope

    ret = []
    for st in qlisp:
        if gateName(st) in scope:
            for st in call_macro(st, scope):
                ret.extend(extend_macro([st], scope))
        else:
            ret.append(st)
    return ret


@macro('u3', 1, 3)
def u3(qubit, theta, phi, lambda_):
    return [(('U', theta, phi, lambda_), qubit)]


@macro('u2', 1, 2)
def u2(qubit, phi, lambda_):
    return [(('U', pi / 2, phi, lambda_), qubit)]


@macro('u1', 1, 1)
def u1(qubit, lambda_):
    return [(('U', 0, 0, lambda_), qubit)]


@macro('H')
def H(qubit):
    return [(('u2', 0, pi), qubit)]


@macro('U', 1, 3)
def U(q, theta, phi, lambda_):
    if theta == 0:
        return [(('Rz', phi + lambda_), q)]
    else:
        return [(('Rz', lambda_), q), (('XYPulse', theta, pi / 2), q),
                (('Rz', phi), q)]


@macro('Cnot', 2)
def cnot(qubits):
    c, t = qubits
    return [('H', t), ('CZ', (c, t)), (('Rz', 0.213), c), (('Rz', -1.819), t),
            ('H', t)]


@macro('crz', 2, 1)
def crz(qubits, lambda_):
    c, t = qubits

    return [(('u1', lambda_ / 2), t), ('Cnot', (c, t)),
            (('u1', -lambda_ / 2), t), ('Cnot', (c, t))]


from collections import defaultdict


def commuteWithRz(st):
    if st[0] in ['CZ', 'I', 'Barrier']:
        return True
    else:
        return False


def reduceVirtualZ(qlisp):
    ret = []
    hold = defaultdict(lambda: 0)

    for st in qlisp:
        if gateName(st) == 'Rz':
            hold[st[1]] = (hold[st[1]] + st[0][1]) % pi
        elif gateName(st) == 'XYPulse':
            ret.append((('XYPulse', st[0][1], st[0][2] - hold[st[1]]), st[1]))
        elif commuteWithRz(st):
            ret.append(st)
        else:
            if isinstance(st[1], int):
                target = (st[1], )
            else:
                target = st[1]
            for q in target:
                if hold[q] != 0:
                    if gateName(st) != 'Measure':
                        ret.append((('Rz', hold[q]), q))
                    hold[q] = 0
            ret.append(st)

    for q in hold:
        if hold[q] != 0:
            ret.append((('Rz', hold[q]), q))
    return ret
    