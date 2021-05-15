from collections import defaultdict
from functools import wraps
from inspect import signature

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


def gate(qnum=1, name=None, scope=None):
    if scope is None:
        scope = nested_scope

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


def extend_control_gate(st, scope):
    # TODO
    gate, qubits = st
    if isinstance(gate[1], str):
        if gate[1] == 'Z':
            return [('CZ', qubits)]
        elif gate[1] == 'X':
            return [('Cnot', qubits)]
        else:
            return [st]
    else:
        return [st]


def extend_macro(qlisp, scope=None):
    if scope is None:
        scope = nested_scope

    ret = []
    for st in qlisp:
        if gateName(st) == 'C':
            ret.extend(extend_control_gate(st, scope))
        if gateName(st) in scope:
            for st in call_macro(st, scope):
                ret.extend(extend_macro([st], scope))
        else:
            ret.append(st)
    return ret


@gate()
def u3(qubit, theta, phi, lambda_):
    return [(('U', theta, phi, lambda_), qubit)]


@gate()
def u2(qubit, phi, lambda_):
    return [(('U', pi / 2, phi, lambda_), qubit)]


@gate()
def u1(qubit, lambda_):
    return [(('U', 0, 0, lambda_), qubit)]


@gate()
def H(qubit):
    return [(('u2', 0, pi), qubit)]


@gate()
def U(q, theta, phi, lambda_):
    if theta == 0:
        return [(('P', phi + lambda_), q)]
    else:
        return [(('P', lambda_), q), (('rfUnitary', theta, -pi / 2), q),
                (('P', phi), q)]


@gate(2)
def Cnot(qubits):
    c, t = qubits
    return [('H', t), ('CZ', (c, t)), ('H', t)]


@gate(2)
def crz(qubits, lambda_):
    c, t = qubits

    return [(('u1', lambda_ / 2), t), ('Cnot', (c, t)),
            (('u1', -lambda_ / 2), t), ('Cnot', (c, t))]


def commuteWithRz(st):
    if gateName(st) in ['CZ', 'I', 'Barrier']:
        return True
    else:
        return False


def exchangeRzWithGate(st, phaseList):
    if gateName(st) in ['iSWAP', 'SWAP']:
        return [st], phaseList[::-1]
    else:
        raise Exception


def reduceVirtualZ(qlisp):
    ret = []
    hold = defaultdict(lambda: 0)

    for st in qlisp:
        if gateName(st) == 'P':
            hold[st[1]] = (hold[st[1]] + st[0][1]) % pi
        elif gateName(st) == 'rfUnitary':
            ret.append(
                (('rfUnitary', st[0][1], st[0][2] + hold[st[1]]), st[1]))
        elif commuteWithRz(st):
            ret.append(st)
        else:
            if isinstance(st[1], int):
                target = (st[1], )
            else:
                target = st[1]
            try:
                stList, phaseList = exchangeRzWithGate(
                    st, [hold[q] for q in target])
                ret.extend(stList)
                for q, p in zip(target, phaseList):
                    hold[q] = p
            except:
                for q in target:
                    if hold[q] != 0:
                        if gateName(st) != 'Measure':
                            ret.append((('P', hold[q]), q))
                        hold[q] = 0
                ret.append(st)

    for q in hold:
        if hold[q] != 0:
            ret.append((('P', hold[q]), q))
    return ret
