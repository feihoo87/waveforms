from collections import defaultdict

from numpy import cos, mod, pi

from waveforms.math.matricies import U, Unitary2Angles

from .base import QLispError, head
from .simulator.simple import seq2mat


def _lookup(name, env):
    try:
        return env.get(name, name)
    except:
        return name


def lookup(st, env):
    if isinstance(st[1], tuple):
        return (st[0], tuple(_lookup(q, env) for q in st[1]))
    elif isinstance(st[1], (str, int)):
        return (st[0], _lookup(st[1], env))
    else:
        return st


def define_macro(name, value, env):
    env[name] = value


def call_macro(gate, st):
    qubits = st[1]
    if isinstance(st[0], str):
        args = ()
    else:
        args = st[0][1:]
    try:
        yield from gate(qubits, *args)
    except:
        raise QLispError(f'extend macro {st} error.')


def extend_control_gate(st, scope):
    # TODO
    gate, qubits = st
    _, unitary = gate
    if isinstance(unitary, str):
        if unitary == 'Z':
            return [('CZ', qubits)]
        elif unitary == 'X':
            return [('Cnot', qubits)]

    # https://doi.org/10.1103/PhysRevA.52.3457

    unitary = [(unitary, 0)]
    c, t = qubits

    W = seq2mat(unitary)
    theta, phi, lam, delta = Unitary2Angles(W)

    if abs(cos(phi + lam) + 1) < 1e-15:
        return [
            (('u3', pi / 2 - theta / 2, 0, -phi), t),
            ('Cnot', c, t),
            (('u3', theta / 2 - pi / 2, phi, 0), t),
            (('Rz', delta - pi / 2), t),
        ]
    else:
        return [
            (('u3', theta / 2, 0, lam), t),
            ('Cnot', c, t),
            (('u3', -theta / 2, -(phi + lam) / 2, 0), t),
            ('Cnot', c, t),
            (('u3', 0, 0, (phi - lam) / 2), t),
            (('Rz', delta), c),
        ]


def extend_macro(qlisp, lib, env=None):
    if env is None:
        env = {}
    for st in qlisp:
        if head(st) == 'define':
            define_macro(st[1], st[2], env)
        elif head(st) == 'C':
            st = lookup(st, env)
            yield from extend_control_gate(st, lib)
        else:
            st = lookup(st, env)
            gate = lib.getGate(head(st))
            if gate is None:
                yield st
            else:
                for st in call_macro(gate, st):
                    yield from extend_macro([st], lib, env)


_VZ_rules = {}


def add_VZ_rule(gateName, rule):
    _VZ_rules[gateName] = rule


def remove_VZ_rule(gateName, rule):
    del _VZ_rules[gateName]


def _VZ_P(st, phaseList):
    return [], [mod(phaseList[0] + st[0][1], 2 * pi)]


def _VZ_rfUnitary(st, phaseList):
    (_, theta, phi, *with_params), qubit = st
    return [(('rfUnitary', theta, phi - phaseList[0], *with_params), qubit)
            ], phaseList


def _VZ_clear(st, phaseList):
    return [st], [0] * len(phaseList)


def _VZ_exchangable(st, phaseList):
    return [st], phaseList


def _VZ_swap(st, phaseList):
    return [st], phaseList[::-1]


add_VZ_rule('P', _VZ_P)
add_VZ_rule('rfUnitary', _VZ_rfUnitary)
add_VZ_rule('Reset', _VZ_clear)
add_VZ_rule('Measure', _VZ_clear)
add_VZ_rule('CZ', _VZ_exchangable)
add_VZ_rule('I', _VZ_exchangable)
add_VZ_rule('Barrier', _VZ_exchangable)
add_VZ_rule('Delay', _VZ_exchangable)
add_VZ_rule('iSWAP', _VZ_swap)
add_VZ_rule('SWAP', _VZ_swap)


def _exchange_VZ_and_gate(st, phaseList, lib):
    gate = head(st)
    if gate in _VZ_rules:
        return _VZ_rules[gate](st, phaseList)
    else:
        raise Exception('Unknow VZ exchange rule.')


def reduceVirtualZ(qlisp, lib):
    hold = defaultdict(lambda: 0)

    for st in qlisp:
        target = st[1]
        if isinstance(target, (int, str)):
            target = (target, )
        try:
            stList, phaseList = _exchange_VZ_and_gate(
                st, [hold[q] for q in target], lib)
            yield from stList
            for q, p in zip(target, phaseList):
                hold[q] = mod(p, 2 * pi)
        except:
            for q in target:
                if hold[q] != 0:
                    yield (('P', hold[q]), q)
                    hold[q] = 0
            yield st

    for q in hold:
        if hold[q] != 0:
            yield (('P', hold[q]), q)
