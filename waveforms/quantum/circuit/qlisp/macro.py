from collections import defaultdict

from numpy import mod, pi

from .qlisp import QLispError, gateName


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
    if isinstance(gate[1], str):
        if gate[1] == 'Z':
            return [('CZ', qubits)]
        elif gate[1] == 'X':
            return [('Cnot', qubits)]
        else:
            return [st]
    else:
        return [st]


def extend_macro(qlisp, lib):
    for st in qlisp:
        if gateName(st) == 'I':
            continue
        elif gateName(st) == 'C':
            yield from extend_control_gate(st, lib)
        else:
            gate = lib.getGate(gateName(st))
            if gate is None:
                yield st
            else:
                for st in call_macro(gate, st):
                    yield from extend_macro([st], lib)


def exchangeRzWithGate(st, phaseList, lib):
    if gateName(st) == 'P':
        return [], [mod(phaseList[0] + st[0][1], 2 * pi)]
    elif gateName(st) == 'rfUnitary':
        (_, theta, phi), qubit = st
        return [(('rfUnitary', theta, phi - phaseList[0]), qubit)], phaseList
    elif gateName(st) in ['Reset', 'Measure']:
        return [st], [0] * len(phaseList)
    elif gateName(st) in ['CZ', 'I', 'Barrier', 'Delay']:
        return [st], phaseList
    elif gateName(st) in ['iSWAP', 'SWAP']:
        return [st], phaseList[::-1]
    else:
        raise Exception


def reduceVirtualZ(qlisp, lib):
    hold = defaultdict(lambda: 0)

    for st in qlisp:
        target = st[1]
        if isinstance(target, (int, str)):
            target = (target, )
        try:
            stList, phaseList = exchangeRzWithGate(st,
                                                   [hold[q] for q in target],
                                                   lib)
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
