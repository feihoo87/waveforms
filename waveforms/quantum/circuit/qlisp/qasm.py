from itertools import chain

from ..qasm import Qasm
from ..qasm.node import *

__qindex = 0
__cindex = 0


def allocaInit():
    global __qindex, __cindex
    __qindex, __cindex = 0, 0


def allocaQubits(size):
    global __qindex
    addresses = tuple(range(__qindex, __qindex + size))
    __qindex += size
    return addresses


def allocaCbits(size):
    global __cindex
    addresses = tuple(range(__cindex, __cindex + size))
    __cindex += size
    return addresses


def get_name(name, scope):
    for s in scope[::-1]:
        if name in s:
            return s[name]
    else:
        raise Exception(f'Undefined {name:r}')


def get_sym(ID, scope, index=None):
    """Return the correspond symbolic number."""
    if not scope:
        if index is not None:
            return ID[index]
        else:
            return ID
    if ID.name not in scope[-1]:
        raise NodeException(
            "Expected local parameter name: ",
            "name=%s, line=%s, file=%s" % (ID.name, ID.line, ID.file))
    if isinstance(ID, IndexedId):
        return get_sym(scope[-1][ID.name], scope[0:-1], ID.index)
    else:
        return get_sym(scope[-1][ID.name], scope[0:-1])


def qasm_eval_single_opaque(st, opaque, scope):
    assert len(opaque.bitlist.children) == len(st.bitlist.children)
    bitlist = [get_sym(q, scope) for q in st.bitlist.children]
    if st.arguments is not None:
        assert len(opaque.arguments.children) == len(st.arguments.children)
        args = [a.real(scope) for a in st.arguments.children]
        gate = (st.name, *args)
    else:
        gate = st.name
    if isinstance(bitlist[0], int):
        return [(gate, tuple(bitlist) if len(bitlist) > 1 else bitlist[0])]
    else:
        return [(gate, qubits if len(qubits) > 1 else qubits[0])
                for qubits in zip(*bitlist)]


def qasm_eval_single(st, scope):
    if isinstance(st, Measure):
        q = get_sym(st.children[0], scope)
        c = get_sym(st.children[1], scope)
        if isinstance(q, tuple):
            assert len(q) == len(c)
            return [(('Measure', c_), q_) for c_, q_ in zip(c, q)]
        else:
            return [(('Measure', c), q)]
    elif isinstance(st, UniversalUnitary):
        args = [a.real(scope) for a in st.children[0].children]
        q = get_sym(st.children[1], scope)
        if isinstance(q, tuple):
            return [(('U', *args), q_) for q_ in q]
        else:
            return [(('U', *args), q)]
    elif isinstance(st, Cnot):
        c = get_sym(st.children[0], scope)
        t = get_sym(st.children[1], scope)
        if isinstance(c, tuple):
            assert len(c) == len(t)
            return [('Cnot', (c_, t_)) for c_, t_ in zip(c, t)]
        else:
            return [('Cnot', (c, t))]
    elif isinstance(st, Barrier):
        q = tuple(chain(*[get_sym(_, scope) for _ in st.children[0].children]))
        return [('Barrier', q)]
    elif isinstance(st, Reset):
        q = get_sym(st.children[0], scope)
        if isinstance(q, tuple):
            return [('Reset', q_) for q_ in q]
        else:
            return [('Reset', q)]
    else:
        return []


def qasm_eval_prog(prog, scope=None):
    current_scope = {}
    if scope is None:
        allocaInit()
        scope = [current_scope]

    qlisp = []

    for st in prog.children:
        if isinstance(st, (Gate, Opaque)):
            current_scope[st.name] = st
        elif isinstance(st, Qreg):
            current_scope[st.name] = allocaQubits(st.index)
        elif isinstance(st, Creg):
            current_scope[st.name] = allocaCbits(st.index)
        elif isinstance(st, CustomUnitary):
            gate = get_name(st.name, scope)
            sub_scope = {}
            assert len(gate.bitlist.children) == len(st.bitlist.children)
            for name, q in zip(gate.bitlist.children, st.bitlist.children):
                sub_scope[name.name] = q
            if gate.arguments is not None:
                assert len(gate.arguments.children) == len(
                    st.arguments.children)
                for name, a in zip(gate.arguments.children,
                                   st.arguments.children):
                    sub_scope[name.name] = a
            if isinstance(gate, Opaque):
                qlisp.extend(qasm_eval_single_opaque(st, gate, scope))
            elif isinstance(gate, Gate):
                qlisp.extend(qasm_eval_prog(gate.body,
                                            [*scope, sub_scope]))
            else:
                raise Exception(f"{st.name:r} is not gate nor opaque")
        elif isinstance(st, Format):
            pass
        else:
            qlisp.extend(qasm_eval_single(st, scope))

    return qlisp


def qasm_eval(data):
    qasm = Qasm(data=data)
    prog = qasm.parse()
    qlisp = qasm_eval_prog(prog)
    return qlisp
