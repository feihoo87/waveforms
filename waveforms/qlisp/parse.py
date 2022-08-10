from __future__ import annotations

import functools
import inspect
import math
import operator as op
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Union

import numpy as np
from waveforms.waveform import Waveform, zero

from .base import ABCCompileConfigMixin, ADChannel, MultADChannel
from .tokenize import Expression, Number, Symbol, tokenize

################ Parsing


def parse(program):
    "Read a Scheme expression from a string."

    stack = []
    end = False

    for token in tokenize(program):
        if end:
            raise SyntaxError('unexpected EOF while reading')
        if token.type == '(':
            stack.append([])
        elif token.type == 'QUOTE':
            stack.append([Symbol('quote')])
        elif token.type == ')':
            try:
                s_expr = Expression(stack.pop())
            except IndexError:
                raise SyntaxError(
                    f'unexpected ) at L{token.line}C{token.column}')
            while True:
                if stack:
                    stack[-1].append(s_expr)
                    if Symbol('quote') == stack[-1][0]:
                        s_expr = Expression(stack.pop())
                    else:
                        break
                else:
                    end = s_expr
                    break
        else:
            s_expr = token.value
            while True:
                if stack:
                    stack[-1].append(s_expr)
                    if Symbol('quote') == stack[-1][0]:
                        s_expr = Expression(stack.pop())
                    else:
                        break
                elif Symbol('quote') == s_expr[0]:
                    end = s_expr
                    break
                else:
                    raise SyntaxError(
                        f'expected ( before L{token.line}C{token.column}')
    if not end:
        raise SyntaxError(f'expected ) after L{token.line}C{token.column}')
    return end


################ Environments


class Env(dict):
    "An environment: a dict of {'var':val} pairs, with an outer Env."

    def __init__(self, params=(), args=(), outer=None):
        self.update(zip(params, args))
        self.outer = outer

    def find(self, var):
        "Find the innermost Env where var appears."
        return self if (var in self) else self.outer.find(var)

    def lookup(self, name):
        "Find the value of a variable, starting in this Env."
        try:
            return self.find(name)[name]
        except:
            raise KeyError(f'can not find {name}')

    def set(self, name, value):
        "Set a variable to a value."
        self.find(name)[name] = value

    def assign(self, name, value):
        "Set a variable to a value."
        self[name] = value


################ Procedures


class Procedure():
    "A user-defined Scheme procedure."

    def __init__(self, params, body, env):
        self.params, self.body, self.env = params, body, env


class Gate(Procedure):
    "A quantum operation."

    def __init__(self, name, params, qubits, body, env, bindings=None):
        self.name = name
        self.params = params
        self.bindings = bindings
        self.qubits = qubits
        self.body = body
        self.env = env


class Channel(NamedTuple):
    qubits: tuple
    name: str


################ eval


class LispError(Exception):
    pass


def error(msg, *args):
    raise LispError(msg, *args)


def make_waveform(*args):
    from waveforms import Waveform, wave_eval

    if len(args) == 1 and isinstance(args[0], str):
        return wave_eval(args[0])
    elif len(args) == 1 and isinstance(args[0], Waveform):
        return args[0]
    else:
        try:
            return Waveform.fromlist(list(args))
        except:
            raise RuntimeError('invalid waveform')


def standard_env():
    "An environment with some Scheme standard procedures."
    env = Env()
    env.update(vars(math))  # sin, cos, sqrt, pi, ...
    env.update({
        '+': lambda *x: functools.reduce(op.add, x),
        '*': lambda *x: functools.reduce(op.mul, x),
        '&': lambda *x: functools.reduce(op.and_, x),
        '|': lambda *x: functools.reduce(op.or_, x),
        '-': op.sub, '/': op.truediv, '%': op.mod, '**': op.pow,
        '>': op.gt, '<': op.lt, '>=': op.ge, '<=': op.le, '==': op.eq,
        '>>': op.rshift, '<<': op.lshift, '^': op.xor,
        'pow':     op.pow,
        'abs':     abs,
        'append':  op.add,
        'car':     lambda x: x[0],
        'cdr':     lambda x: x[1:],
        'cons':    lambda x,y: (x,) + y,
        'is?':     op.is_,
        'eq?':     op.eq,
        'equal?':  op.eq,
        'length':  len,
        'list':    lambda *x: x,
        'list?':   lambda x: isinstance(x, tuple),
        'max':     max,
        'min':     min,
        'and':     all,
        'or':      any,
        'all':     all,
        'any':     any,
        'not':     op.not_,
        'null?':   lambda x: x == (),
        'number?': lambda x: isinstance(x, Number),
        'procedure?': lambda x: isinstance(x, Procedure) or callable(x),
        'round':   round,
        'symbol?': lambda x: isinstance(x, Symbol),
        'string?': lambda x: isinstance(x, str),
        'display!':print,
        'error!':  error,
        'waveform':make_waveform,
        'pi':      np.pi,
        'e':       np.e,
        'inf':     np.inf,
        '-inf':    -np.inf,
        'nil':     None,
        'null':    None,
        'None':    None,
        'true':    True,
        'false':   False,
        'True':    True,
        'False':   False,
    }) # yapf: disable
    return env


class Capture(NamedTuple):
    qubit: str
    cbit: int
    time: float
    signal: Union[str, tuple[str]]
    params: dict
    hardware: Union[ADChannel, MultADChannel] = None
    shift: float = 0


@dataclass
class Stack():
    cfg: ABCCompileConfigMixin = field(default_factory=dict)
    scopes: list[dict[str, Any]] = field(default_factory=lambda: [dict()])
    qlisp: list = field(default_factory=list)
    time: dict[str,
               float] = field(default_factory=lambda: defaultdict(lambda: 0))
    addressTable: dict = field(default_factory=dict)
    waveforms: dict[str, Waveform] = field(
        default_factory=lambda: defaultdict(zero))
    raw_waveforms: dict[tuple[str, ...], Waveform] = field(
        default_factory=lambda: defaultdict(zero))
    measures: dict[int, Capture] = field(default_factory=dict)
    phases_ext: dict[str, dict[Union[int, str], float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    biases: dict[str,
                 float] = field(default_factory=lambda: defaultdict(lambda: 0))
    end: float = 0
    _ret: list = field(default_factory=list)

    @property
    def channel(self):
        return self.raw_waveforms

    @property
    def phases(self):

        class D():
            __slots__ = ('ctx', )

            def __init__(self, ctx):
                self.ctx = ctx

            def __getitem__(self, qubit):
                return self.ctx.phases_ext[qubit][1]

            def __setitem__(self, qubit, phase):
                self.ctx.phases_ext[qubit][1] = phase

        return D(self)

    @property
    def params(self):
        return self.scopes[-1]

    @property
    def vars(self):
        return self.scopes[-2]

    @property
    def globals(self):
        return self.scopes[0]

    def qubit(self, q):
        return self.addressTable[q]

    def push(self, x):
        self._ret.append(x)

    def pop(self):
        return self._ret.pop()

    def pick(self):
        return self._ret[-1]


def eval_quote(exp, env, stack):
    (_, exp) = exp
    stack.push(exp)


def eval_if(exp, env, stack):
    (_, test, conseq, alt) = exp
    eval(test, env, stack)
    if stack.pop():
        eval(conseq, env, stack)
    else:
        eval(alt, env, stack)


def eval_cond(exp, env, stack):
    for (test, conseq) in exp[1:]:
        eval(test, env, stack)
        if stack.pop():
            eval(conseq, env, stack)
            break
    stack.push(None)


def eval_while(exp, env, stack):
    (_, test, body) = exp
    ret = None
    while True:
        eval(test, env, stack)
        if not stack.pop():
            break
        eval(body, env, stack)
        ret = stack.pop()
    stack.push(ret)


def eval_define(exp, env, stack):
    (_, var, exp) = exp
    eval(exp, env, stack)
    if not isinstance(var, Symbol):
        raise TypeError(f'var must be a symbol')
    env.assign(var.name, stack.pop())
    stack.push(None)


def eval_set(exp, env, stack):
    (_, var, exp) = exp
    eval(exp, env, stack)
    eval(var, env, stack)
    var = stack.pop()
    if not isinstance(var, Symbol):
        raise TypeError(f'var must be a symbol')
    env.set(var.name, stack.pop())
    stack.push(None)


def eval_setq(exp, env, stack):
    (_, var, exp) = exp
    if not isinstance(var, Symbol):
        raise TypeError(f'var must be a symbol')
    eval(exp, env, stack)
    env.set(var.name, stack.pop())
    stack.push(None)


def eval_lambda(exp, env, stack):
    (_, params, body) = exp
    for i, p in enumerate(params, start=1):
        if not isinstance(p, Symbol):
            raise TypeError(f'the {i} param must be a symbol')
    stack.push(Procedure([p.name for p in params], body, env))


def eval_defun(exp, env, stack):
    (_, var, params, *body) = exp
    if not isinstance(var, Symbol):
        raise TypeError(f'var must be a symbol')
    env.assign(
        var.name,
        Procedure([p.name for p in params],
                  Expression([Symbol('begin'), *body]), env))
    stack.push(None)


def eval_gate(exp, env, stack):
    (_, var, qubits, *body) = exp
    if isinstance(var, Expression):
        gate, *params = var
    elif isinstance(var, Symbol):
        gate = var
        params = []
    else:
        raise TypeError(f'var must be a symbol')
    env.assign(
        gate.name,
        Gate(gate.name, [p.name for p in params], [q.name for q in qubits],
             Expression([Symbol('begin'), *body]), env))
    stack.push(None)


def eval_begin(exp, env, stack):
    for x in exp[1:-1]:
        eval(x, env, stack)
        stack.pop()
    eval(exp[-1], env, stack)


def eval_let(exp, env, stack):
    (_, bindings, body) = exp
    let_env = Env(params=[], args=[], outer=env)
    for (var, val) in bindings:
        eval(val, env, stack)
        if not isinstance(var, Symbol):
            raise TypeError(f'var must be a symbol')
        let_env.assign(var.name, stack.pop())
    eval(body, let_env, stack)


def eval_letstar(exp, env, stack):
    (_, bindings, body) = exp
    letstar_env = Env(params=[], args=[], outer=env)
    for (var, val) in bindings:
        eval(val, letstar_env, stack)
        if not isinstance(var, Symbol):
            raise TypeError(f'var must be a symbol')
        letstar_env.assign(var.name, stack.pop())
    eval(body, letstar_env, stack)


def python_to_qlisp(exp):
    if isinstance(exp, (Expression, Channel)):
        return exp
    elif isinstance(exp, tuple):
        return Expression([python_to_qlisp(x) for x in exp])
    elif isinstance(exp, str):
        if len(exp) >= 2 and exp.startswith('"') and exp.endswith('"'):
            return exp[1:-1]
        else:
            return Symbol(exp)
    else:
        return exp


def get_ret(gen, env, stack):
    ret = yield from gen
    eval(python_to_qlisp(ret), env, stack)


def apply_gate(gate: Gate, args, env, stack):
    if gate.bindings is None and len(gate.params) > 0:
        bindings = dict(zip(gate.params, args))
        stack.push(
            Gate(gate.name, gate.params, gate.qubits, gate.body, gate.env,
                 bindings))
        return
    else:
        qubits = args

    if isinstance(gate.body, Expression):
        inner_env = Env(gate.qubits, qubits, gate.env)
        if gate.bindings is not None:
            inner_env.update(gate.bindings)
        eval(gate.body, inner_env, stack)
    else:
        pass


def apply(proc, args, env, stack):
    if isinstance(proc, Gate):
        apply_gate(proc, args, env, stack)
    elif isinstance(proc, Procedure):
        eval(proc.body, Env(proc.params, args, proc.env), stack)
    elif callable(proc):
        x = proc(*args)
        if inspect.isgenerator(x):
            for instruction in get_ret(x, env, stack):
                instruction = python_to_qlisp(instruction)
                if instruction[0].name.startswith('!'):
                    eval_instruction(instruction, env, stack)
                    # try:
                    #     cmd, target, *args = instruction
                    #     for a in reversed(args):
                    #         eval(a, env, stack)
                    #     args = [stack.pop() for _ in args]
                    #     if isinstance(target, Symbol):
                    #         target = target.name
                    #     execute(stack, cmd.name, target, *args)
                    # except:
                    #     raise Exception(f'bad instruction {instruction}')
                else:
                    eval(instruction, env, stack)
                    stack.pop()
        else:
            stack.push(x)
    else:
        raise TypeError('{} is not callable'.format(proc))


def eval_apply(exp, env, stack):
    (_, proc, args) = exp
    for exp in args[::-1]:
        eval(exp, env, stack)
    eval(proc, env, stack)
    proc = stack.pop()
    args = [stack.pop() for _ in args]
    apply(proc, args, env, stack)


def eval_channel(exp, env, stack):
    (_, *qubits, name) = exp
    if isinstance(name, Symbol):
        name = name.name
    qubits = tuple(q.name for q in qubits)
    stack.push(Channel(name, qubits))


def eval_instruction(instruction, env, stack):
    env = Env(params=[], args=[], outer=env)
    try:
        cmd, target, *args = instruction
        for a in reversed(args):
            eval(a, env, stack)
        args = [stack.pop() for _ in args]
        eval(target, env, stack)
        target = stack.pop()
        execute(stack, cmd.name, target, *args)
    except:
        raise Exception(f'bad instruction {instruction}')


__eval_table = {
    'quote': eval_quote,
    'if': eval_if,
    'cond': eval_cond,
    'while': eval_while,
    'define': eval_define,
    'defun': eval_defun,
    'gate': eval_gate,
    'set!': eval_set,
    'setq': eval_setq,
    'lambda': eval_lambda,
    'begin': eval_begin,
    'let': eval_let,
    'let*': eval_letstar,
    'apply': eval_apply,
    'channel': eval_channel,
}


def eval(x, env, stack):
    "Evaluate an expression in an environment."
    if isinstance(x, Symbol):  # variable reference
        stack.push(env.lookup(x.name))
    elif isinstance(x, Expression):
        if len(x) == 0:
            stack.push(None)
        elif isinstance(x[0], Symbol) and x[0].name.startswith('!'):
            eval_instruction(x, env, stack)
        elif isinstance(x[0], Symbol) and x[0].name in __eval_table:
            __eval_table[x[0].name](x, env, stack)
        else:
            eval(Expression([Symbol('apply'), x[0], x[1:]]), env, stack)
    else:  # constant literal
        stack.push(x)


################ virtual machine


def execute(stack, cmd, target, *args):
    if cmd == '!nop':
        pass
    elif cmd == '!set_waveform':
        stack.raw_waveforms[target] = args[0]
    elif cmd == '!add_waveform':
        stack.raw_waveforms[target] += args[0]
    elif cmd == '!set_phase':
        stack.phases[target] = args[0]
    else:
        pass
    stack.push(None)


################ Interaction: A REPL


def lispstr(exp):
    "Convert a Python object back into a Lisp-readable string."
    if isinstance(exp, Expression):
        return '(' + ' '.join(map(lispstr, exp)) + ')'
    else:
        return str(exp)


def repl(prompt='lis.py> '):
    "A prompt-read-eval-print loop."
    global_stack = Stack()
    global_env = standard_env()
    while True:
        eval(parse(input(prompt)), global_env, global_stack)
        val = global_stack.pop()
        if val is not None:
            print(lispstr(val))


if __name__ == "__main__":
    repl()
