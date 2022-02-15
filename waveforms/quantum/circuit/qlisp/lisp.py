################ Lispy: Scheme Interpreter in Python

## (c) Peter Norvig, 2010-16; See http://norvig.com/lispy.html

from __future__ import annotations

import inspect
import math
import operator as op
import re
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Optional, Union

import numpy as np
from waveforms.waveform import Waveform, zero

from .qlisp import ABCCompileConfigMixin, ADChannel, MultADChannel


class Token(NamedTuple):
    type: str
    value: str
    line: int
    column: int


class Symbol():
    __slots__ = ('name', )

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        if isinstance(self.name, str):
            return self.name
        return repr(self.name)

    def __eq__(self, other):
        return isinstance(other, Symbol) and other.name == self.name


class Expression(tuple):
    pass


def atom(token):
    "Numbers become numbers; every other token is a symbol."
    for kind in [int, float, complex, Symbol]:
        try:
            return kind(token)
        except ValueError:
            continue
    raise ValueError()


Number = (int, float, complex
          )  # A Lisp Number is implemented as a Python int or float

################ Parsing


def tokenize(code):
    keywords = {}
    token_specification = [
        ('BRACKET', r'[\(\)]'),
        ('STRING', r'\"([^\\\"]|\\.)*\"'),
        ('NEWLINE', r'\n'),  # Line endings
        ('SKIP', r'[ \t]+'),  # Skip over spaces and tabs
        ('ATOM', r'[A-Za-z0-9\.!@#$%^&\*/_\-\+:\?\|\<\>=]+'),  # Atom
        ('COMMENT', r';'),  # Comment
        ('QUOTE', r"'"),
        ('MISMATCH', r'.'),  # Any other character
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    line_num = 1
    line_start = 0
    in_comment = False
    for mo in re.finditer(tok_regex, code):
        kind = mo.lastgroup
        value = mo.group()
        column = mo.start() - line_start

        if kind == 'NEWLINE':
            line_start = mo.end()
            line_num += 1
            in_comment = False
            continue
        elif kind == 'COMMENT':
            in_comment = True
            continue
        elif in_comment or kind == 'SKIP':
            continue
        elif kind == 'STRING':
            value = value[1:-1]
        elif kind == 'BRACKET':
            kind = value
        elif kind == 'ATOM':
            value = atom(value)
        elif kind == 'MISMATCH':
            raise RuntimeError(f'{value!r} unexpected on line {line_num}')
        yield Token(kind, value, line_num, column)


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


class LispError(Exception):
    pass


class Env(dict):
    "An environment: a dict of {'var':val} pairs, with an outer Env."

    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer

    def find(self, var):
        "Find the innermost Env where var appears."
        return self if (var in self) else self.outer.find(var)

    def lookup(self, name):
        try:
            return self.find(name)[name]
        except:
            raise KeyError(f'can not find {name}')

    def set(self, name, value):
        self.find(name)[name] = value

    def assign(self, name, value):
        self[name] = value


################ Procedures


class Procedure():
    "A user-defined Scheme procedure."

    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env


################ eval
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
        '+':op.add, '-':op.sub, '*':op.mul, '/':op.truediv, '%':op.mod,
        '>':op.gt, '<':op.lt, '>=':op.ge, '<=':op.le, '==':op.eq, '**':op.pow,
        '>>':op.rshift, '<<':op.lshift, '&':op.and_, '^':op.xor, '|':op.or_,
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
        'map':     map,
        'max':     max,
        'min':     min,
        'not':     op.not_,
        'null?':   lambda x: x == (),
        'number?': lambda x: isinstance(x, Number),
        'procedure?': lambda x: isinstance(x, Procedure) or callable(x),
        'round':   round,
        'symbol?': lambda x: isinstance(x, Symbol),
        'string?': lambda x: isinstance(x, str),
        'display!':print,
        'error':   error,
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

    def execute(self, cmd, target, *args):
        if cmd == '!set_waveform':
            self.raw_waveforms[target] = args[0]
        elif cmd == '!set_phase':
            self.phases[target] = args[0]
        else:
            pass


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
    (_, parms, body) = exp
    for i, p in enumerate(parms, start=1):
        if not isinstance(p, Symbol):
            raise TypeError(f'the {i} param must be a symbol')
    stack.push(Procedure([p.name for p in parms], body, env))


def eval_begin(exp, env, stack):
    for x in exp[1:-1]:
        eval(x, env, stack)
        stack.pop()
    eval(exp[-1], env, stack)


def eval_let(exp, env, stack):
    (_, bindings, body) = exp
    let_env = Env(parms=[], args=[], outer=env)
    for (var, val) in bindings:
        eval(val, env, stack)
        if not isinstance(var, Symbol):
            raise TypeError(f'var must be a symbol')
        let_env.assign(var.name, stack.pop())
    eval(body, let_env, stack)


def eval_letstar(exp, env, stack):
    (_, bindings, body) = exp
    letstar_env = Env(parms=[], args=[], outer=env)
    for (var, val) in bindings:
        eval(val, letstar_env, stack)
        if not isinstance(var, Symbol):
            raise TypeError(f'var must be a symbol')
        letstar_env.assign(var.name, stack.pop())
    eval(body, letstar_env, stack)


def python_to_qlisp(exp):
    if isinstance(exp, Expression):
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


def apply(proc, args, env, stack):
    if isinstance(proc, Procedure):
        eval(proc.body, Env(proc.parms, args, proc.env), stack)
    elif callable(proc):
        x = proc(*args)
        if inspect.isgenerator(x):
            for instruction in get_ret(x, env, stack):
                instruction = python_to_qlisp(instruction)
                if instruction[0].name.startswith('!'):
                    try:
                        cmd, target, *args = instruction
                        for a in reversed(args):
                            eval(a, env, stack)
                        args = [stack.pop() for _ in args]
                        stack.execute(cmd.name, target.name, *args)
                    except:
                        raise Exception(f'bad instruction {instruction}')
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


__eval_table = {
    'quote': eval_quote,
    'if': eval_if,
    'cond': eval_cond,
    'while': eval_while,
    'define': eval_define,
    'set!': eval_set,
    'setq': eval_setq,
    'lambda': eval_lambda,
    'begin': eval_begin,
    'let': eval_let,
    'let*': eval_letstar,
    'apply': eval_apply,
}


def eval(x, env, stack):
    "Evaluate an expression in an environment."
    if isinstance(x, Symbol):  # variable reference
        stack.push(env.lookup(x.name))
    elif isinstance(x, Expression):
        if len(x) == 0:
            stack.push(None)
        elif isinstance(x[0], Symbol) and x[0].name in __eval_table:
            __eval_table[x[0].name](x, env, stack)
        else:
            eval(Expression([Symbol('apply'), x[0], x[1:]]), env, stack)
    else:  # constant literal
        stack.push(x)


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
