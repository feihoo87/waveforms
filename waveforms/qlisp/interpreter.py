from __future__ import annotations

import operator
from collections import defaultdict
from enum import Enum, auto
from math import e, pi
from typing import Any

from .parse import parse as qlisp_parser
from .tokenize import Expression, Symbol


class OPCODE(Enum):
    NOP = auto()
    DUP = auto()
    DROP = auto()
    SWAP = auto()
    OVER = auto()
    SLOAD = auto()
    SSTORE = auto()
    LOAD = auto()
    STORE = auto()
    CALL = auto()
    RET = auto()
    CALL_RET = auto()
    JMP = auto()
    JNE = auto()
    JE = auto()
    EXIT = auto()

    # Waveform operators
    PLAY = auto()
    CAPTURE = auto()
    DELAY = auto()
    SET_PHASE = auto()
    ADD_PHASE = auto()
    SET_TIME = auto()
    BARRIER = auto()

    # Register commands
    _SP = auto()
    _BP = auto()
    _SL = auto()
    _PC = auto()
    _WRITE_SP = auto()
    _WRITE_BP = auto()
    _WRITE_SL = auto()
    _WRITE_PC = auto()

    def __repr__(self):
        return self.name


class Frame():

    def __init__(self, frequency: int = 0):
        self.frequency = frequency
        self.phase = 0
        self.time = 0


class Channel():

    def __init__(self, name: str = 'demo', group: str = 'default'):
        self.name = name
        self.group = group
        self.frames = defaultdict(Frame)


def barrier(*channels: Channel):
    t_max = 0
    for channel in channels:
        for frame in channel.frames.values():
            t_max = max(t_max, frame.time)
    for channel in channels:
        for frame in channel.frames.values():
            frame.time = t_max


def vm_pop(vm):
    vm._pop()


def vm_dup(vm):
    x = vm._pop()
    vm._push(x)
    vm._push(x)


def vm_swap(vm):
    v2 = vm._pop()
    v1 = vm._pop()
    vm._push(v2)
    vm._push(v1)


def vm_over(vm):
    x = vm._pop()
    y = vm._pop()
    vm._push(y)
    vm._push(x)
    vm._push(y)


def vm_sload(vm):
    level = vm._pop()
    n = vm._pop()
    addr = vm.bp
    for _ in range(level):
        addr = vm.stack[addr - 3]
    vm._push(vm.stack[addr + n])


def vm_sstore(vm):
    level = vm._pop()
    n = vm._pop()
    value = vm._pop()

    addr = vm.bp
    for _ in range(level):
        addr = vm.stack[addr - 3]
    vm.stack[addr + n] = value


def vm_load(vm):
    addr = vm._pop()
    vm._push(vm.mem[addr])


def vm_store(vm):
    addr = vm._pop()
    value = vm._pop()
    vm.mem[addr] = value


def vm_call(vm):
    func = vm._pop()
    argc = vm._pop()
    args = [vm._pop() for _ in range(argc)]

    if callable(func):
        vm._push(func(*args))
    elif isinstance(func, int):
        vm._push(vm.bp)
        vm._push(vm.sl)
        vm._push(vm.pc)
        vm.bp = vm.sp
        for arg in reversed(args):
            vm._push(arg)
        vm.sl = func
        vm.pc = func
    else:
        raise RuntimeError(f"not callable {func}")


def vm_ret(vm):
    result = vm._pop()
    vm.sp = vm.bp - 3
    vm.bp, vm.sl, vm.pc = vm.stack[vm.bp - 3:vm.bp]
    vm._push(result)


def vm_call_ret(vm):
    func = vm._pop()
    argc = vm._pop()

    if callable(func):
        args = reversed(vm.stack[vm.sp - argc:vm.sp])
        vm.sp = vm.bp - 3
        vm.bp, vm.sl, vm.pc = vm.stack[vm.bp - 3:vm.bp]
        vm._push(func(*args))
    elif isinstance(func, int):
        if vm.sp != vm.bp + argc:
            vm.stack[vm.bp:vm.bp + argc] = vm.stack[vm.sp - argc:vm.sp]
            vm.sp = vm.bp + argc
        vm.sl = func
        vm.pc = func
    else:
        raise RuntimeError(f"not callable {func}")


def vm_jmp(vm):
    vm.pc = vm._pop() + vm.sl


def vm_jne(vm):
    addr = vm._pop()
    cond = vm._pop()
    if cond:
        vm.pc = addr + vm.sl


def vm_je(vm):
    addr = vm._pop()
    cond = vm._pop()
    if cond == 0:
        vm.pc = addr + vm.sl


dispatch_table = {
    OPCODE.DUP: vm_dup,
    OPCODE.DROP: vm_pop,
    OPCODE.SWAP: vm_swap,
    OPCODE.OVER: vm_over,
    OPCODE.SLOAD: vm_sload,
    OPCODE.SSTORE: vm_sstore,
    OPCODE.LOAD: vm_load,
    OPCODE.STORE: vm_store,
    OPCODE.CALL: vm_call,
    OPCODE.RET: vm_ret,
    OPCODE.CALL_RET: vm_call_ret,
    OPCODE.JMP: vm_jmp,
    OPCODE.JNE: vm_jne,
    OPCODE.JE: vm_je,
}


class VirtualMachine:

    def __init__(self, debug=False, dispatch=dispatch_table):
        self.mem = []
        self.stack = []
        self.channels = defaultdict(Channel)
        self.sp = 0  # stack pointer
        self.bp = 0  # base pointer
        self.sl = 0  # static link
        self.pc = 0  # program counter
        self.clk = 0
        self.debug = debug
        self.dispatch = dispatch

        self._dispatch_register = {
            OPCODE._SP: 'sp',
            OPCODE._BP: 'bp',
            OPCODE._SL: 'sl',
            OPCODE._PC: 'pc',
            OPCODE._WRITE_SP: 'sp',
            OPCODE._WRITE_BP: 'bp',
            OPCODE._WRITE_SL: 'sl',
            OPCODE._WRITE_PC: 'pc',
        }

    def _next(self):
        self.pc += 1
        return self.mem[self.pc - 1]

    def _pop(self):
        self.sp -= 1
        return self.stack[self.sp]

    def _push(self, value):
        if len(self.stack) > self.sp:
            self.stack[self.sp] = value
        else:
            self.stack.append(value)
        self.sp += 1

    def _pick(self, n=0):
        return self.stack[self.sp - n - 1]

    def _play(self):
        frame = self._pop()
        pulse = self._pop()

    def _captrue(self):
        frame = self._pop()
        cbit = self._pop()

    def display(self):
        if not self.debug:
            return
        print(f'State[{self.clk}] ====================')
        print(f'      OP: ', self.mem[self.pc])
        print(f'   STACK: ', self.stack[:self.sp])
        print(f'      BP: ', self.bp)
        print(f'      SL: ', self.sl)
        print(f'      PC: ', self.pc)
        print('')

    def trace(self):
        if not self.debug:
            return
        self.display()

    def run(self, code, step_limit=-1):
        if len(code) > len(self.mem):
            self.mem.extend([0] * (len(code) - len(self.mem)))
        for i, c in enumerate(code):
            self.mem[i] = c
        self.sp = 0  # stack pointer
        self.bp = 0  # base pointer
        self.sl = 0  # static link
        self.pc = 0  # program counter
        self.clk = 0  # clock

        while True:
            self.trace()
            op = self._next()
            if self.clk == step_limit:
                break
            self.clk += 1
            if isinstance(op, OPCODE):
                if op in self.dispatch:
                    self.dispatch[op](self)
                elif op == OPCODE.NOP:
                    continue
                elif op == OPCODE.EXIT:
                    break
                elif op in [OPCODE._SP, OPCODE._BP, OPCODE._SL, OPCODE._PC]:
                    self._push(getattr(self, self._dispatch_register[op]))
                elif op in [
                        OPCODE._WRITE_SP, OPCODE._WRITE_BP, OPCODE._WRITE_SL,
                        OPCODE._WRITE_PC
                ]:
                    setattr(self, self._dispatch_register[op], self._pop())
                else:
                    raise RuntimeError(f"unknown command {op}")
            else:
                self._push(op)


class Context:

    def __init__(self):
        self.internal = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '//': operator.floordiv,
            '%': operator.mod,
            '==': operator.eq,
            '!=': operator.ne,
            '<': operator.lt,
            '>': operator.gt,
            '<=': operator.le,
            '>=': operator.ge,
            'and': operator.and_,
            'or': operator.or_,
            'not': operator.not_,
            '>>': operator.rshift,
            '<<': operator.lshift,
            '|': operator.or_,
            '&': operator.and_,
            '^': operator.xor,
            '~': operator.invert,
            'print': print,
            'input': input,
            'cast_int': int,
            'cast_float': float,
        }
        self.functions = {}
        self.constants = {}
        self.counter = 0
        self.env = {}
        self.outer = None
        self.namespaces = []

    def new_label(self, label):
        self.counter += 1
        try:
            namespace = '.'.join(self.namespaces)
        except:
            print(self.namespaces)
            raise
        return f':{namespace}.{label}-{self.counter}'

    def assign(self, name, value):
        self.env[name] = value

    def find(self, name, level=0):
        if name in self.env:
            return self.env, level
        elif self.outer:
            return self.outer.find(name, level + 1)
        else:
            return None, None

    def lookup(self, name):
        env, level = self.find(name)
        if env:
            return env[name], level
        else:
            raise RuntimeError(f"undefined variable {name}")

    def child(self, namespace):
        child = Context()
        child.internal = self.internal
        child.outer = self
        child.functions = self.functions
        child.constants = self.constants
        child.namespaces = self.namespaces + [namespace]
        return child


def head(expr):
    if isinstance(expr, Expression) and len(expr) == 0:
        return 'None'
    if isinstance(expr, Expression) and isinstance(expr[0], Symbol):
        return expr[0].name
    if isinstance(expr, Expression):
        return f"OP{expr[0]}"
    if isinstance(expr, Symbol):
        return 'Symbol'
    return 'Atom'


def compile_call(expr, ctx, ret):
    func, *args = expr
    if isinstance(func, Symbol) and func.name == '!asm':
        return list(args)

    code = []
    for arg in reversed(args):
        code.extend(compile_expr(arg, ctx, False))
    code.extend([
        len(args), *compile_expr(func, ctx, ret),
        OPCODE.CALL_RET if ret else OPCODE.CALL
    ])

    return code


def compile_define(expr, ctx, ret):
    is_function = False

    name, value = expr[1:]
    name = name.name
    label = ctx.new_label(name)
    code = compile_expr(value, ctx, False)

    ret = []

    if len(code) == 1 and isinstance(code[0], str):
        if 'lambda' in code[0]:
            is_function = True
            func_code = ctx.functions[code[0]]
            for i in range(len(func_code)):
                if isinstance(func_code[i],
                              str) and func_code[i] == f":external_ref:{name}":
                    func_code[i] = label

    if is_function:
        ctx.functions[label] = ctx.functions.pop(code[0])
    else:
        if len(code) == 2 and code[1] == OPCODE.LOAD:
            label = code[0]
        elif len(code) == 1:
            ctx.constants[label] = value
        else:
            ctx.constants[label] = 0
            ret = [*code, label, OPCODE.STORE]
    ctx.assign(name, label)
    return ret


def compile_lambda(expr, ctx, ret):
    args, body = expr[1:]
    args = [arg.name for arg in args[::-1]]
    label = ctx.new_label('lambda')
    ctx.functions[label] = []
    sub_ctx = ctx.child(label)
    for i, arg in enumerate(args):
        sub_ctx.assign(arg, i)
    code = compile_function_body(body, sub_ctx)
    ctx.functions[label] = code
    return [label]


def compile_function_body(expr, ctx):
    return [*compile_expr(expr, ctx, True), OPCODE.RET]


def compile_symbol(expr, ctx, ret):
    if expr.name in ctx.internal:
        return [ctx.internal[expr.name]]

    try:
        ref, level = ctx.lookup(expr.name)
    except:
        return [f":external_ref:{expr.name}"]

    if ref in ctx.functions:
        return [ref]
    if ref in ctx.constants:
        return [ref, OPCODE.LOAD]

    return [ref, level, OPCODE.SLOAD]


def compile_value(expr, ctx, ret):
    if isinstance(expr, (int, float)):
        return [expr]
    label = ctx.new_label('value')
    ctx.constants[label] = expr
    return [label, OPCODE.LOAD]


def compile_if(expr, ctx, ret):
    cond, then, else_ = expr[1:]
    else_label = ctx.new_label('else')
    end_label = ctx.new_label('end')
    code = compile_expr(cond, ctx, False)
    code.extend([else_label, OPCODE.JE])
    code.extend([*compile_expr(then, ctx, ret), end_label, OPCODE.JMP])
    code.extend([f"label{else_label}"])
    code.extend(compile_expr(else_, ctx, ret))
    code.extend([f"label{end_label}"])
    return code


def compile_cond(expr, ctx, ret):
    # TODO
    code = []
    return code


def compile_begin(expr, ctx, ret):
    code = []
    for e in expr[1:-1]:
        c = compile_expr(e, ctx, False)
        code.extend(c)
        if len(c) > 0:
            code.append(OPCODE.DROP)
    code.extend(compile_expr(expr[-1], ctx, ret))
    return code


def compile_setq(expr, ctx, ret):
    name, value = expr[1:]
    name = name.name
    ref, level = ctx.lookup(name)
    code = compile_expr(value, ctx, False)
    code.extend([OPCODE.DUP, ref, level, OPCODE.SSTORE])
    return code


def compile_let(expr, ctx, ret):
    _, bindings, body = expr
    args = []
    params = []
    for name, value in bindings:
        args.append(name)
        params.append(value)
    expr = Expression(
        [Expression([Symbol('lambda'),
                     Expression(args), body]), *params])
    return compile_expr(expr, ctx, ret)


def compile_let_star(expr, ctx, ret):
    _, bindings, body = expr
    tmp_bindings = []
    inner_bindings = []
    args = []
    for name, _ in bindings:
        args.append(name)
        tmp_bindings.append(Expression([name, 0]))
        inner_bindings.append(Expression([name, name]))
    expr = Expression([
        Symbol('let'),
        Expression([*tmp_bindings]),
        Expression([
            Symbol('begin'), *[
                Expression([Symbol('setq'), name, value])
                for name, value in bindings
            ],
            Expression([Symbol('let'),
                        Expression(inner_bindings), body])
        ])
    ])
    return compile_expr(expr, ctx, ret)


def compile_quote(expr, ctx, ret):
    # TODO
    return [expr]


def compile_while(expr, ctx, ret):
    cond, body = expr[1:]

    loop_start = ctx.new_label('loop')
    loop_end = ctx.new_label('end')
    code = [
        f"label{loop_start}",
        *compile_expr(cond, ctx, False),
        loop_end, OPCODE.JE,
        *compile_expr(body, ctx, ret),
        OPCODE.DROP,
        loop_start, OPCODE.JMP,
        f"label{loop_end}", None
    ] #yapf: disable
    return code


def compile_expr(expr, ctx: Context, ret: bool = False) -> list[Any]:
    dispatch_table = {
        'if': compile_if,
        'cond': compile_cond,
        'begin': compile_begin,
        'setq': compile_setq,
        'let': compile_let,
        'let*': compile_let_star,
        'quote': compile_quote,
        'while': compile_while,
        'lambda': compile_lambda,
        'define': compile_define,
    }

    if isinstance(expr, Expression):
        cond = head(expr)
        if cond == 'None':
            return [None]
        if cond in dispatch_table:
            return dispatch_table[cond](expr, ctx, ret)
        else:
            return compile_call(expr, ctx, ret)
    elif isinstance(expr, Symbol):
        return compile_symbol(expr, ctx, ret)
    else:
        return compile_value(expr, ctx, ret)


def compile(prog: str,
            extra_commands: dict[str, Any] = {},
            language='qlisp') -> list[Any]:
    ctx = Context()
    ctx.internal.update(extra_commands)
    functions = {}

    parsers = {'qlisp': qlisp_parser}
    parse = parsers[language]

    functions['main'] = compile_function_body(parse(prog), ctx)

    for name, code in ctx.functions.items():
        functions[name[1:]] = code

    constants = {}
    for name, value in ctx.constants.items():
        constants[name[1:]] = value

    lib = (functions, constants, ctx.env)

    code = link(functions, constants)
    return code  #, functions, ctx


def _link(code, functions, constants):

    def process(code):
        count = 0
        tmp = []
        labels = {}
        pointers = {}
        functions = {}
        for c in code:
            if isinstance(c, str) and c.startswith('label:'):
                labels[c[6:]] = count
            else:
                count += 1
                tmp.append(c)
                if isinstance(c, str) and c.startswith(':'):
                    pointers[count] = c[1:]
        for i, label in pointers.items():
            if label in labels:
                tmp[i - 1] = labels[label]
            else:
                functions[i - 1] = label
        return tmp, functions

    for name in list(functions.keys()):
        if isinstance(functions[name], tuple):
            continue
        functions[name] = process(functions[name])

    fun_ptrs = {}
    const_ptrs = {}
    code, fun_refs = process(code)

    for name, value in constants.items():
        const_ptrs[name] = len(code)
        code.append(value)

    for name, (c, f_refs) in functions.items():
        fun_ptrs[name] = len(code)
        code.extend(c)
        for i, n in f_refs.items():
            fun_refs[i + fun_ptrs[name]] = n
    external_refs = {}
    for addr, label in fun_refs.items():
        if label in fun_ptrs:
            code[addr] = fun_ptrs[label]
        elif label in const_ptrs:
            code[addr] = const_ptrs[label]
        else:
            external_refs[addr] = label
    return code, external_refs


def link(functions=None, constants=None, dynamic=False):
    if functions is None:
        functions = {}
    if constants is None:
        constants = {}
    if "main" not in functions and not dynamic:
        raise RuntimeError("main function not defined")

    enter_code = [0, ":main", OPCODE.CALL, OPCODE.EXIT]
    code, external_refs = _link(enter_code, functions, constants)
    if external_refs and not dynamic:
        raise RuntimeError(
            f"external references: {set(external_refs.values())}")
    return code
