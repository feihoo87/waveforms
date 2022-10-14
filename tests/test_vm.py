import operator

from waveforms.qlisp.interpreter import OPCODE, VirtualMachine, compile, link

result = None


def output(arg):
    global result
    result = arg


#!yapf: disable
prog1 = [
    1., 4.,
    2, ':sub', OPCODE.CALL,
    1, output, OPCODE.CALL_RET
]

prog2 = [
    5,  # limit
    0,  # ret
    0,  # i

    'label:startloop',
    0, 0, OPCODE.SLOAD,     # load limit
    ':end', OPCODE.JE,      # if limit == 0 goto end

    1, 0, OPCODE.SLOAD,     # load ret
    2, 0, OPCODE.SLOAD,     # load i
    2, operator.add, OPCODE.CALL,
    1, 0, OPCODE.SSTORE,    # set ret

    2, 0, OPCODE.SLOAD,
    1,
    2, operator.add, OPCODE.CALL,
    2, 0, OPCODE.SSTORE,

    1,
    0, 0, OPCODE.SLOAD,
    2, operator.sub, OPCODE.CALL,
    0, 0, OPCODE.SSTORE,

    ':startloop', OPCODE.JMP,

    'label:end',

    1, 0, OPCODE.SLOAD,     # load ret
    1, output, OPCODE.CALL_RET
]

prog3 = [
    5,
    1, ':frac', OPCODE.CALL,
    1, output, OPCODE.CALL_RET
]

prog4 = [
    15,
    1,
    ':feb',
    OPCODE.CALL,
    1,
    output,
    OPCODE.CALL_RET,
]

sub = [
    2, operator.sub, OPCODE.CALL_RET
]

frac = [
    OPCODE.DUP,
    ':ret0', OPCODE.JE,
    ':retn', OPCODE.JMP,

    'label:ret0',
    1, OPCODE.RET,

    'label:retn',
    OPCODE.DUP,
    1,
    OPCODE.SWAP,
    2, operator.sub, OPCODE.CALL,
    1, ':frac', OPCODE.CALL,
    2, operator.mul, OPCODE.CALL_RET
]

feb = [
    OPCODE.DUP,
    ':ret0', OPCODE.JE,

    OPCODE.DUP, 1, OPCODE.SWAP, 2, ":sub", OPCODE.CALL,
    ':ret1', OPCODE.JE,
    ':retn', OPCODE.JMP,

    'label:ret0',
    0, OPCODE.RET,

    'label:ret1',
    1, OPCODE.RET,

    'label:retn',
    OPCODE.DUP, 1, OPCODE.SWAP,
    2, ":sub", OPCODE.CALL,
    1, ':feb', OPCODE.CALL,
    OPCODE.SWAP,

    2, OPCODE.SWAP,
    2, ":sub", OPCODE.CALL,
    1, ':feb', OPCODE.CALL,
    2, operator.add, OPCODE.CALL_RET,
]
#!yapf: enable

functions = {
    "sub": sub,
    "frac": frac,
    "feb": feb,
}


def test_VM1():
    vm = VirtualMachine(debug=False)
    functions['main'] = prog1
    code = link(functions)
    vm.run(code)
    assert result == 3.0


def test_VM2():
    vm = VirtualMachine(debug=False)
    functions['main'] = prog2
    code = link(functions)
    vm.run(code)
    assert result == 10


def test_VM3():
    vm = VirtualMachine(debug=False)
    functions['main'] = prog3
    code = link(functions)
    vm.run(code)
    assert result == 120


def test_VM4():
    vm = VirtualMachine(debug=False)
    functions['main'] = prog4
    code = link(functions)
    vm.run(code)
    assert result == 610


prog5 = """
(output
((lambda (a b c d n t) (begin
    (while (!= n 0)
    (begin
        (if (== (% n 2) 1)
        (begin
            (setq t (+ (* a c) (+ (* b c) (* a d))))
            (setq b (+ (* a c) (* b d)))
            (setq a t))
        ())
    (setq n (>> n 1))
    (setq t (+ (* c c) (+ (* d c) (* c d))))
    (setq d (+ (* c c) (* d d)))
    (setq c t)))
(+ a b)))
1 -1 1 0 (cast_int {}) 0))
"""

prog6 = """
(begin

(define f1 (lambda (a b c d) (+ (* a c) (+ (* b c) (* a d)))))
(define f2 (lambda (a b c d) (+ (* a c) (* b d))))

(output
((lambda (a b c d n t) (begin
    (while (!= n 0)
    (begin
        (if (== (% n 2) 1)
        (begin
            (setq t (f1 a b c d))
            (setq b (f2 a b c d))
            (setq a t))
        ())
    (setq n (>> n 1))
    (setq t (f1 c d c d))
    (setq d (f2 c d c d))
    (setq c t)))
(+ a b)))
1 -1 1 0 (cast_int {}) 0)))
"""

prog7 = """
(begin

(define frac (lambda (n) (if (== n 0) 1 (* n (frac (- n 1))))))

(output (frac {})))
"""

prog8 = """
(begin
    (define frac (lambda (n ret)
                         (if (== n 0)
                             ret
                             (frac (- n 1) (* ret n)))))
    (output (frac {} 1)))
"""


def feb(n):

    def feb_iter(a, b, c, d):
        return a * c + b * c + a * d, a * c + b * d

    a, b = 1, -1
    c, d = 1, 0
    while n != 0:
        if n % 2 == 1:
            a, b = feb_iter(a, b, c, d)
        n >>= 1
        c, d = feb_iter(c, d, c, d)
    return a + b


def frac(n):
    ret = 1
    while n != 0:
        ret = ret * n
        n = n - 1
    return ret


def test_comiler1():
    code = compile(prog5.format(10), extra_commands={'output': output})
    vm = VirtualMachine(debug=False)
    vm.run(code)
    assert result == feb(10)


def test_comiler2():
    code = compile(prog6.format(10), extra_commands={'output': output})
    vm = VirtualMachine(debug=False)
    vm.run(code)
    assert result == feb(10)


def test_comiler3():
    code = compile(prog7.format(10), extra_commands={'output': output})
    vm = VirtualMachine(debug=False)
    vm.run(code)
    assert result == frac(10)


def test_comiler4():
    code = compile(prog8.format(100), extra_commands={'output': output})
    vm = VirtualMachine(debug=False)
    vm.run(code)
    assert result == frac(100)
    assert len(vm.stack) < 20
