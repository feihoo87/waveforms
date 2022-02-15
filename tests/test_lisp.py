from numpy import inf
from waveforms import Waveform, sin, square
from waveforms.quantum.circuit.qlisp.lisp import *


def test_lisp_eval():
    stack = Stack()
    env = standard_env()

    eval(parse("""
        (define double (lambda (x) (+ x x)))
    """), env, stack)
    assert stack.pop() is None

    eval(
        parse("""
        (defun f (x y)
            (setq x (+ x y))
            (* x y))
    """), env, stack)
    assert stack.pop() is None

    eval(parse("""
        (double 2)
    """), env, stack)
    assert stack.pop() == 4

    eval(parse("""
        (f 2 6)
    """), env, stack)
    assert stack.pop() == 48

    eval(
        parse("""
        (let
        ((x 1)
        (y 2))
        (while (< x 10)       ; while loop
            (begin            ; test comment
            (setq x (+ x 1))
            (setq y (+ x y))
            (display! x y (f x y))
            (* x y))))
    """), env, stack)
    assert stack.pop() == 560

    eval(parse("""
        'double
    """), env, stack)
    assert stack.pop() == Symbol('double')

    eval(parse("""
        ((lambda (x) (* x x)) 11)
    """), env, stack)
    assert stack.pop() == 121

    def f(x):
        yield ('!set_waveform', 'test', sin(12e6))
        yield ('!set_phase', 'test', ('+', x, x))
        return ('double', x)

    eval(Expression((f, 8)), env, stack)
    assert stack.pop() == 16
    assert stack.raw_waveforms['test'] == sin(12e6)
    assert stack.phases_ext['test'][1] == 16

    eval(
        parse("""
        (waveform inf -inf nil nil nil
            3 -1e-07 5e-07 inf 0 1 1.0 2
            1 1 3 2 1.20112240878645e-07 2e-07
            3 4 100000000.0 0.0 0)
    """), env, stack)
    assert stack.pop() == Waveform.fromlist([
        inf, -inf, None, None, None, 3, -1e-07, 5e-07, inf, 0, 1, 1.0, 2, 1, 1,
        3, 2, 1.20112240878645e-07, 2e-07, 3, 4, 100000000.0, 0.0, 0
    ])

    eval(
        parse("""
        (* (waveform "square(10)")
        (waveform "sin(10)"))
    """), env, stack)
    assert stack.pop() == (square(10) * sin(10))
