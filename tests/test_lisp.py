from numpy import inf, pi
from waveforms import Waveform, sin, square
from waveforms.qlisp.parse import *


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
        yield ('!set_waveform', Channel(('Q1', ), 'test'), sin(12e6))
        yield ('!set_phase', ('quote', 'Q1'), ('+', x, x))
        return ('double', x)

    eval(Expression((f, 8)), env, stack)
    assert stack.pop() == 16
    assert Channel(('Q1', ), 'test') in stack.raw_waveforms
    assert stack.raw_waveforms[Channel(('Q1', ), 'test')] == sin(12e6)
    assert stack.phases_ext[Symbol('Q1')][1] == 16

    eval(
        parse("""
        (waveform inf -inf nil nil nil
            3 -1e-07 0 5e-07 1 1.0 2 1 3 2 1.20112240878645e-07
            2e-07 1 3 4 100000000.0 0.0 inf 0)
    """), env, stack)
    assert stack.pop() == Waveform.fromlist([
        inf, -inf, None, None, None, 3, -1e-07, 0, 5e-07, 1, 1.0, 2, 1, 3, 2,
        1.20112240878645e-07, 2e-07, 1, 3, 4, 100000000.0, 0.0, inf, 0
    ])

    eval(
        parse("""
        (* (waveform "square(10)")
        (waveform "sin(10)"))
    """), env, stack)
    assert stack.pop() == (square(10) * sin(10))


def test_gate():
    from waveforms import wave_eval
    stack = Stack()
    env = standard_env()

    eval(
        parse("""
        (gate Y (qubit) ((rfUnitary pi (/ pi 2)) qubit))
    """), env, stack)
    assert stack.pop() is None

    eval(
        parse("""
        (gate (rfUnitary theta phi) (qubit)
            (!set_waveform (channel qubit "RF") (waveform "gaussian(20e-9) * sin(4e9)"))
            (!set_phase (channel qubit "RF") phi)
            )
    """), env, stack)
    assert stack.pop() is None

    eval(parse("""
        (rfUnitary pi 0)
    """), env, stack)
    gate = stack.pop()
    assert gate.name == 'rfUnitary'
    assert gate.params == ['theta', 'phi']
    assert gate.bindings == {'theta': pi, 'phi': 0}

    eval(parse("""
        (Y 'Q1)
    """), env, stack)
    assert stack.pop() is None
    assert stack.raw_waveforms[Channel(
        qubits='RF',
        name=('qubit', ))] == wave_eval("gaussian(20e-9) * sin(4e9)")
    assert stack.phases_ext[Channel(qubits='RF',
                                    name=('qubit', ))][1] == pi / 2
