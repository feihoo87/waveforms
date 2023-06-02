from os import name
from pathlib import Path
from typing import Type

import numpy as np
from pkg_resources import yield_lines
from waveforms.waveform import mixing, step, square, D, zero, exp, cosPulse, gaussian
from waveforms import stdlib, libraries
from waveforms.quantum.circuit.qlisp.stdlib import (MeasurementTask, Parameter,
                                                    _rfUnitary)

lib = libraries(stdlib)
lib.qasmLib = {
    'qelib1.inc': Path(__file__).parent / 'qelib1.inc'
}


def _rfUnitary(ctx, qubits, theta, phi, level1=0, level2=1):

    from numpy import mod, pi

    qubit, = qubits

    if theta < 0:
        theta = -theta
        phi += pi
    theta = mod(theta, 2 * pi)
    if theta > pi:
        theta = 2 * pi - theta
        phi += pi

    phi = mod(phi + ctx.phases_ext[qubit][level1] -
              ctx.phases_ext[qubit][level2], 2 * pi)
    phi = phi if abs(level2-level1) % 2 == 0 else phi-np.pi
    if phi > pi:
        phi -= 2 * pi
    phi = phi/abs(level2-level1)

    shape = ctx.params.get('shape', 'CosPulse')
    buffer = ctx.params.get('buffer', 0)
    alpha = ctx.params.get('alpha', 1)
    beta = ctx.params.get('beta', 0)
    delta = ctx.params.get('delta', 0)

    phase = np.pi * np.interp(phi / np.pi, *ctx.params['phase'])

    pulseLib = {
        'CosPulse': cosPulse,
        'Gaussian': gaussian,
        'square': square,
        'DC': square,
    }

    while theta > 0:
        if theta > pi / 2:
            theta1 = pi / 2
            theta -= pi / 2
        else:
            theta1 = theta
            theta = 0

        duration = np.interp(theta1 / np.pi, *ctx.params['duration'])
        amp = np.interp(theta1 / np.pi, *ctx.params['amp'])
        pulse = pulseLib[shape](duration)

        if duration > 0 and amp > 0:
            I, Q = mixing(amp * alpha * pulse,
                          phase=phase,
                          freq=delta,
                          DRAGScaling=beta / alpha)
            t = (duration + buffer) / 2 + ctx.time[qubit]
            wav, _ = mixing(I >> t, Q >> t, freq=ctx.params['frequency'])
            yield ('!add', 'waveform', wav), ('RF', qubit)
            yield ('!add', 'time', duration + buffer), qubit


@lib.opaque('rfUnitary12',
            params=[
                Parameter('shape', str, 'CosPulse'),
                Parameter('amp', list, [[0, 1], [0, 0.653]]),
                Parameter('duration', list, [[0, 1], [10e-9, 10e-9]]),
                Parameter('phase', list, [[-1, 1], [-1, 1]]),
                Parameter('frequency', float, 5e9, 'Hz'),
                Parameter('alpha', float, 1, 'Hz'),
                Parameter('beta', float, 0, 'Hz'),
                Parameter('delta', float, 0, 'Hz'),
                Parameter('buffer', float, 0, 's'),
            ])
def rfUnitary12(ctx, qubits, theta, phi):
    yield from _rfUnitary(ctx, qubits, theta, phi, 1, 2)


@lib.opaque('rfUnitary23',
            params=[
                Parameter('shape', str, 'CosPulse'),
                Parameter('amp', list, [[0, 1], [0, 0.653]]),
                Parameter('duration', list, [[0, 1], [10e-9, 10e-9]]),
                Parameter('phase', list, [[-1, 1], [-1, 1]]),
                Parameter('frequency', float, 5e9, 'Hz'),
                Parameter('alpha', float, 1, 'Hz'),
                Parameter('beta', float, 0, 'Hz'),
                Parameter('delta', float, 0, 'Hz'),
                Parameter('buffer', float, 0, 's'),
            ])
def rfUnitary23(ctx, qubits, theta, phi):
    yield from _rfUnitary(ctx, qubits, theta, phi, 2, 3)


@lib.opaque('rfUnitary34',
            params=[
                Parameter('shape', str, 'CosPulse'),
                Parameter('amp', list, [[0, 1], [0, 0.653]]),
                Parameter('duration', list, [[0, 1], [10e-9, 10e-9]]),
                Parameter('phase', list, [[-1, 1], [-1, 1]]),
                Parameter('frequency', float, 5e9, 'Hz'),
                Parameter('alpha', float, 1, 'Hz'),
                Parameter('beta', float, 0, 'Hz'),
                Parameter('delta', float, 0, 'Hz'),
                Parameter('buffer', float, 0, 's'),
            ])
def rfUnitary34(ctx, qubits, theta, phi):
    yield from _rfUnitary(ctx, qubits, theta, phi, 3, 4)


@lib.opaque('_rfUnitary02',
            params=[
                Parameter('shape', str, 'CosPulse'),
                Parameter('amp', list, [[0, 1], [0, 0.653]]),
                Parameter('duration', list, [[0, 1], [10e-9, 10e-9]]),
                Parameter('phase', list, [[-1, 1], [-1, 1]]),
                Parameter('frequency', float, 5e9, 'Hz'),
                Parameter('alpha', float, 1, 'Hz'),
                Parameter('beta', float, 0, 'Hz'),
                Parameter('delta', float, 0, 'Hz'),
                Parameter('buffer', float, 0, 's'),
            ])
def rfUnitary02(ctx, qubits, theta, phi):
    yield from _rfUnitary(ctx, qubits, theta, phi, 0, 2)


@lib.gate(1, name='rfUnitary02')
def rfUnitary02(qubits, theta, phi):
    yield (('rfUnitary01', np.pi, np.pi/2), qubits)
    yield (('rfUnitary12', theta, phi), qubits)
    yield (('rfUnitary01', np.pi, -np.pi/2), qubits)


@lib.opaque('_rfUnitary13',
            params=[
                Parameter('shape', str, 'CosPulse'),
                Parameter('amp', list, [[0, 1], [0, 0.653]]),
                Parameter('duration', list, [[0, 1], [10e-9, 10e-9]]),
                Parameter('phase', list, [[-1, 1], [-1, 1]]),
                Parameter('frequency', float, 5e9, 'Hz'),
                Parameter('alpha', float, 1, 'Hz'),
                Parameter('beta', float, 0, 'Hz'),
                Parameter('delta', float, 0, 'Hz'),
                Parameter('buffer', float, 0, 's'),
            ])
def rfUnitary13(ctx, qubits, theta, phi):
    yield from _rfUnitary(ctx, qubits, theta, phi, 1, 3)


@lib.gate(1, name='rfUnitary13')
def rfUnitary13(qubits, theta, phi):
    yield (('rfUnitary12', np.pi, np.pi/2), qubits)
    yield (('rfUnitary23', theta, phi), qubits)
    yield (('rfUnitary12', np.pi, -np.pi/2), qubits)


@lib.opaque('_rfUnitary24',
            params=[
                Parameter('shape', str, 'CosPulse'),
                Parameter('amp', list, [[0, 1], [0, 0.653]]),
                Parameter('duration', list, [[0, 1], [10e-9, 10e-9]]),
                Parameter('phase', list, [[-1, 1], [-1, 1]]),
                Parameter('frequency', float, 5e9, 'Hz'),
                Parameter('alpha', float, 1, 'Hz'),
                Parameter('beta', float, 0, 'Hz'),
                Parameter('delta', float, 0, 'Hz'),
                Parameter('buffer', float, 0, 's'),
            ])
def rfUnitary24(ctx, qubits, theta, phi):
    yield from _rfUnitary(ctx, qubits, theta, phi, 2, 4)


@lib.gate(1, name='rfUnitary24')
def rfUnitary24(qubits, theta, phi):
    yield (('rfUnitary23', np.pi, np.pi/2), qubits)
    yield (('rfUnitary34', theta, phi), qubits)
    yield (('rfUnitary23', np.pi, -np.pi/2), qubits)


@lib.opaque('_rfUnitary03',
            params=[
                Parameter('shape', str, 'CosPulse'),
                Parameter('amp', list, [[0, 1], [0, 0.653]]),
                Parameter('duration', list, [[0, 1], [10e-9, 10e-9]]),
                Parameter('phase', list, [[-1, 1], [-1, 1]]),
                Parameter('frequency', float, 5e9, 'Hz'),
                Parameter('alpha', float, 1, 'Hz'),
                Parameter('beta', float, 0, 'Hz'),
                Parameter('delta', float, 0, 'Hz'),
                Parameter('buffer', float, 0, 's'),
            ])
def rfUnitary03(ctx, qubits, theta, phi):
    yield from _rfUnitary(ctx, qubits, theta, phi, 0, 3)


@lib.gate(1, name='rfUnitary03')
def rfUnitary24(qubits, theta, phi):
    yield (('rfUnitary01', np.pi, np.pi/2), qubits)
    yield (('rfUnitary12', np.pi, np.pi/2), qubits)
    yield (('rfUnitary23', theta, phi), qubits)
    yield (('rfUnitary12', np.pi, -np.pi/2), qubits)
    yield (('rfUnitary01', np.pi, -np.pi/2), qubits)


@lib.opaque('Phase')
def Phase(ctx, qubits, levels, phis):

    qubit, = qubits
    for i, level in enumerate(levels):
        phi = phis[i] + ctx.phases_ext[qubit][level]
        yield ('!set', 'phase_ext', level, phi), qubit


def _phase2rfUnitary(ctx, qubits, phis, max_level):
    phis_ave = phis-np.mean(phis)
    for level in range(max_level-1, 0, -1):
        yield from _rfUnitary(ctx, qubits, np.pi/2, np.pi/2, level-1, level)
        yield from _rfUnitary(ctx, qubits, phis_ave[level]*2, 0, level-1, level)
        yield from _rfUnitary(ctx, qubits, np.pi/2, -np.pi/2, level-1, level)
        phis_ave[level-1] += phis_ave[level]


@lib.opaque('P')
def Phase(ctx, qubits, max_level):

    qubit, = qubits
    phis = [ctx.phases_ext[qubit][level] for level in range(max_level)]
    for level in enumerate(max_level):
        yield ('!set', 'phase_ext', level, 0), qubit
    yield from _phase2rfUnitary(ctx, qubits, phis, max_level)


@lib.gate(1, name='_rfUnitary_binary')
def _rfUnitary_binary(q, theta, level1=0, level2=1):
    min_level, max_level = min(level1, level2), max(level1, level2)
    for level in range(min_level, max_level-1):
        yield ((f'rfUnitary{level}{level+1}', np.pi, np.pi/2), q)
    yield ((f'rfUnitary{max_level-1}{max_level}', np.pi/2, 0), q)
    yield ((f'rfUnitary{max_level-1}{max_level}', np.pi/2, np.pi-theta), q)
    for level in range(max_level-1, min_level, -1):
        yield ((f'rfUnitary{level-1}{level}', np.pi, -np.pi/2), q)


@lib.gate(1, name='UU')
def UU(q, theta, phi, lambda_, delta=0, level1=0, level2=1):

    EPS = 1e-7

    if abs(theta) < EPS:
        yield (('Phase', (level1, level2), (delta-(phi+lambda_)/2, delta+(phi+lambda_)/2)), q)
    else:
        yield (('Phase', (level1, level2), (-lambda_/2, lambda_/2)), q)
        # yield ((f'rfUnitary{level1}{level2}', theta, np.pi/2), q)
        if abs(theta - np.pi / 2) < EPS:
            yield ((f'rfUnitary{level1}{level2}', np.pi/2, np.pi/2), q)
        else:
            yield (('_rfUnitary_binary', theta, level1, level2), q)
            # yield ((f'rfUnitary{level1}{level2}', np.pi/2, 0), q)
            # yield ((f'rfUnitary{level1}{level2}', np.pi/2, np.pi-theta), q)
            yield (('Phase', (level1, level2), (-theta/2, theta/2)), q)
        yield (('Phase', (level1, level2), (delta-phi/2, delta+phi/2)), q)

    # yield ((f'rfUnitary{level1}{level2}', pi/2, -lambda_), q)
    # yield ((f'rfUnitary{level1}{level2}', pi/2, pi-theta-lambda_), q)
    # yield (('Phase', (level1, level2), (delta-(theta+phi+lambda_)/2, delta+(theta+phi+lambda_)/2)), q)


@lib.gate(1, name='UU2')
def UU2(qubit, phi, lambda_, delta=np.pi/2, level1=0, level2=1):
    yield (('UU', np.pi/2, phi, lambda_, delta, level1, level2), qubit)


@lib.gate(1, name='UU1')
def u1(qubit, lambda_, delta=np.pi/2, level1=0, level2=1):
    yield (('UU', 0, 0, lambda_, delta, level1, level2), qubit)


@lib.gate(1, name='H')
def H(qubit, level1=0, level2=1):
    yield (('UU2', 0, np.pi, np.pi/2, level1, level2), qubit)


@lib.gate(1, name='X')
def X(q, level1=0, level2=1):
    yield (('UU', np.pi, -np.pi/2, np.pi/2, 0, level1, level2), q)
    # yield (('UU', np.pi, 0, np.pi, np.pi/2, level1, level2), q)


@lib.gate(1, name='-X')
def X(q, level1=0, level2=1):
    yield (('UU', np.pi, np.pi/2, -np.pi/2, 0, level1, level2), q)
    # yield (('UU', np.pi, 0, -np.pi, -np.pi/2, level1, level2), q)


@lib.gate(1, name='Y')
def Y(q, level1=0, level2=1):
    yield (('UU', np.pi, 0, 0, 0, level1, level2), q)
    # yield (('UU', np.pi, np.pi/2, np.pi/2, np.pi/2, level1, level2), q)


@lib.gate(1, name='-Y')
def Y(q, level1=0, level2=1):
    yield (('UU', np.pi, -np.pi, np.pi, 0, level1, level2), q)
    # yield (('UU', np.pi, -np.pi/2, -np.pi/2, np.pi/2, level1, level2), q)


@lib.gate(1, name='Z')
def Z(q, level1=0, level2=1):
    yield (('UU1', np.pi, np.pi/2, level1, level2), q)


@lib.gate(1, name='S')
def S(q, level1=0, level2=1):
    yield (('UU1', np.pi/2, np.pi/4, level1, level2), q)


@lib.gate(1, name='-S')
def Sdg(q, level1=0, level2=1):
    yield (('UU1', -np.pi/2, -np.pi/4, level1, level2), q)


@lib.gate(1, name='T')
def T(q, level1=0, level2=1):
    yield (('UU1', np.pi/4, np.pi/8, level1, level2), q)


@lib.gate(1, name='-T')
def Tdg(q, level1=0, level2=1):
    yield (('UU1', -np.pi/4, -np.pi/8, level1, level2), q)


@lib.gate(1, name='X/2')
def sx(q, level1=0, level2=1):
    yield (('UU', np.pi/2, -np.pi/2, np.pi/2, 0, level1, level2), q)
    # yield (('UU', np.pi/2, -np.pi/2, np.pi/2, np.pi/4, level1, level2), q)


@lib.gate(1, name='-X/2')
def sxdg(q, level1=0, level2=1):
    yield (('UU', np.pi/2, -np.pi*3/2, -np.pi/2, -np.pi, level1, level2), q)
    # yield (('UU', np.pi/2, np.pi/2, -np.pi/2, -np.pi/4, level1, level2), q)


@lib.gate(1, name='Y/2')
def sy(q, level1=0, level2=1):
    yield (('UU', np.pi/2, 0, 0, 0, level1, level2), q)
    # yield (('UU', np.pi/2, 0, 0, np.pi/4, level1, level2), q)


@lib.gate(1, name='-Y/2')
def sydg(q, level1=0, level2=1):
    yield (('UU', np.pi/2, -np.pi, np.pi, 0, level1, level2), q)
    # yield (('UU', -np.pi/2, 0, 0, -np.pi/4, level1, level2), q)


@lib.gate(1, name='Rx')
def Rx(q, theta, level1 = 0, level2 = 1):
    yield (('UU', theta, -np.pi/2, np.pi/2, 0, level1, level2), q)


@lib.gate(1, name='Ry')
def Ry(q, theta, level1 = 0, level2 = 1):
    yield (('UU', theta, 0, 0, 0, level1, level2), q)


@lib.gate(1, name='Rz')
def Rz(q, phi, level1 = 0, level2 = 1):
    yield (('UU', 0, 0, phi, 0, level1, level2), q)
