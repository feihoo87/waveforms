from pathlib import Path

from numpy import mod, pi
from waveforms.waveform import cos, cosPulse, gaussian, mixing, square, zero

from .library import Library, MeasurementTask

std = Library()
std.qasmLib = {
    'qelib1.inc': Path(__file__).parent.parent / 'qasm' / 'libs' / 'qelib1.inc'
}


@std.gate()
def u3(qubit, theta, phi, lambda_):
    yield (('U', theta, phi, lambda_), qubit)


@std.gate()
def u2(qubit, phi, lambda_):
    yield (('U', pi / 2, phi, lambda_), qubit)


@std.gate()
def u1(qubit, lambda_):
    yield (('U', 0, 0, lambda_), qubit)


@std.gate()
def H(qubit):
    yield (('u2', 0, pi), qubit)


@std.gate()
def U(q, theta, phi, lambda_):
    if theta == 0:
        yield (('P', phi + lambda_), q)
    else:
        yield (('P', lambda_), q)
        yield (('rfUnitary', theta, pi / 2), q)
        yield (('P', phi), q)


@std.gate()
def X(q):
    yield (('u3', pi, 0, pi), q)


@std.gate()
def Y(q):
    yield (('u3', pi, pi / 2, pi / 2), q)


@std.gate()
def Z(q):
    yield (('u1', pi), q)


@std.gate()
def S(q):
    yield (('u1', pi / 2), q)


@std.gate(name='-S')
def Sdg(q):
    yield (('u1', -pi / 2), q)


@std.gate()
def T(q):
    yield (('u1', pi / 4), q)


@std.gate(name='-T')
def Tdg(q):
    yield (('u1', -pi / 4), q)


@std.gate(name='X/2')
def sx(q):
    yield ('-S', q)
    yield ('H', q)
    yield ('-S', q)


@std.gate(name='-X/2')
def sxdg(q):
    yield ('S', q)
    yield ('H', q)
    yield ('S', q)


@std.gate(name='Y/2')
def sy(q):
    yield ('Z', q)
    yield ('H', q)


@std.gate(name='-Y/2')
def sydg(q):
    yield ('H', q)
    yield ('Z', q)


@std.gate()
def Rx(q, theta):
    yield (('u3', theta, -pi / 2, pi / 2), q)


@std.gate()
def Ry(q, theta):
    yield (('u3', theta, 0, 0), q)


@std.gate()
def Rz(q, phi):
    yield (('u1', phi), q)


@std.gate(2)
def Cnot(qubits):
    c, t = qubits
    yield ('H', t)
    yield ('CZ', (c, t))
    yield ('H', t)


@std.gate(2)
def crz(qubits, lambda_):
    c, t = qubits

    yield (('u1', lambda_ / 2), t)
    yield ('Cnot', (c, t))
    yield (('u1', -lambda_ / 2), t)
    yield ('Cnot', (c, t))


@std.opaque('rfUnitary')
def rfUnitary(ctx, qubits, theta, phi):
    qubit, = qubits

    phi = mod(phi - ctx.phases[qubit], 2 * pi)
    if phi > pi:
        phi -= pi

    gate = ctx.cfg.getGate('rfUnitary', qubit)
    shape = gate.shape(theta, phi)
    pulse = {
        'CosPulse': cosPulse,
        'Gaussian': gaussian,
        'square': square,
        'DC': square,
    }[shape['shape']](
        shape['duration']) >> (shape['duration'] / 2 + ctx.time[qubit])

    if shape['duration'] > 0 and shape['amp'] != 0:
        pulse, _ = mixing(pulse,
                          phase=shape['phase'],
                          freq=shape['frequency'],
                          DRAGScaling=shape['DRAGScaling'])
        ctx.channel['RF', qubit] += shape['amp'] * pulse
    else:
        ctx.channel['RF', qubit] += zero()
    ctx.time[qubit] += shape['duration']


@std.opaque('Delay')
def delay(ctx, qubits, time):
    qubit, = qubits
    ctx.time[qubit] += time


@std.opaque('P')
def P(ctx, qubits, phi):
    phi += ctx.phases[qubits[0]]
    ctx.phases[qubits[0]] = 0

    rfUnitary(ctx, qubits, pi / 2, pi / 2)
    rfUnitary(ctx, qubits, phi, 0)
    rfUnitary(ctx, qubits, pi / 2, -pi / 2)


@std.opaque('Barrier')
def barrier(ctx, qubits):
    time = max(ctx.time[qubit] for qubit in qubits)
    for qubit in qubits:
        ctx.time[qubit] = time


@std.opaque('Measure')
def mesure(ctx, qubits, cbit):
    qubit, = qubits

    gate = ctx.cfg.getGate('Measure', qubit)
    amp = gate.params.amp
    duration = gate.params.duration
    frequency = gate.params.frequency
    t = ctx.time[qubit]

    pulse = square(duration) >> duration / 2
    ctx.channel['readoutLine.RF',
                qubit] += amp * pulse * cos(2 * pi * frequency) >> t
    ctx.channel['readoutLine.AD.trigger', qubit] += pulse >> t

    params = {k: v for k, v in gate.params.items()}
    ctx.measures[cbit].append(
        MeasurementTask(qubit, cbit, ctx.time[qubit],
                        gate.get('signal', 'state'), params, {
                            'channel': {},
                            'params': {}
                        }))
    ctx.time[qubit] += duration
    ctx.phases[qubit] = 0
