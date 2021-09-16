from pathlib import Path

from numpy import mod, pi
from waveforms.waveform import (cos, cosPulse, gaussian, mixing, sin, square,
                                zero)
from waveforms.waveform_parser import wave_eval

from .library import Library, Parameter
from .qlisp import MeasurementTask

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


@std.opaque('Delay')
def delay(ctx, qubits, time):
    qubit, = qubits
    ctx.time[qubit] += time


@std.opaque('P')
def P(ctx, qubits, phi):
    from .compiler import call_opaque

    phi += ctx.phases[qubits[0]]
    ctx.phases[qubits[0]] = 0

    call_opaque((('rfUnitary', pi / 2, pi / 2), *qubits), ctx, std)
    call_opaque((('rfUnitary', phi, 0), *qubits), ctx, std)
    call_opaque((('rfUnitary', pi / 2, -pi / 2), *qubits), ctx, std)


@std.opaque('Barrier')
def barrier(ctx, qubits):
    time = max(ctx.time[qubit] for qubit in qubits)
    for qubit in qubits:
        ctx.time[qubit] = time


@std.opaque('rfPulse')
def rfPulse(ctx, qubits, waveform):
    qubit, = qubits

    if isinstance(waveform, str):
        waveform = wave_eval(waveform)

    ctx.channel['RF', qubit] += waveform >> ctx.time[qubit]


@std.opaque('fluxPulse')
def fluxPulse(ctx, qubits, waveform):
    qubit, = qubits

    if isinstance(waveform, str):
        waveform = wave_eval(waveform)

    ctx.channel['Z', qubit] += waveform >> ctx.time[qubit]


@std.opaque('Pulse')
def Pulse(ctx, qubits, channel, waveform):

    if isinstance(waveform, str):
        waveform = wave_eval(waveform)

    t = max(ctx.time[qubit] for qubit in qubits)

    ctx.channel[(channel, *qubits)] += waveform >> t


@std.opaque('setBias')
def setBias(ctx, qubits, channel, bias):
    ctx.biases[(channel, *qubits)] = bias


@std.opaque('rfUnitary',
            params=[
                Parameter('shape', str, 'cosPulse'),
                Parameter('amp', list, [[0, 1], [0, 0.653]]),
                Parameter('duration', list, [[0, 1], [10e-9, 10e-9]]),
                Parameter('phase', list, [[-1, 1], [-1, 1]]),
                Parameter('frequency', float, 5e9, 'Hz'),
                Parameter('DRAGScaling', float, 1e-10, 'a.u.')
            ])
def rfUnitary(ctx, qubits, theta, phi):
    import numpy as np

    qubit, = qubits

    if theta < 0:
        theta = -theta
        phi += pi
    theta = mod(theta, 2 * pi)
    if theta > pi:
        theta = 2 * pi - theta
        phi += pi

    phi = mod(phi - ctx.phases[qubit], 2 * pi)
    if phi > pi:
        phi -= 2 * pi

    shape = ctx.params['shape']
    duration = np.interp(theta / np.pi, *ctx.params['duration'])
    amp = np.interp(theta / np.pi, *ctx.params['amp'])
    phase = np.pi * np.interp(phi / np.pi, *ctx.params['phase'])

    pulseLib = {
        'CosPulse': cosPulse,
        'Gaussian': gaussian,
        'square': square,
        'DC': square,
    }

    if duration > 0 and amp != 0:
        pulse = pulseLib[shape](duration) >> (duration / 2 + ctx.time[qubit])
        pulse, _ = mixing(pulse,
                          phase=phase,
                          freq=ctx.params['frequency'],
                          DRAGScaling=ctx.params['DRAGScaling'])
        ctx.channel['RF', qubit] += amp * pulse
    else:
        ctx.channel['RF', qubit] += zero()
    ctx.time[qubit] += duration


@std.opaque('Measure',
            params=[
                Parameter('duration', float, 1e-6, 's'),
                Parameter('amp', float, 0.1, 'a.u.'),
                Parameter('frequency', float, 6.5e9, 'Hz'),
                Parameter('signal', str, 'state'),
                Parameter('weight', str, 'const(1)'),
                Parameter('phi', float, 0),
                Parameter('threshold', float, 0),
                Parameter('ring_up_amp', float, 0.1, 'a.u.'),
                Parameter('ring_up_time', float, 50e-9, 's')
            ])
def measure(ctx, qubits, cbit=None):
    import numpy as np
    from waveforms import exp, step

    qubit, = qubits

    if cbit is None:
        if len(ctx.measures) == 0:
            cbit = 0
        else:
            cbit = max(ctx.measures.keys()) + 1

    # lo = ctx.cfg._getReadoutADLO(qubit)
    amp = ctx.params['amp']
    duration = ctx.params['duration']
    frequency = ctx.params['frequency']
    signal = ctx.params.get('signal', 'state')
    ring_up_amp = ctx.params.get('ring_up_amp', amp)
    ring_up_time = ctx.params.get('ring_up_time', 50e-9)

    try:
        w = ctx.params['w']
        weight = None
    except:
        weight = ctx.params.get('weight',
                                f'square({duration}) >> {duration/2}')
        w = None

    t = ctx.time[qubit]

    # phi = 2 * np.pi * (lo - frequency) * t

    pulse = (ring_up_amp * (step(0) >> t) - (ring_up_amp - amp) *
             (step(0) >> (t + ring_up_time)) - amp * (step(0) >>
                                                      (t + duration)))
    ctx.channel['readoutLine.RF',
                qubit] += amp * pulse * cos(2 * pi * frequency)

    # pulse = square(2 * duration) >> duration
    # ctx.channel['readoutLine.AD.trigger', qubit] |= pulse.marker

    params = {k: v for k, v in ctx.params.items()}
    params['w'] = w
    params['weight'] = weight
    ctx.measures[cbit].append(
        MeasurementTask(qubit, cbit, ctx.time[qubit], signal, params))
    ctx.time[qubit] = t + duration
    ctx.phases[qubit] = 0


def parametric(ctx, qubits):
    t = max(ctx.time[q] for q in qubits)

    duration = ctx.params['duration']
    amp = ctx.params['amp']
    offset = ctx.params['offset']
    frequency = ctx.params['frequency']

    if duration > 0:
        pulse = square(duration) >> duration / 2
        pulse = offset * pulse + amp * pulse * sin(2 * pi * frequency)
        ctx.channel[('coupler.Z', *qubits)] += pulse >> t

    for qubit in qubits:
        ctx.time[qubit] = t + duration

    ctx.phases[qubits[0]] += ctx.params['phi1']
    ctx.phases[qubits[1]] += ctx.params['phi2']


@std.opaque('CZ',
            params=[
                Parameter('duration', float, 50e-9, 's'),
                Parameter('amp', float, 0.8, 'a.u.'),
                Parameter('phi1', float, 0),
                Parameter('phi2', float, 0)
            ])
def CZ(ctx, qubits):
    t = max(ctx.time[q] for q in qubits)

    duration = ctx.params['duration']
    amp = ctx.params['amp']

    if amp > 0 and duration > 0:
        pulse = amp * (cos(pi / duration) * square(duration)) >> duration / 2
        ctx.channel[('coupler.Z', *qubits)] += pulse >> t

    for qubit in qubits:
        ctx.time[qubit] = t + duration

    ctx.phases[qubits[0]] += ctx.params['phi1']
    ctx.phases[qubits[1]] += ctx.params['phi2']


@std.opaque('CZ',
            type='parametric',
            params=[
                Parameter('duration', float, 50e-9, 's'),
                Parameter('amp', float, 0.8, 'a.u.'),
                Parameter('offset', float, 0, 'a.u.'),
                Parameter('frequency', float, 0, 'Hz'),
                Parameter('phi1', float, 0),
                Parameter('phi2', float, 0)
            ])
def CZ(ctx, qubits):
    parametric(ctx, qubits)


@std.opaque('iSWAP',
            type='parametric',
            params={
                Parameter('duration', float, 50e-9, 's'),
                Parameter('amp', float, 0.8, 'a.u.'),
                Parameter('offset', float, 0, 'a.u.'),
                Parameter('frequency', float, 0, 'Hz'),
                Parameter('phi1', float, 0),
                Parameter('phi2', float, 0)
            })
def iSWAP(ctx, qubits):
    parametric(ctx, qubits)
