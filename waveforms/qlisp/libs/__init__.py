from pathlib import Path

from numpy import mod, pi

from waveforms.waveform import (cos, cosPulse, gaussian, mixing, sin, square,
                                zero)
from waveforms.waveform_parser import wave_eval

from ..base import Capture
from ..library import Library

EPS = 1e-9

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
    # yield (('rfUnitary', pi / 2, -lambda_), qubit)
    # yield (('rfUnitary', pi / 2, -pi - theta - lambda_), qubit)
    # yield (('P', theta + phi + lambda_), qubit)

    if abs(theta) < EPS:
        yield (('P', phi + lambda_), q)
    else:
        yield (('P', lambda_), q)
        if abs(theta - pi / 2) < EPS:
            yield (('rfUnitary', pi / 2, pi / 2), q)
        else:
            yield (('rfUnitary', pi / 2, 0), q)
            yield (('rfUnitary', pi / 2, pi - theta), q)
            yield (('P', theta), q)
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


@std.gate(name='W/2')
def W2(q):
    yield (('u3', pi / 2, -pi / 4, pi / 4), q)


@std.gate(name='-W/2')
def W2(q):
    yield (('u3', -pi / 2, -pi / 4, pi / 4), q)


@std.gate(name='V/2')
def W2(q):
    yield (('u3', pi / 2, -3 * pi / 4, 3 * pi / 4), q)


@std.gate(name='-V/2')
def W2(q):
    yield (('u3', -pi / 2, -3 * pi / 4, 3 * pi / 4), q)


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
    yield ('!add_time', time), qubit


@std.opaque('P')
def P(ctx, qubits, phi):
    # (('rfUnitary', pi / 2, arb_phase), qubit)
    # (('rfUnitary', pi / 2, arb_phase), qubit)
    # (('rfUnitary', pi / 2, phi / 2 + arb_phase + k * pi), qubit)
    # (('rfUnitary', pi / 2, phi / 2 + arb_phase + k * pi), qubit)
    import numpy as np

    from ..compiler import call_opaque

    phi += ctx.phases[qubits[0]]
    yield ('!set_phase', 0), qubits[0]
    x = 2 * np.pi * np.random.random()
    y = np.pi * np.random.randint(0, 2) + x

    call_opaque((('rfUnitary', pi / 2, x), *qubits), ctx, std)
    call_opaque((('rfUnitary', pi / 2, x), *qubits), ctx, std)
    call_opaque((('rfUnitary', pi / 2, phi / 2 + y), *qubits), ctx, std)
    call_opaque((('rfUnitary', pi / 2, phi / 2 + y), *qubits), ctx, std)


@std.opaque('Barrier')
def barrier(ctx, qubits):
    time = max(ctx.time[qubit] for qubit in qubits)
    for qubit in qubits:
        yield ('!set_time', time), qubit


@std.opaque('rfPulse')
def rfPulse(ctx, qubits, waveform):
    qubit, = qubits

    if isinstance(waveform, str):
        waveform = wave_eval(waveform)

    yield ('!play', waveform >> ctx.time[qubit]), ('RF', qubit)


@std.opaque('fluxPulse')
def fluxPulse(ctx, qubits, waveform):
    qubit, = qubits

    if isinstance(waveform, str):
        waveform = wave_eval(waveform)

    yield ('!play', waveform >> ctx.time[qubit]), ('Z', qubit)


@std.opaque('Pulse')
def Pulse(ctx, qubits, channel, waveform):

    if isinstance(waveform, str):
        waveform = wave_eval(waveform)

    t = max(ctx.time[qubit] for qubit in qubits)

    yield ('!play', waveform >> t), (channel, *qubits)


@std.opaque('setBias')
def setBias(ctx, qubits, channel, bias, edge=0, buffer=0):
    if channel.startswith('coupler.') and len(qubits) == 2:
        qubits = sorted(qubits)
    yield ('!set_bias', (bias, edge, buffer)), (channel, *qubits)
    time = max(ctx.time[qubit] for qubit in qubits)
    for qubit in qubits:
        yield ('!set_time', time + buffer), qubit


@std.opaque('I')
def I(ctx, qubits):
    qubit = qubits[0]
    yield ('!play', zero()), ('RF', qubit)
    yield ('!play', zero()), ('Z', qubit)
    yield ('!play', zero()), ('readoutLine.RF', qubit)


def _rfUnitary(ctx, qubits, theta, phi, level1=0, level2=1):
    from numpy import interp

    qubit, = qubits

    if theta < 0:
        theta = -theta
        phi += pi
    theta = mod(theta, 2 * pi)
    if theta > pi:
        theta = 2 * pi - theta
        phi += pi

    phi = mod(
        phi + ctx.phases_ext[qubit][level1] - ctx.phases_ext[qubit][level2],
        2 * pi)
    phi = phi if abs(level2 - level1) % 2 else phi - pi
    if phi > pi:
        phi -= 2 * pi
    phi = phi / (level2 - level1)

    shape = ctx.params.get('shape', 'CosPulse')
    buffer = ctx.params.get('buffer', 0)
    alpha = ctx.params.get('alpha', 1)
    beta = ctx.params.get('beta', 0)
    delta = ctx.params.get('delta', 0)
    edge = ctx.params.get('edge', 0)
    edge_type = ctx.params.get('edge_type', 'cos')

    phase = pi * interp(phi / pi, *ctx.params['phase'])

    pulseLib = {
        'CosPulse': cosPulse,
        'Gaussian': gaussian,
        'Square': square,
        'cosPulse': cosPulse,
        'gaussian': gaussian,
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

        duration = interp(theta1 / pi, *ctx.params['duration'])
        amp = interp(theta1 / pi, *ctx.params['amp'])
        if shape == 'square':
            pulse = square(duration, type=edge_type, edge=edge)
        else:
            pulse = pulseLib[shape](duration)

        if duration > 0 and amp > 0:
            I, Q = mixing(amp * alpha * pulse,
                          phase=phase,
                          freq=delta,
                          DRAGScaling=beta / alpha)
            t = (duration + buffer) / 2 + ctx.time[qubit]
            if 'levels' in ctx.params:
                freq = (ctx.params['levels'][level2] -
                        ctx.params['levels'][level1]) / (level2 - level1)
            else:
                freq = ctx.params['frequency']
            wav, _ = mixing(I >> t, Q >> t, freq=freq)
            yield ('!play', wav), ('RF', qubit)
            yield ('!add_time', duration + buffer), qubit


@std.opaque('rfUnitary')
def rfUnitary(ctx, qubits, theta, phi):
    yield from _rfUnitary(ctx, qubits, theta, phi)


@std.opaque('rfUnitary', type='BB1')
def rfUnitary_BB1(ctx, qubits, theta, phi):
    import numpy as np

    p1 = np.arccos(-theta / (4 * pi))
    p2 = 3 * p1
    p1, p2 = p1 + phi, p2 + phi

    yield from _rfUnitary(ctx, qubits, pi, p1)
    yield from _rfUnitary(ctx, qubits, pi, p2)
    yield from _rfUnitary(ctx, qubits, pi, p2)
    yield from _rfUnitary(ctx, qubits, pi, p1)
    yield from _rfUnitary(ctx, qubits, theta, phi)


@std.opaque('Measure')
def measure(ctx, qubits, cbit=None):
    from waveforms import cos, exp, pi, step

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
    bias = ctx.params.get('bias', None)
    signal = ctx.params.get('signal', 'state')
    ring_up_amp = ctx.params.get('ring_up_amp', amp)
    ring_up_time = ctx.params.get('ring_up_time', 50e-9)
    rsing_edge_time = ctx.params.get('rsing_edge_time', 5e-9)

    try:
        w = ctx.params['w']
        weight = None
    except:
        weight = ctx.params.get('weight',
                                f'square({duration}) >> {duration/2}')
        w = None

    t = ctx.time[qubit]

    # phi = 2 * np.pi * (lo - frequency) * t

    pulse = (ring_up_amp * (step(rsing_edge_time) >> t) - (ring_up_amp - amp) *
             (step(rsing_edge_time) >>
              (t + ring_up_time)) - amp * (step(rsing_edge_time) >>
                                           (t + duration)))
    yield ('!play', pulse * cos(2 * pi * frequency)), ('readoutLine.RF', qubit)
    if bias is not None:
        yield ('!set_bias', bias), ('Z', qubit)

    # pulse = square(2 * duration) >> duration
    # ctx.channel['readoutLine.AD.trigger', qubit] |= pulse.marker

    params = {k: v for k, v in ctx.params.items()}
    params['w'] = w
    params['weight'] = weight
    if cbit >= 0:
        yield ('!capture', Capture(qubit, cbit, ctx.time[qubit], signal,
                                   params)), cbit
    yield ('!set_time', t + duration), qubit
    yield ('!set_phase', 0), qubit


def parametric(ctx, qubits):
    t = max(ctx.time[q] for q in qubits)

    duration = ctx.params['duration']
    amp = ctx.params['amp']
    offset = ctx.params['offset']
    frequency = ctx.params['frequency']

    if duration > 0:
        pulse = square(duration) >> duration / 2
        pulse = offset * pulse + amp * pulse * sin(2 * pi * frequency)
        yield ('!play', pulse >> t), ('coupler.Z', *qubits)

    for qubit in qubits:
        yield ('!set_time', t + duration), qubit

    yield ('!add_phase', ctx.params['phi1']), qubits[0]
    yield ('!add_phase', ctx.params['phi2']), qubits[1]


@std.opaque('CZ')
def CZ(ctx, qubits):
    t = max(ctx.time[q] for q in qubits)

    duration = ctx.params['duration']
    amp = ctx.params['amp']

    if amp != 0 and duration > 0:
        pulse = amp * (cos(pi / duration) * square(duration)) >> duration / 2
        yield ('!play', pulse >> t), ('coupler.Z', *qubits)

    for qubit in qubits:
        yield ('!set_time', t + duration), qubit

    yield ('!add_phase', ctx.params['phi1']), qubits[0]
    yield ('!add_phase', ctx.params['phi2']), qubits[1]


@std.opaque('CZ', type='parametric')
def CZ(ctx, qubits):
    yield from parametric(ctx, qubits)


@std.opaque('iSWAP', type='parametric')
def iSWAP(ctx, qubits):
    yield from parametric(ctx, qubits)
