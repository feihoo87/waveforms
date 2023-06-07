from pathlib import Path

from numpy import mod, pi

from waveforms.qlisp.base import MeasurementTask
from waveforms.qlisp.library import Library
from waveforms.qlisp.macro import add_VZ_rule
from waveforms.waveform import (cos, coshPulse, cosPulse, gaussian, mixing, pi,
                                sin, square, zero)
from waveforms.waveform_parser import wave_eval

EPS = 1e-9

lib = Library()


def get_frequency_phase(ctx, qubit, phi, level1, level2):
    freq = ctx.params.get('frequency', ctx.params.get('freq', 0.5))
    phi = mod(
        phi + ctx.phases_ext[qubit][level1] - ctx.phases_ext[qubit][level2],
        2 * pi)
    phi = phi if abs(level2 - level1) % 2 else phi - pi
    if phi > pi:
        phi -= 2 * pi
    phi = phi / (level2 - level1)

    return freq, phi


def R(ctx, qubits, phi=0, level1=0, level2=1):
    qubit, = qubits

    freq, phase = get_frequency_phase(ctx, qubit, phi, level1, level2)

    shape = ctx.params.get('shape', 'coshPulse')
    amp = ctx.params.get('amp', 0.5)
    width = ctx.params.get('width', 5e-9)
    plateau = ctx.params.get('plateau', 0.0)
    eps = ctx.params.get('eps', 1.0)
    buffer = ctx.params.get('buffer', 0)
    alpha = ctx.params.get('alpha', 1)
    beta = ctx.params.get('beta', 0)
    delta = ctx.params.get('delta', 0)
    channel = ctx.params.get('channel', 'RF')

    pulseLib = {
        'CosPulse': cosPulse,
        'CoshPulse': coshPulse,
        'Gaussian': gaussian,
        'cosPulse': cosPulse,
        'coshPulse': coshPulse,
        'gaussian': gaussian,
    }

    if shape in ['CoshPulse', 'coshPulse']:
        pulse = pulseLib[shape](width, plateau=plateau, eps=eps)
    else:
        pulse = pulseLib[shape](width, plateau=plateau)

    if (width > 0 or plateau > 0):
        I, Q = mixing(amp * alpha * pulse,
                      phase=phase,
                      freq=delta,
                      DRAGScaling=beta / alpha)
        t = (width + plateau + buffer) / 2 + ctx.time[qubit]
        wav, _ = mixing(I >> t, Q >> t, freq=freq)
        yield ('!add', 'waveform', wav), (channel, qubit)
        yield ('!add', 'time', width + plateau + buffer), qubit


@lib.opaque('R')
def _R(ctx, qubits, phi=0, level1=0, level2=1):
    yield from R(ctx, qubits, phi, level1, level2)


@lib.opaque('R12')
def _R12(ctx, qubits, phi=0, level1=0, level2=1):
    yield from R(ctx, qubits, phi, level1, level2)


@lib.opaque('P')
def P(ctx, qubits, phi):
    import numpy as np

    from waveforms.qlisp.assembly import call_opaque

    phi += ctx.phases[qubits[0]]
    yield ('!set', 'phase', 0), qubits[0]
    x = 2 * np.pi * np.random.random()
    y = np.pi * np.random.randint(0, 2) + x

    call_opaque((('R', x), qubits), ctx, lib)
    call_opaque((('R', x), qubits), ctx, lib)
    call_opaque((('R', phi / 2 + y), qubits), ctx, lib)
    call_opaque((('R', phi / 2 + y), qubits), ctx, lib)


def _VZ_R(st, phaseList):
    (_, phi, *with_params), qubit = st
    return [(('R', phi - phaseList[0], *with_params), qubit)], phaseList


add_VZ_rule('R', _VZ_R)


@lib.gate()
def U(q, theta, phi, lambda_):
    yield (('R', -lambda_), q)
    yield (('R', -pi - theta - lambda_), q)
    yield (('P', theta + phi + lambda_), q)


@lib.gate()
def u3(qubit, theta, phi, lambda_):
    yield (('U', theta, phi, lambda_), qubit)


@lib.gate()
def u2(qubit, phi, lambda_):
    yield (('U', pi / 2, phi, lambda_), qubit)


@lib.gate()
def u1(qubit, lambda_):
    yield (('P', lambda_), qubit)


@lib.gate()
def H(qubit):
    yield (('u2', 0, pi), qubit)


@lib.gate()
def I(q):
    yield (('u3', 0, 0, 0), q)


@lib.gate()
def X(q):
    yield (('u3', pi, 0, pi), q)


@lib.gate()
def Y(q):
    yield (('u3', pi, pi / 2, pi / 2), q)


@lib.gate()
def Z(q):
    yield (('u1', pi), q)


@lib.gate()
def S(q):
    yield (('u1', pi / 2), q)


@lib.gate(name='-S')
def Sdg(q):
    yield (('u1', -pi / 2), q)


@lib.gate()
def T(q):
    yield (('u1', pi / 4), q)


@lib.gate(name='-T')
def Tdg(q):
    yield (('u1', -pi / 4), q)


@lib.gate(name='X/2')
def sx(q):
    yield ('-S', q)
    yield ('H', q)
    yield ('-S', q)


@lib.gate(name='-X/2')
def sxdg(q):
    yield ('S', q)
    yield ('H', q)
    yield ('S', q)


@lib.gate(name='Y/2')
def sy(q):
    yield ('Z', q)
    yield ('H', q)


@lib.gate(name='-Y/2')
def sydg(q):
    yield ('H', q)
    yield ('Z', q)


@lib.gate()
def Rx(q, theta):
    yield (('u3', theta, -pi / 2, pi / 2), q)


@lib.gate()
def Ry(q, theta):
    yield (('u3', theta, 0, 0), q)


@lib.gate(name='W/2')
def W2(q):
    yield (('u3', pi / 2, -pi / 4, pi / 4), q)


@lib.gate(name='-W/2')
def W2(q):
    yield (('u3', -pi / 2, -pi / 4, pi / 4), q)


@lib.gate(name='V/2')
def W2(q):
    yield (('u3', pi / 2, -3 * pi / 4, 3 * pi / 4), q)


@lib.gate(name='-V/2')
def W2(q):
    yield (('u3', -pi / 2, -3 * pi / 4, 3 * pi / 4), q)


@lib.gate()
def Rz(q, phi):
    yield (('u1', phi), q)


@lib.gate(2)
def Cnot(qubits):
    c, t = qubits
    yield ('H', t)
    yield ('CZ', (c, t))
    yield ('H', t)


@lib.gate(2)
def crz(qubits, lambda_):
    c, t = qubits

    yield (('u1', lambda_ / 2), t)
    yield ('Cnot', (c, t))
    yield (('u1', -lambda_ / 2), t)
    yield ('Cnot', (c, t))


@lib.opaque('Delay')
def delay(ctx, qubits, time):
    qubit, = qubits
    yield ('!add', 'waveform', zero()), ('RF', qubit)
    yield ('!add', 'waveform', zero()), ('Z', qubit)
    yield ('!add', 'waveform', zero()), ('readoutLine.RF', qubit)
    yield ('!add', 'time', time), qubit


@lib.opaque('Barrier')
def barrier(ctx, qubits):
    time = max(ctx.time[qubit] for qubit in qubits)
    for qubit in qubits:
        yield ('!set', 'time', time), qubit


@lib.opaque('rfPulse')
def rfPulse(ctx, qubits, waveform):
    qubit, = qubits

    if isinstance(waveform, str):
        waveform = wave_eval(waveform)

    yield ('!add', 'waveform', waveform >> ctx.time[qubit]), ('RF', qubit)


@lib.opaque('fluxPulse')
def fluxPulse(ctx, qubits, waveform):
    qubit, = qubits

    if isinstance(waveform, str):
        waveform = wave_eval(waveform)

    yield ('!add', 'waveform', waveform >> ctx.time[qubit]), ('Z', qubit)


@lib.opaque('Pulse')
def Pulse(ctx, qubits, channel, waveform):

    if isinstance(waveform, str):
        waveform = wave_eval(waveform)

    t = max(ctx.time[qubit] for qubit in qubits)

    yield ('!add', 'waveform', waveform >> t), (channel, *qubits)


@lib.opaque('setBias')
def setBias(ctx, qubits, channel, bias, edge=0, buffer=0):
    if channel.startswith('coupler.') and len(qubits) == 2:
        qubits = sorted(qubits)
    yield ('!set', 'bias', (bias, edge, buffer)), (channel, *qubits)
    time = max(ctx.time[qubit] for qubit in qubits)
    for qubit in qubits:
        yield ('!set', 'time', time + buffer), qubit


@lib.opaque('Measure')
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
    buffer = ctx.params.get('buffer', 0)
    space = ctx.params.get('space', 0)

    try:
        w = ctx.params['w']
        weight = None
    except:
        weight = ctx.params.get('weight',
                                f'square({duration}) >> {duration/2}')
        w = None

    t = ctx.time[qubit]

    # phi = 2 * np.pi * (lo - frequency) * t

    pulse = (ring_up_amp * (step(rsing_edge_time) >>
                            (t + space / 2 + buffer / 2)) -
             (ring_up_amp - amp) *
             (step(rsing_edge_time) >>
              (t + space / 2 + buffer / 2 + ring_up_time)) - amp *
             (step(rsing_edge_time) >>
              (t + space / 2 + buffer / 2 + duration)))
    yield ('!add', 'waveform',
           pulse * cos(2 * pi * frequency)), ('readoutLine.RF', qubit)
    # if bias is not None:
    #     yield ('!set', 'bias', bias), ('Z', qubit)
    if bias is not None:
        b = ctx.biases[('Z', qubit)]
        if isinstance(b, tuple):
            b = b[0]
        pulse = (bias - b) * square(duration + space) >> (duration + space +
                                                          buffer) / 2
        yield ('!add', 'waveform', pulse >> t), ('Z', qubit)

    # pulse = square(2 * duration) >> duration
    # ctx.channel['readoutLine.AD.trigger', qubit] |= pulse.marker

    params = {k: v for k, v in ctx.params.items()}
    params['w'] = w
    params['weight'] = weight
    if cbit >= 0:
        yield ('!set', 'cbit',
               MeasurementTask(qubit, cbit, ctx.time[qubit], signal,
                               params)), cbit
    yield ('!set', 'time', t + duration), qubit
    yield ('!set', 'phase', 0), qubit


@lib.opaque('CZ')
def CZ(ctx, qubits):
    t = max(ctx.time[q] for q in qubits)

    duration = ctx.params['duration']
    amp = ctx.params['amp']
    eps = ctx.params.get('eps', 1.0)
    plateau = ctx.params.get('plateau', 0.0)

    if amp != 0 and duration > 0:
        pulse = amp * (cos(pi / duration) * square(duration)) >> duration / 2
        yield ('!add', 'waveform', pulse >> t), ('coupler.Z', *qubits)

    for qubit in qubits:
        yield ('!set', 'time', t + duration), qubit

    yield ('!add', 'phase', ctx.params['phi1']), qubits[0]
    yield ('!add', 'phase', ctx.params['phi2']), qubits[1]


@lib.opaque('Reset')
def Reset(ctx, qubits):
    qubit, *_ = qubits

    f12 = ctx.cfg.query(f'gate.R12.{qubit}.params.frequency')
    duration = ctx.params.get('duration', 1e-6)
    amp_ef = ctx.params.get('amp_ef', 1.0)
    amp_f0_g1 = ctx.params.get('amp_f0_g1', 1.0)
    freq_f0_g1 = ctx.params.get('freq_f0_g1', 1e9)

    wav1 = amp_ef * square(duration) * cos(2 * pi * f12)
    wav2 = amp_f0_g1 * square(duration) * cos(2 * pi * freq_f0_g1)

    t = ctx.time[qubit]

    yield ('!add', 'waveform', wav1 >> t), ('RF', qubit)
    yield ('!add', 'waveform', wav2 >> t), ('Z', qubit)
    yield ('!add', 'time', duration), qubit
