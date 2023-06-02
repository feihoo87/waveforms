import numpy as np
from waveforms.quantum.circuit.qlisp.stdlib import MeasurementTask, _rfUnitary
from waveforms.waveform import D, mixing, square, step, zero

from qlisp import libraries, stdlib

lib = libraries(stdlib)


@lib.opaque('rfUnitary12')
def rfUnitary12(ctx, qubits, theta, phi):
    yield from _rfUnitary(ctx, qubits, theta, phi)


@lib.opaque('rfUnitary02')
def rfUnitary02(ctx, qubits, theta, phi):
    yield from _rfUnitary(ctx, qubits, theta, phi)


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
    amp = np.abs(ctx.params['amp'])
    duration = ctx.params['duration']
    frequency = ctx.params['frequency']
    bias = ctx.params.get('bias', None)
    signal = ctx.params.get('signal', 'state')
    ring_up_amp = ctx.params.get('ring_up_amp', amp)
    ring_up_time = ctx.params.get('ring_up_time', 50e-9)
    rsing_edge_time = ctx.params.get('rsing_edge_time', 20e-9)
    buffer = ctx.params.get('buffer', 0)
    space = ctx.params.get('space', 0)

    try:
        w = ctx.params['w']
        weight = None
    except:
        weight = ctx.params.get('weight',
                                f'square({duration}) >> {duration/2}')
        w = None

    t = ctx.time[qubit] + buffer

    # phi = 2 * np.pi * (lo - frequency) * t

    pulse = (ring_up_amp * (step(rsing_edge_time) >> t) - (ring_up_amp - amp) *
             (step(rsing_edge_time) >>
              (t + ring_up_time)) - amp * (step(rsing_edge_time) >>
                                           (t + duration)))
    pulse >>= space
    yield ('!add', 'waveform',
           pulse * cos(2 * pi * frequency)), ('readoutLine.RF', qubit)
    if bias is not None:
        b = ctx.biases[('Z', qubit)]
        if isinstance(b, tuple):
            b = b[0]
        bias_duration = 7e-6
        pulse = (bias - b) * square(bias_duration + space + buffer) >> (
            bias_duration + space + buffer) / 2
        yield ('!add', 'waveform', pulse >> t - buffer), ('Z', qubit)

    # pulse = square(2 * duration) >> duration
    # ctx.channel['readoutLine.AD.trigger', qubit] |= pulse.marker

    params = {k: v for k, v in ctx.params.items()}
    params['w'] = w
    params['weight'] = weight
    if cbit >= 0:
        yield ('!set', 'cbit',
               MeasurementTask(qubit, cbit, ctx.time[qubit], signal,
                               params)), cbit
    yield ('!set', 'time', t + duration + space), qubit
    yield ('!set', 'phase', 0), qubit


@lib.gate(name='-X')
def X(q):
    yield (('u3', np.pi, 0, -np.pi), q)


@lib.gate(name='-Y')
def Y(q):
    yield (('u3', np.pi, -np.pi / 2, -np.pi / 2), q)


def _CR_pulse(ctx, qubits, positive=1):

    t = max(ctx.time[q] for q in qubits)

    duration = ctx.params['duration']
    amp1 = ctx.params['amp1']
    amp2 = ctx.params['amp2']
    global_phase = ctx.params['global_phase']
    relative_phase = ctx.params['relative_phase']

    edge_type = ctx.params.get('edge_type', 'cos')
    edge = ctx.params.get('edge', duration / 10)
    drag = ctx.params.get('drag', 0) * 1e-9
    skew = ctx.params.get('skew', 0) * 1e-9
    buffer = ctx.params.get('buffer', 10e-9)

    pulse = square(width=duration, edge=edge,
                   type=edge_type) >> t + duration / 2 + edge / 2 + buffer / 2
    d_step_pulse = D(step(edge=edge, type=edge_type))
    drag_pulse = (d_step_pulse >> t +
                  (edge + buffer) / 2) - (d_step_pulse >> t +
                                          (edge + buffer) / 2 + duration)
    skew_pulse = (d_step_pulse >> t +
                  (edge + buffer) / 2) + (d_step_pulse >> t +
                                          (edge + buffer) / 2 + duration)

    if amp1 > 0 and duration > 0:
        wav, _ = mixing(amp1 * pulse,
                        phase=global_phase - ctx.phases[qubits[1]],
                        freq=ctx.params['frequency'])
        # ctx.channel['RF', qubits[0]] += wav*positive
        yield ('!add', 'waveform', wav * positive), ('RF', qubits[0])
        # ctx.time[qubits[0]] += duration + edge + buffer
        yield ('!set', 'time', t + duration + edge + buffer), qubits[0]

    if amp2 > 0 and duration > 0:
        I2, Q2 = amp2 * pulse, zero()
        if drag > 0:
            Q2 += drag * drag_pulse
        if skew > 0:
            Q2 += skew * skew_pulse
        wav, _ = mixing(I2,
                        Q2,
                        phase=global_phase + relative_phase -
                        ctx.phases[qubits[1]],
                        freq=ctx.params['frequency'])
        # ctx.channel['RF', qubits[1]] += wav*positive
        yield ('!add', 'waveform', wav * positive), ('RF', qubits[1])
        # ctx.time[qubits[1]] += duration + edge + buffer
        yield ('!set', 'time', t + duration + edge + buffer), qubits[1]

    # ctx.phases[qubits[0]] += ctx.params['phi1']
    yield ('!add', 'phase', ctx.params['phi1']), qubits[0]
    # ctx.phases[qubits[1]] += ctx.params['phi2']
    yield ('!add', 'phase', ctx.params['phi2']), qubits[1]


@lib.opaque(name='CR')
def _CR(ctx, qubits):
    yield from _CR_pulse(ctx, qubits, positive=1)


@lib.opaque(name='CR', type='echo')
def _CR(ctx, qubits):
    yield from _CR_pulse(ctx, qubits, 1)
    yield ('X', qubits[0])
    yield from _CR_pulse(ctx, qubits, -1)
    yield ('X', qubits[0])


@lib.opaque('iSWAP')
def iSWAP(ctx, qubits):
    t = max(ctx.time[q] for q in qubits)

    duration = ctx.params['duration']
    amp1 = ctx.params.get('amp1', 0)
    amp2 = ctx.params.get('amp2', 0)
    ampc = ctx.params.get('ampc', 0)
    edge = ctx.params.get('edge', 0)
    buffer = ctx.params.get('buffer', 0)

    t += buffer

    pulse = amp1 * (square(duration, edge=edge)) >> duration / 2
    yield ('!add', 'waveform', pulse >> t), ('Z', qubits[0])
    pulse = amp2 * (square(duration, edge=edge)) >> duration / 2
    yield ('!add', 'waveform', pulse >> t), ('Z', qubits[1])

    pulse = ampc * (square(duration, edge=edge)) >> duration / 2
    yield ('!add', 'waveform', pulse >> t), ('coupler.Z', *qubits)

    for qubit in qubits:
        yield ('!set', 'time', t + duration + buffer), qubit

    yield ('!add', 'phase', ctx.params.get('phi1', 0)), qubits[0]
    yield ('!add', 'phase', ctx.params.get('phi2', 0)), qubits[1]


@lib.opaque('CZ')
def CZ(ctx, qubits):
    t = max(ctx.time[q] for q in qubits)

    duration = ctx.params['duration']
    amp1 = ctx.params.get('amp1', 0)
    amp2 = ctx.params.get('amp2', 0)
    ampc = ctx.params.get('ampc', 0)
    edge = ctx.params.get('edge', 0)
    buffer = ctx.params.get('buffer', 0)

    t += buffer

    pulse = amp1 * (square(duration, edge=edge)) >> duration / 2
    yield ('!add', 'waveform', pulse >> t), ('Z', qubits[0])
    pulse = amp2 * (square(duration, edge=edge)) >> duration / 2
    yield ('!add', 'waveform', pulse >> t), ('Z', qubits[1])

    pulse = ampc * (square(duration, edge=edge)) >> duration / 2
    yield ('!add', 'waveform', pulse >> t), ('coupler.Z', *qubits)

    for qubit in qubits:
        yield ('!set', 'time', t + duration + buffer), qubit

    yield ('!add', 'phase', ctx.params.get('phi1', 0)), qubits[0]
    yield ('!add', 'phase', ctx.params.get('phi2', 0)), qubits[1]
