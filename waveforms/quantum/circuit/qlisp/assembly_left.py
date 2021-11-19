from numpy import pi
from waveforms.waveform import Waveform, cos, sin, step

from .library import Library
from .qlisp import (ADChannel, AWGChannel, Context, GateConfig,
                    MeasurementTask, MultADChannel, MultAWGChannel, QLispCode,
                    QLispError, create_context, gateName)
from .stdlib import std


@std.opaque('__finally__')
def _finally_opaque(ctx, qubits):
    """_finally_opaque

    clean all biases.
    """
    for ch in ctx.biases:
        ctx.biases[ch] = 0


def call_opaque(st: tuple, ctx: Context, lib: Library):
    name = gateName(st)
    gate, qubits = st
    gatecfg = ctx.cfg._getGateConfig(name, *qubits)
    if gatecfg is None:
        gatecfg = GateConfig(name, qubits)

    func, params_declaration = lib.getOpaque(name, gatecfg.type)
    if func is None:
        raise KeyError('Undefined {gatecfg.type} type of {name} opaque.')
    for p in params_declaration:
        if p.name not in gatecfg.params:
            pass
            # raise ValueError(
            #     f'{name} (type={gatecfg.type}) opaque of {qubits} missing parameter {k}.'
            # )

    if isinstance(gate, str):
        args = ()
    else:
        args = gate[1:]

    sub_ctx = create_context(ctx, scopes=[*ctx.scopes, gatecfg.params])

    func(sub_ctx, gatecfg.qubits, *args)

    for channel, bias in sub_ctx.biases.items():
        if isinstance(bias, tuple):
            bias, edge, buffer = bias
        else:
            edge, buffer = 0, 0
        if ctx.biases[channel] != bias:
            _, *qubits = channel
            t = max(ctx.time[q] for q in qubits)
            wav = (bias - ctx.biases[channel]) * step(edge) >> (t + buffer / 2)
            _addWaveforms(ctx, channel, wav)
            ctx.biases[channel] = bias

    ctx.time.update(sub_ctx.time)
    ctx.phases.update(sub_ctx.phases)

    for channel, wav in sub_ctx.raw_waveforms.items():
        _addWaveforms(ctx, channel, wav)
    for cbit, taskList in sub_ctx.measures.items():
        for task in taskList:
            hardware = ctx.cfg._getADChannel(task.qubit)
            ctx.measures[cbit].append(
                MeasurementTask(task.qubit, task.cbit, task.time, task.signal,
                                task.params, hardware))


def _addWaveforms(ctx: Context, channel: tuple, wav: Waveform):
    name, *qubits = channel
    ch = ctx.cfg._getAWGChannel(name, *qubits)
    if isinstance(ch, AWGChannel):
        ctx.waveforms[ch.name] += wav
    else:
        _addMultChannelWaveforms(ctx, wav, ch)


def _addMultChannelWaveforms(ctx: Context, wav, ch: MultAWGChannel):
    lofreq = ch.lo_freq
    if ch.I is not None:
        try:
            I = (2 * wav * cos(-2 * pi * lofreq)).filter(high=2 * pi * lofreq)
        except:
            w = (2 * wav * cos(-2 * pi * lofreq))
            print("====== ERROR WAVEFORM ======")
            print("    lofreq =", lofreq)
            print("")
            print(w.bounds)
            print("")
            print(w.seq)
            print("====== ERROR WAVEFORM ======")
            raise
        ctx.waveforms[ch.I.name] += I
    if ch.Q is not None:
        Q = (2 * wav * sin(-2 * pi * lofreq)).filter(high=2 * pi * lofreq)
        ctx.waveforms[ch.Q.name] += Q


def _allocQubits(ctx, qlisp):
    for i, q in enumerate(ctx.cfg._getAllQubitLabels()):
        ctx.addressTable[q] = q
        ctx.addressTable[i] = q


def _assembly_align_left(qlisp, ctx: Context, lib: Library):
    _allocQubits(ctx, qlisp)

    allQubits = set()

    for gate, qubits in qlisp:
        ctx.qlisp.append((gate, qubits))
        if isinstance(qubits, (int, str)):
            qubits = (ctx.qubit(qubits), )
        else:
            qubits = tuple(
                [ctx.qubit(q) if isinstance(q, int) else q for q in qubits])
        try:
            call_opaque((gate, qubits), ctx, lib=lib)
            allQubits.update(set(qubits))
        except:
            raise QLispError(f'assembly statement {(gate, qubits)} error.')
    call_opaque(('Barrier', tuple(allQubits)), ctx, lib=lib)
    call_opaque(('__finally__', tuple(allQubits)), ctx, lib=lib)
    ctx.end = max(ctx.time.values())

    code = QLispCode(cfg=ctx.cfg,
                     qlisp=ctx.qlisp,
                     waveforms=dict(ctx.waveforms),
                     measures=dict(ctx.measures),
                     end=ctx.end)
    return code
