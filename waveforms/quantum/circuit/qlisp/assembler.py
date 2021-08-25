from typing import Optional

from numpy import pi
from waveforms.waveform import cos, sin, step

from .config import Config, getConfig
from .library import Library
from .qlisp import (Context, MeasurementTask, QLispCode, QLispError,
                    create_context, gateName)
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
    gatecfg = ctx._getGateConfig(name, *qubits)

    func = lib.getOpaque(name, gatecfg['type'])
    if func is None:
        raise KeyError('Undefined {type} type of {name} opaque.')

    if isinstance(gate, str):
        args = ()
    else:
        args = gate[1:]

    sub_ctx = create_context(ctx, scopes=[*ctx.scopes, gatecfg['params']])

    func(sub_ctx, qubits, *args)

    for channel, bias in sub_ctx.biases.items():
        if ctx.biases[channel] != bias:
            _, *qubits = channel
            t = max(ctx.time[q] for q in qubits)
            wav = (bias - ctx.biases[channel]) * step(0) >> t
            _addWaveforms(ctx, channel, wav)
            ctx.biases[channel] = bias

    ctx.time.update(sub_ctx.time)
    ctx.phases.update(sub_ctx.phases)

    for channel, wav in sub_ctx.raw_waveforms.items():
        _addWaveforms(ctx, channel, wav)
    for cbit, taskList in sub_ctx.measures.items():
        for task in taskList:
            _addMeasurementHardwareInfo(ctx, task)
            ctx.measures[cbit].append(task)


def _addWaveforms(ctx, channel, wav):
    name, *qubits = channel
    chInfo = ctx._getAWGChannel(name, *qubits)
    if isinstance(chInfo, str):
        ctx.waveforms[chInfo] += wav
    else:
        _addMultChannelWaveforms(ctx, wav, chInfo)


def _addMultChannelWaveforms(ctx, wav, chInfo):
    lofreq = ctx._getLOFrequencyOfChannel(chInfo)
    if 'I' in chInfo:
        try:
            I = (2 * wav * cos(-2 * pi * lofreq)).filter(high=2 * pi * lofreq)
        except:
            w = (2 * wav * cos(-2 * pi * lofreq))
            print("====== ERROR WAVEFORM ======")
            print("    lofreq =", lofreq)
            print("")
            print(w.bound)
            print("")
            print(w.seq)
            print("====== ERROR WAVEFORM ======")
            raise
        ctx.waveforms[chInfo['I']] += I
    if 'Q' in chInfo:
        Q = (2 * wav * sin(-2 * pi * lofreq)).filter(high=2 * pi * lofreq)
        ctx.waveforms[chInfo['Q']] += Q


def _addMeasurementHardwareInfo(ctx: Context, task: MeasurementTask):
    AD = ctx._getADChannel(task.qubit)
    task.hardware.update(ctx._getADChannelDetails(AD))


def _allocQubits(ctx, qlisp):
    for i, q in enumerate(ctx._getAllQubitLabels()):
        ctx.addressTable[q] = q
        ctx.addressTable[i] = q


def assembly(qlisp,
             cfg: Optional[Config] = None,
             lib: Library = std,
             ctx: Optional[Context] = None):

    if cfg is None:
        cfg = getConfig()

    if ctx is None:
        ctx = create_context(cfg=cfg)

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
