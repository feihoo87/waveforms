from typing import Optional

from numpy import pi
from waveforms.waveform import cos, sin

from .config import Config, getConfig
from .library import Library
from .macro import extend_macro, reduceVirtualZ
from .qasm import qasm_eval
from .qlisp import Context, MeasurementTask, QLispCode, QLispError, gateName
from .stdlib import std


def call_opaque(st: tuple, ctx: Context, lib: Library):
    name = gateName(st)
    gate, qubits = st
    try:
        g = ctx.cfg.getGate(name, *qubits)
        type = g.type
    except:
        type = 'default'

    func = lib.getOpaque(name, type)
    if func is None:
        raise KeyError('Undefined {type} type of {name} opaque.')
    qubits = st[1]
    if isinstance(st[0], str):
        args = ()
    else:
        args = st[0][1:]
    sub_ctx = Context(cfg=ctx.cfg)
    sub_ctx.time.update(ctx.time)
    sub_ctx.phases.update(ctx.phases)

    func(sub_ctx, qubits, *args)

    ctx.time.update(sub_ctx.time)
    ctx.phases.update(sub_ctx.phases)

    for k, wav in sub_ctx.raw_waveforms.items():
        _addWaveforms(ctx, k, wav)
    for k, taskList in sub_ctx.measures.items():
        for task in taskList:
            _addMeasurementInfo(ctx, task)
            ctx.measures[k].append(task)


def _getSharedCoupler(qubits):
    s = set(qubits[0]['couplers'])
    for qubit in qubits[1:]:
        s = s & set(qubit['couplers'])
    return s


def _getChannel(ctx, key):
    name, *qubits = key

    qubits = [ctx.cfg.getQubit(q) for q in qubits]

    if name.startswith('readoutLine.'):
        #name = name.removeprefix('readoutLine.')
        name = name[len('readoutLine.'):]
        rl = ctx.cfg.getReadoutLine(qubits[0]['readoutLine'])
        chInfo = rl.query('channels.' + name)
    elif name.startswith('coupler.'):
        #name = name.removeprefix('coupler.')
        name = name[len('coupler.'):]
        c = _getSharedCoupler(qubits).pop()
        c = ctx.cfg.getCoupler(c)
        chInfo = c.query('channels.' + name)
    else:
        chInfo = qubits[0].query('channels.' + name)
    return chInfo


def _addWaveforms(ctx, key, wav):
    chInfo = _getChannel(ctx, key)
    if isinstance(chInfo, str):
        ctx.waveforms[chInfo] += wav
    else:
        _addMultChannelWaveforms(ctx, wav, chInfo)


def _addMultChannelWaveforms(ctx, wav, chInfo):
    lo = ctx.cfg.getChannel(chInfo['LO'])
    lofreq = lo.status.frequency
    if 'I' in chInfo:
        I = (2 * wav * cos(-2 * pi * lofreq)).filter(high=2 * pi * lofreq)
        ctx.waveforms[chInfo['I']] += I
    if 'Q' in chInfo:
        Q = (2 * wav * sin(-2 * pi * lofreq)).filter(high=2 * pi * lofreq)
        ctx.waveforms[chInfo['Q']] += Q


def _addMeasurementInfo(ctx: Context, task: MeasurementTask):
    rl = ctx.cfg.getQubit(task.qubit).readoutLine
    rl = ctx.cfg.getReadoutLine(rl)
    if isinstance(rl.channels.AD, dict):
        if 'LO' in rl.channels.AD:
            lo = ctx.cfg.getChannel(rl.channels.AD.LO)
            loFreq = lo.status.frequency
            task.hardware['channel']['LO'] = rl.channels.AD.LO
            task.hardware['params']['LOFrequency'] = loFreq

        task.hardware['params']['sampleRate'] = {}
        for ch in ['I', 'Q', 'IQ', 'Ref']:
            if ch in rl.channels.AD:
                task.hardware['channel'][ch] = rl.channels.AD[ch]
                sampleRate = ctx.cfg.getChannel(
                    rl.channels.AD[ch]).params.sampleRate
                task.hardware['params']['sampleRate'][ch] = sampleRate
    elif isinstance(rl.channels.AD, str):
        task.hardware['channel'] = rl.channels.AD


def assembly(qlisp,
             cfg: Optional[Config] = None,
             lib: Library = std,
             ctx: Optional[Context] = None):

    if cfg is None:
        cfg = getConfig()

    if ctx is None:
        ctx = Context(cfg=cfg)

    allQubits = sorted(cfg['chip']['qubits'].keys(), key=lambda s: int(s[1:]))

    for gate, qubits in qlisp:
        ctx.qlisp.append((gate, qubits))
        if isinstance(qubits, int):
            qubits = (allQubits[qubits], )
        elif isinstance(qubits, str):
            qubits = (qubits, )
        else:
            qubits = tuple(
                [allQubits[q] if isinstance(q, int) else q for q in qubits])

        try:
            call_opaque((gate, qubits), ctx, lib=lib)
        except:
            raise QLispError(f'assembly statement {(gate, qubits)} error.')

    end = max(ctx.time.values())
    for q in ctx.time:
        ctx.time[q] = end
    ctx.end = end
    return ctx


def compile(prog, cfg: Optional[Config] = None, lib: Library = std, **options):
    """
    options: 
        qasm_only = True: only compile qasm to qlisp
        no_virtual_z = True: keep P gates as original form.
        no_assembly = True: return simplified qlisp.
    """
    if cfg is None:
        cfg = getConfig()

    if isinstance(prog, str):
        prog = qasm_eval(prog, lib)
    if 'qasm_only' in options:
        return list(prog)
    prog = extend_macro(prog, lib)
    if 'no_virtual_z' in options:
        return list(prog)
    prog = reduceVirtualZ(prog, lib)
    if 'no_assembly' in options:
        return list(prog)
    ctx = assembly(prog, cfg, lib)

    code = QLispCode(cfg=ctx.cfg,
                     qlisp=ctx.qlisp,
                     waveforms=dict(ctx.waveforms),
                     measures=dict(ctx.measures),
                     end=ctx.end)
    return code
