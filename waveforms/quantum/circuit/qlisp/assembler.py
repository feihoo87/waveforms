from typing import Optional

from numpy import pi
from waveforms.waveform import cos, sin

from .config import Config, getConfig
from .library import Library
from .qlisp import Context, MeasurementTask, QLispError, gateName
from .stdlib import std

## query


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


def _getLOFrequency(ctx, chInfo):
    lo = ctx.cfg.getChannel(chInfo['LO'])
    lofreq = lo.status.frequency
    return lofreq


def _getADChannel(ctx, qubit):
    rl = ctx.cfg.getQubit(qubit).readoutLine
    rl = ctx.cfg.getReadoutLine(rl)
    return rl.channels.AD


def _getSampleRate(ctx, channel):
    return ctx.cfg.getChannel(channel).params.sampleRate


def _getGateConfig(ctx, name, *qubits):
    try:
        gate = ctx.cfg.getGate(name, *qubits)
    except:
        return {'type': 'default', 'params': {}}
    params = gate['params']
    type = gate.get('type', 'default')
    return {'type': type, 'params': params}


##


def call_opaque(st: tuple, ctx: Context, lib: Library):
    name = gateName(st)
    gate, qubits = st
    gatecfg = _getGateConfig(ctx, name, *qubits)

    func = lib.getOpaque(name, gatecfg['type'])
    if func is None:
        raise KeyError('Undefined {type} type of {name} opaque.')

    if isinstance(gate, str):
        args = ()
    else:
        args = gate[1:]

    sub_ctx = Context(cfg=ctx.cfg, scopes=[*ctx.scopes, gatecfg['params']])
    sub_ctx.time.update(ctx.time)
    sub_ctx.phases.update(ctx.phases)

    func(sub_ctx, qubits, *args)

    ctx.time.update(sub_ctx.time)
    ctx.phases.update(sub_ctx.phases)

    for channel, wav in sub_ctx.raw_waveforms.items():
        _addWaveforms(ctx, channel, wav)
    for cbit, taskList in sub_ctx.measures.items():
        for task in taskList:
            _addMeasurementInfo(ctx, task)
            ctx.measures[cbit].append(task)


def _getSharedCoupler(qubits):
    s = set(qubits[0]['couplers'])
    for qubit in qubits[1:]:
        s = s & set(qubit['couplers'])
    return s


def _addWaveforms(ctx, key, wav):
    chInfo = _getChannel(ctx, key)
    if isinstance(chInfo, str):
        ctx.waveforms[chInfo] += wav
    else:
        _addMultChannelWaveforms(ctx, wav, chInfo)


def _addMultChannelWaveforms(ctx, wav, chInfo):
    lofreq = _getLOFrequency(ctx, chInfo)
    if 'I' in chInfo:
        I = (2 * wav * cos(-2 * pi * lofreq)).filter(high=2 * pi * lofreq)
        ctx.waveforms[chInfo['I']] += I
    if 'Q' in chInfo:
        Q = (2 * wav * sin(-2 * pi * lofreq)).filter(high=2 * pi * lofreq)
        ctx.waveforms[chInfo['Q']] += Q


def _addMeasurementInfo(ctx: Context, task: MeasurementTask):
    AD = _getADChannel(ctx, task.qubit)
    if isinstance(AD, dict):
        if 'LO' in AD:
            loFreq = _getLOFrequency(ctx, AD)
            task.hardware['channel']['LO'] = AD['LO']
            task.hardware['params']['LOFrequency'] = loFreq

        task.hardware['params']['sampleRate'] = {}
        for ch in ['I', 'Q', 'IQ', 'Ref']:
            if ch in AD:
                task.hardware['channel'][ch] = AD[ch]
                sampleRate = _getSampleRate(ctx, AD[ch])
                task.hardware['params']['sampleRate'][ch] = sampleRate
    elif isinstance(AD, str):
        task.hardware['channel'] = AD


def _allocQubits(cfg, qlisp):
    allQubits = sorted(cfg['chip']['qubits'].keys(), key=lambda s: int(s[1:]))
    addressTable = {q: q for q in allQubits}
    for i, q in enumerate(allQubits):
        addressTable[i] = q
    return addressTable


def assembly(qlisp,
             cfg: Optional[Config] = None,
             lib: Library = std,
             ctx: Optional[Context] = None):

    if cfg is None:
        cfg = getConfig()

    if ctx is None:
        ctx = Context(cfg=cfg)

    addressTable = _allocQubits(cfg, qlisp)

    for gate, qubits in qlisp:
        ctx.qlisp.append((gate, qubits))
        if isinstance(qubits, (int, str)):
            qubits = (addressTable[qubits], )
        else:
            qubits = tuple(
                [addressTable[q] if isinstance(q, int) else q for q in qubits])
        try:
            call_opaque((gate, qubits), ctx, lib=lib)
        except:
            raise QLispError(f'assembly statement {(gate, qubits)} error.')

    end = max(ctx.time.values())
    for q in ctx.time:
        ctx.time[q] = end
    ctx.end = end
    return ctx
