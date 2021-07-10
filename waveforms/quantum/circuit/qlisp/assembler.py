from typing import Optional, Union

from numpy import pi
from waveforms.waveform import cos, sin, step

from .config import Config, getConfig
from .library import Library
from .qlisp import Context, MeasurementTask, QLispCode, QLispError, gateName
from .stdlib import std

## query


def _getAWGChannel(ctx, name, *qubits) -> Union[str, dict]:

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


def _getADChannel(ctx, qubit) -> Union[str, dict]:
    rl = ctx.cfg.getQubit(qubit).readoutLine
    rl = ctx.cfg.getReadoutLine(rl)
    return rl.channels.AD


def _getLOFrequency(ctx, chInfo) -> float:
    lo = ctx.cfg.getChannel(chInfo['LO'])
    lofreq = lo.status.frequency
    return lofreq


def _getADChannelDetails(ctx, chInfo) -> dict:
    def _getADSampleRate(ctx, channel):
        return ctx.cfg.getChannel(channel).params.sampleRate

    hardware = {'channel': {}, 'params': {}}
    if isinstance(chInfo, dict):
        if 'LO' in chInfo:
            loFreq = _getLOFrequency(ctx, chInfo)
            hardware['channel']['LO'] = chInfo['LO']
            hardware['params']['LOFrequency'] = loFreq

        hardware['params']['sampleRate'] = {}
        for ch in ['I', 'Q', 'IQ', 'Ref']:
            if ch in chInfo:
                hardware['channel'][ch] = chInfo[ch]
                sampleRate = _getADSampleRate(ctx, chInfo[ch])
                hardware['params']['sampleRate'][ch] = sampleRate
    elif isinstance(chInfo, str):
        hardware['channel'] = chInfo

    return hardware


def _getGateConfig(ctx, name, *qubits):
    try:
        gate = ctx.cfg.getGate(name, *qubits)
    except:
        return {'type': 'default', 'params': {}}
    params = gate['params']
    type = gate.get('type', 'default')
    return {'type': type, 'params': params}


def _getQubitList(ctx):
    return sorted(ctx.cfg['chip']['qubits'].keys(), key=lambda s: int(s[1:]))


##


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
    sub_ctx.biases.update(ctx.biases)

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


def _getSharedCoupler(qubits):
    s = set(qubits[0]['couplers'])
    for qubit in qubits[1:]:
        s = s & set(qubit['couplers'])
    return s


def _addWaveforms(ctx, channel, wav):
    name, *qubits = channel
    chInfo = _getAWGChannel(ctx, name, *qubits)
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


def _addMeasurementHardwareInfo(ctx: Context, task: MeasurementTask):
    AD = _getADChannel(ctx, task.qubit)
    task.hardware.update(_getADChannelDetails(ctx, AD))


def _allocQubits(ctx, qlisp):
    for i, q in enumerate(_getQubitList(ctx)):
        ctx.addressTable[q] = q
        ctx.addressTable[i] = q


def assembly(qlisp,
             cfg: Optional[Config] = None,
             lib: Library = std,
             ctx: Optional[Context] = None):

    if cfg is None:
        cfg = getConfig()

    if ctx is None:
        ctx = Context(cfg=cfg)

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
