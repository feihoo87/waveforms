from typing import Optional

from numpy import pi
from waveforms.waveform import cos, sin

from .config import Config, getConfig
from .library import Context, Library
from .macro import extend_macro, reduceVirtualZ
from .qasm import qasm_eval
from .qlisp import gateName
from .stdlib import std


def call_opaque(st: tuple, ctx: Context, scope: dict):
    name = gateName(st)
    gate, qubits = st
    try:
        g = ctx.cfg.getGate(name, *qubits)
        type = g.type
    except:
        type = 'default'
    if name not in scope:
        raise KeyError('Undefined opaque {name}')
    elif type not in scope[name]:
        raise KeyError('Undefined {type} type of {name} opaque.')
    func = scope[name][type]
    qubits = st[1]
    if isinstance(st[0], str):
        args = ()
    else:
        args = st[0][1:]
    func(ctx, qubits, *args)


def _getSharedCoupler(qubits):
    s = set(qubits[0]['couplers'])
    for qubit in qubits[1:]:
        s = s & set(qubit['couplers'])
    return s


def _getChannel(ctx, key):
    name, *qubits = key

    qubits = [ctx.cfg.getQubit(q) for q in qubits]

    if name.startswith('readoutLine.'):
        name = name.removeprefix('readoutLine.')
        rl = ctx.cfg.getReadoutLine(qubits[0]['readoutLine'])
        chInfo = rl.query('channels.' + name)
    elif name.startswith('coupler.'):
        name = name.removeprefix('coupler.')
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
        I = (2 * wav * cos(2 * pi * lofreq)).filter(high=2 * pi * lofreq)
        ctx.waveforms[chInfo['I']] += I
    if 'Q' in chInfo:
        Q = (2 * wav * sin(2 * pi * lofreq)).filter(high=2 * pi * lofreq)
        ctx.waveforms[chInfo['Q']] += Q


def assembly(qlisp,
             cfg: Optional[Config] = None,
             lib: Library = std,
             ctx: Optional[Context] = None):

    if cfg is None:
        cfg = getConfig()

    if ctx is None:
        ctx = Context(cfg)

    allQubits = sorted(cfg['chip']['qubits'].keys(), key=lambda s: int(s[1:]))

    for gate, qubits in qlisp:
        if isinstance(qubits, int):
            qubits = (allQubits[qubits], )
        else:
            qubits = tuple([allQubits[q] for q in qubits])

        call_opaque((gate, qubits), ctx, scope=lib.opaques)

    for k, wav in ctx.raw_waveforms.items():
        _addWaveforms(ctx, k, wav)
    return ctx


def compile(prog, cfg: Optional[Config] = None, lib: Library = std):
    if cfg is None:
        cfg = getConfig()

    if isinstance(prog, str):
        prog = qasm_eval(prog, lib.qasmLib)
    prog = extend_macro(prog, lib.gates)
    prog = reduceVirtualZ(prog, lib.gates)
    ctx = assembly(prog, cfg, lib)
    waveforms = dict(ctx.waveforms)
    measures = dict(ctx.measures)
    return waveforms, measures, ctx
