import inspect

from numpy import pi

from ..dicttree import NOTSET
from ..waveform import Waveform, cos, sin, step
from .base import (ADChannel, AWGChannel, Context, GateConfig, MeasurementTask,
                   MultADChannel, MultAWGChannel, QLispCode, QLispError,
                   create_context, gateName)
from .library import Library


def _ctx_update_biases(sub_ctx: Context, ctx: Context):
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


def _ctx_update_time(sub_ctx: Context, ctx: Context):
    ctx.time.update(sub_ctx.time)


def _ctx_update_phases(sub_ctx: Context, ctx: Context):
    #ctx.phases.update(sub_ctx.phases)
    for k, v in sub_ctx.phases_ext.items():
        ctx.phases_ext[k].update(v)


def _ctx_update_waveforms(sub_ctx: Context, ctx: Context):
    for channel, wav in sub_ctx.raw_waveforms.items():
        _addWaveforms(ctx, channel, wav)


def _ctx_update_measurement_tasks(sub_ctx: Context, ctx: Context):
    for cbit, task in sub_ctx.measures.items():
        hardware = ctx.cfg._getADChannel(task.qubit)
        ctx.measures[cbit] = MeasurementTask(task.qubit, task.cbit, task.time,
                                             task.signal, task.params,
                                             hardware)


def _execute(ctx, cmd):
    (op, target, *values), key = cmd
    if op == '!nop':
        return
    if (op, target) == ('!set', 'time'):
        ctx.time[key] = values[0]
    elif (op, target) == ('!set', 'phase'):
        ctx.phases[key] = values[0]
    elif (op, target) == ('!set', 'phase_ext'):
        ctx.phases_ext[key][values[0]] = values[1]
    elif (op, target) == ('!set', 'bias'):
        ctx.biases[key] = values[0]
    elif (op, target) == ('!set', 'waveform'):
        ctx.raw_waveforms[key] = values[0]
    elif (op, target) == ('!set', 'cbit'):
        ctx.measures[key] = values[0]
    elif (op, target) == ('!add', 'time'):
        ctx.time[key] += values[0]
    elif (op, target) == ('!add', 'phase'):
        ctx.phases[key] += values[0]
    elif (op, target) == ('!add', 'phase_ext'):
        ctx.phases_ext[key][values[0]] += values[1]
    elif (op, target) == ('!add', 'bias'):
        ctx.biases[key] += values[0]
    elif (op, target) == ('!add', 'waveform'):
        ctx.raw_waveforms[key] += values[0]
    else:
        raise QLispError(f'Unknown command: {cmd}')


def _call_func_with_kwds(func, kwds):
    sig = inspect.signature(func)
    for p in sig.parameters.values():
        if p.kind == p.VAR_KEYWORD:
            return func(**kwds)
    kw = {k: v for k, v in kwds.items() if k in sig.parameters}
    return func(**kw)


def _try_to_call(x, kwds):
    if callable(x):
        return _call_func_with_kwds(x, kwds)
    return x


def _try_to_lookup_config(cfg, key):
    if isinstance(key, str) and key.startswith('cfg:'):
        value = cfg.query(key[4:])
        if isinstance(value, tuple) and len(value) == 2 and value[0] is NOTSET:
            raise QLispError(f'Unknown config key: {key[4:]}')
        return value
    else:
        return key


def call_opaque(st: tuple, ctx: Context, lib: Library):
    name = gateName(st)
    gate, qubits = st
    type = None
    tmp_params = {}
    args = []

    if isinstance(gate, tuple):
        for arg in gate[1:]:
            if isinstance(arg, tuple) and isinstance(arg[0],
                                                     str) and arg[0] == 'with':
                for p in arg[1:]:
                    if isinstance(p, tuple) and len(p) == 2 and isinstance(
                            p[0], str):
                        if p[0] == 'type':
                            type = _try_to_lookup_config(ctx.cfg, p[1])
                        elif p[0].startswith('param:'):
                            tmp_params[p[0][6:]] = _try_to_lookup_config(
                                ctx.cfg, p[1])
            else:
                args.append(_try_to_lookup_config(ctx.cfg, arg))
    gatecfg = ctx.cfg._getGateConfig(name, *qubits, type=type)
    if gatecfg is None:
        gatecfg = GateConfig(name, qubits)

    tmp_params = {
        k: _try_to_call(v, gatecfg.params)
        for k, v in tmp_params.items()
    }
    params = gatecfg.params.copy()
    params.update(tmp_params)

    func, params_declaration = lib.getOpaque(name, gatecfg.type)
    if func is None:
        raise KeyError(f'Undefined {gatecfg.type} type of {name} opaque.')
    for p in params_declaration:
        if p.name not in params:
            pass
            # raise ValueError(
            #     f'{name} (type={gatecfg.type}) opaque of {qubits} missing parameter {k}.'
            # )

    args = tuple(args)

    sub_ctx = create_context(ctx, scopes=[*ctx.scopes, params])

    for cmd in func(sub_ctx, gatecfg.qubits, *args):
        _execute(sub_ctx, cmd)
    _ctx_update_biases(sub_ctx, ctx)
    _ctx_update_time(sub_ctx, ctx)
    _ctx_update_phases(sub_ctx, ctx)
    _ctx_update_waveforms(sub_ctx, ctx)
    _ctx_update_measurement_tasks(sub_ctx, ctx)


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
            print("    waveform =", w.tolist())
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


def assembly_align_left(qlisp, ctx: Context, lib: Library):
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
    for ch in ctx.biases:
        ctx.biases[ch] = 0
    _ctx_update_biases(ctx, ctx)
    ctx.end = max(ctx.time.values())

    code = QLispCode(cfg=ctx.cfg,
                     qlisp=ctx.qlisp,
                     waveforms=dict(ctx.waveforms),
                     measures=dict(ctx.measures),
                     end=ctx.end)
    return code


def assembly_align_right(qlisp, ctx: Context, lib: Library):
    raise NotImplementedError()

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
    for ch in ctx.biases:
        ctx.biases[ch] = 0
    _ctx_update_biases(ctx, ctx)
    ctx.end = max(ctx.time.values())

    code = QLispCode(cfg=ctx.cfg,
                     qlisp=ctx.qlisp,
                     waveforms=dict(ctx.waveforms),
                     measures=dict(ctx.measures),
                     end=ctx.end)
    return code
