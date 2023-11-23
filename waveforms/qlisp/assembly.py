import inspect
import warnings

from numpy import pi

from ..dicttree import NOTSET
from ..waveform import Waveform, WaveVStack, cos, sin, step, zero
from .base import (ADChannel, AWGChannel, Capture, Context, GateConfig,
                   MultADChannel, MultAWGChannel, QLispCode, QLispError, head)
from .library import Library


def _set_bias(ctx: Context, channel: str, bias: float | tuple):
    if isinstance(bias, tuple):
        bias, edge, buffer = bias
    else:
        edge, buffer = 0, 0
    if ctx.biases[channel] != bias:
        _, *qubits = channel
        t = max(ctx.time[q] for q in qubits)
        wav = (bias - ctx.biases[channel]) * step(edge) >> (t + buffer / 2)
        _play(ctx, channel, wav)
        ctx.biases[channel] = bias


def _add_bias(ctx: Context, channel: str, bias: float | tuple):
    if isinstance(bias, tuple):
        bias, edge, buffer = bias
        _set_bias(ctx, channel, (bias + ctx.biases[channel], edge, buffer))
    else:
        _set_bias(ctx, channel, bias)


def _set_time(ctx: Context, target: tuple, time: float):
    from waveforms.waveform import NDIGITS
    ctx.time[target] = round(time, NDIGITS)


def _add_time(ctx: Context, target: tuple, time: float):
    from waveforms.waveform import NDIGITS
    ctx.time[target] = round(ctx.time[target] + time, NDIGITS)


def _set_phase(ctx: Context, target: tuple, phase: float):
    ctx.phases_ext[target][1] = phase + ctx.phases_ext[target][0]


def _add_phase(ctx: Context, target: tuple, phase: float):
    ctx.phases_ext[target][1] += phase


def _set_phase_ext(ctx: Context, target: tuple, level: int, phase: float):
    ctx.phases_ext[target][level] = phase


def _add_phase_ext(ctx: Context, target: tuple, level: int, phase: float):
    ctx.phases_ext[target][level] += phase


def _play(ctx: Context, channel: tuple, wav: Waveform):
    if wav is zero():
        return
    name, *qubits = channel
    ch = ctx.get_awg_channel(name, qubits)
    if isinstance(ch, AWGChannel):
        ctx.waveforms[ch.name].append(wav)
    else:
        _mult_channel_play(ctx, wav, ch)


def _mult_channel_play(ctx: Context, wav, ch: MultAWGChannel):
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
        ctx.waveforms[ch.I.name].append(I)
    if ch.Q is not None:
        Q = (2 * wav * sin(-2 * pi * lofreq)).filter(high=2 * pi * lofreq)
        ctx.waveforms[ch.Q.name].append(Q)


def _capture(ctx: Context, cbit: tuple[str, int], info: Capture):
    hardware = ctx.get_ad_channel(info.qubit)
    name, index = cbit
    ctx.measures[name][index] = Capture(info.qubit, cbit, info.time,
                                        info.signal, info.params, hardware)


def _execute(ctx: Context, cmd: tuple[tuple, tuple | str]):

    (op, *args), target = cmd
    match op:
        case '!nop':
            return
        case '!set_time':
            _set_time(ctx, target, args[0])
        case '!set_phase':
            _set_phase(ctx, target, args[0])
        case '!set_phase_ext':
            _set_phase_ext(ctx, target, args[0], args[1])
        case '!set_bias':
            _set_bias(ctx, target, args[0])
        case '!play':
            _play(ctx, target, args[0])
        case '!capture':
            _capture(ctx, target, args[0])
        case '!add_time':
            _add_time(ctx, target, args[0])
        case '!add_phase':
            _add_phase(ctx, target, args[0])
        case '!add_phase_ext':
            _add_phase_ext(ctx, target, args[0], args[1])
        case '!add_bias':
            _add_bias(ctx, target, args[0])
        case '!set' | '!add':
            new_op = f'{op}_{args[0]}'
            new_op = {
                '!add_waveform': '!play',
                '!set_waveform': '!play',
                '!set_cbit': '!capture'
            }.get(new_op, new_op)
            warnings.warn(
                f"('{op}', '{args[0]}', ...) are deprecated, use '{new_op}' instead.",
                DeprecationWarning, 2)
            _execute(ctx, ((new_op, *args[1:]), target))
        case _:
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


def _get_opaque_params(ctx: Context, name: str, gate: str | tuple,
                       qubits: tuple):
    try:
        return ctx.cache[(name, gate, qubits)]
    except:
        pass
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
    gatecfg = ctx.get_gate_config(name, qubits, type=type)
    if gatecfg is None:
        gatecfg = GateConfig(name, qubits)

    tmp_params = {
        k: _try_to_call(v, gatecfg.params)
        for k, v in tmp_params.items()
    }
    params = gatecfg.params.copy()
    params.update(tmp_params)

    ctx.cache[(name, gate, qubits)] = args, params, gatecfg
    return args, params, gatecfg


def call_opaque(st: tuple, ctx: Context, lib: Library):
    name = head(st)
    gate, qubits = st

    args, params, gatecfg = _get_opaque_params(ctx, name, gate, qubits)

    func, *_ = lib.getOpaque(name, gatecfg.type)
    if func is None:
        raise KeyError(f'Undefined {gatecfg.type} type of {name} opaque.')

    ctx.scopes.append(params)
    for cmd in func(ctx, gatecfg.qubits, *args):
        _execute(ctx, cmd)
    ctx.scopes.pop()


def _allocQubits(ctx, qlisp):
    for i, q in enumerate(ctx.all_qubits):
        ctx.addressTable[q] = q
        ctx.addressTable[i] = q


def assembly_align_left(qlisp, ctx: Context, lib: Library):
    ctx.cache.clear()
    _allocQubits(ctx, qlisp)

    allQubits = set()

    for gate, *qubits in qlisp:
        if len(qubits) == 1:
            qubits = qubits[0]
        if isinstance(gate, str) and gate.startswith('!'):
            cmd = (gate, *qubits)
            _execute(ctx, cmd)
            continue
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
    ctx.end = max(ctx.time.values())

    waveforms = {ch: WaveVStack(waves) for ch, waves in ctx.waveforms.items()}

    measures = {}
    for var, m in ctx.measures.items():
        for i, c in m.items():
            measures[(var, i)] = c

    code = QLispCode(cfg=ctx.cfg,
                     qlisp=ctx.qlisp,
                     waveforms=waveforms,
                     measures=dict(sorted(measures.items())),
                     end=ctx.end)
    return code


def assembly_align_right(qlisp, ctx: Context, lib: Library):
    raise NotImplementedError()
