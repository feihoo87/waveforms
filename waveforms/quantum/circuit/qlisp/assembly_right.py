from numpy import pi
from waveforms.waveform import Waveform, cos, sin, step

from .assembly_left import _allocQubits, _ctx_update_biases, call_opaque
from .library import Library
from .qlisp import (ADChannel, AWGChannel, Context, GateConfig,
                    MeasurementTask, MultADChannel, MultAWGChannel, QLispCode,
                    QLispError, create_context, gateName)


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
