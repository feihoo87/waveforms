import logging
import threading
import time
from collections import deque
from typing import Any, Optional, Union

from waveforms.quantum.circuit.qlisp.config import Config
from waveforms.quantum.circuit.qlisp.library import Library
from waveforms.waveform import Waveform

from .base import Task, READ
from .scan_iters import scan_iters

log = logging.getLogger(__name__)


def exec_circuit(task: Task, circuit: Union[str, list], lib: Library,
                 cfg: Config, signal: str, compile_once: bool) -> int:
    """Execute a circuit."""
    from waveforms import compile
    from waveforms.backends.quark.executable import getCommands

    task.runtime.prog.steps[-1][2].extend(task.runtime.cmds)
    if task.runtime.step == 0 or not compile_once:
        code = compile(circuit, lib=lib, cfg=cfg)
        cmds, dataMap = getCommands(code, signal=signal, shots=task.shots)
        task.runtime.cmds.extend(cmds)
        task.runtime.prog.data_maps[-1].update(dataMap)
        task.runtime.prog.steps[-1][0].extend(circuit)
    else:
        for cmd in task.runtime.prog.commands[-1]:
            if (isinstance(cmd, READ) or cmd.address.endswith('.StartCapture')
                    or cmd.address.endswith('.CaptureMode')):
                task.runtime.cmds.append(cmd)
        task.runtime.prog.data_maps[-1] = task.runtime.prog.data_maps[0]
        task.runtime.prog.steps[-1][2].extend(task.runtime.cmds)
        task.runtime.prog.steps[-1][3].update(task.runtime.prog.data_maps[-1])
    return task.runtime.step


def expand_task(task: Task):
    task.runtime.step = 0
    task.runtime.prog.index = []
    task.runtime.prog.commands = []
    task.runtime.prog.data_maps = []
    task.runtime.prog.side_effects = {}
    task.runtime.prog.steps = []
    task.runtime.prog.shots = task.shots
    task.runtime.prog.signal = task.signal

    iters = task.scan_range()
    if isinstance(iters, tuple) and len(iters) == 2:
        iters, filter_func = iters
    elif isinstance(iters, dict):
        iters, filter_func = iters, None
    else:
        raise ValueError(f"Invalid scan range: {iters}")
    for step in scan_iters(iters, filter_func):
        try:
            if threading.current_thread()._kill_event.is_set():
                break
        except AttributeError:
            pass

        task.runtime.prog.index.append(step)
        task.runtime.prog.data_maps.append({})
        task.runtime.prog.steps.append(([], {}, [], {}))

        for k, v in step.kwds.items():
            if k in task.runtime.result['index']:
                task.runtime.result['index'][k].append(v)
            else:
                task.runtime.result['index'][k] = [v]

        task.runtime.cmds = []
        yield step
        task.trig()
        task.runtime.prog.commands.append(task.runtime.cmds)

        for cmd in task.runtime.cmds:
            if isinstance(cmd.value, Waveform):
                task.runtime.prog.side_effects.setdefault(
                    cmd.address, 'zero()')
            else:
                task.runtime.prog.side_effects.setdefault(
                    cmd.address, task.cfg._history.query(cmd.address))
        task.runtime.step += 1
