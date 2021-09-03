import logging
import threading
import time
from collections import deque
from typing import Any, Optional, Union

from waveforms.quantum.circuit.qlisp.config import Config
from waveforms.quantum.circuit.qlisp.library import Library
from waveforms.waveform import Waveform

from .base import READ, Task, ProgramFrame
from .scan_iters import scan_iters

log = logging.getLogger(__name__)


def exec_circuit(task: Task,
                 circuit: Union[str, list],
                 lib: Library,
                 cfg: Config,
                 signal: str,
                 skip_compile: bool = False) -> int:
    """Execute a circuit."""
    from waveforms import compile

    if skip_compile and task.runtime.step > 0:
        for cmd in task.runtime.prog.steps[-2].cmds:
            if (isinstance(cmd, READ) or cmd.address.endswith('.StartCapture')
                    or cmd.address.endswith('.CaptureMode')):
                task.runtime.cmds.append(cmd)
        task.runtime.prog.steps[-1].data_map = task.runtime.prog.steps[-2].data_map
    else:
        code = compile(circuit, lib=lib, cfg=cfg)
        code.signal = signal
        code.shots = task.shots
        cmds, dataMap = task.runtime.arch.assembly_code(code)
        task.runtime.prog.steps[-1].code = code
        task.runtime.prog.steps[-1].data_map.update(dataMap)
        task.runtime.cmds.extend(cmds)
    return task.runtime.step


def expand_task(task: Task):
    task.runtime.step = 0
    task.runtime.prog.index = []
    task.runtime.prog.side_effects = {}
    task.runtime.prog.steps = []
    task.runtime.prog.shots = task.shots
    task.runtime.prog.signal = task.signal

    for step in scan_iters(**task.scan_range()):
        try:
            if threading.current_thread()._kill_event.is_set():
                break
        except AttributeError:
            pass

        task.runtime.prog.index.append(step)
        task.runtime.prog.steps.append(ProgramFrame(task.runtime.step))

        for k, v in step.kwds.items():
            if k in task.runtime.result['index']:
                task.runtime.result['index'][k].append(v)
            else:
                task.runtime.result['index'][k] = [v]

        task.runtime.cmds = []
        yield step
        task.trig()
        task.runtime.prog.steps[-1].cmds = task.runtime.cmds

        for cmd in task.runtime.cmds:
            if isinstance(cmd.value, Waveform):
                task.runtime.prog.side_effects.setdefault(
                    cmd.address, 'zero()')
            else:
                task.runtime.prog.side_effects.setdefault(
                    cmd.address, task.cfg._history.query(cmd.address))
        task.runtime.step += 1
