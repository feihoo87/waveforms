import asyncio
import logging
import time
from concurrent.futures import Future

from waveforms.scan_iter import Begin, End, StepStatus, scan_iters

from qlisp import READ, WRITE, Config, Library, ProgramFrame

from .base import Task

log = logging.getLogger(__name__)


def create_future(task: Task, step: int) -> asyncio.Future | Future:
    try:
        return asyncio.get_running_loop().create_future()
    except:

        return Future()


def exec_circuit(task: Task,
                 circuit: str | list,
                 lib: Library,
                 cfg: Config,
                 signal: str,
                 skip_compile: bool = False) -> int:
    """Execute a circuit."""
    from qlisp import compile

    if skip_compile and task.runtime.compiled_step > 0:
        task.runtime.skip_compile = True
        for cmd in task.runtime.prog.steps[-2].cmds:
            if (isinstance(cmd, READ) or cmd.address.endswith('.StartCapture')
                    or cmd.address.endswith('.CaptureMode')):
                task.runtime.cmds.append(cmd)
        task.runtime.prog.steps[-1].circuit = circuit.copy()
        task.runtime.prog.steps[-1].data_map = task.runtime.prog.steps[
            -2].data_map
        task.runtime.prog.steps[-1].context = task.runtime.prog.steps[
            -2].context.copy()
    else:
        task.runtime.skip_compile = False
        code = compile(circuit, lib=lib, cfg=cfg)
        code.signal = signal
        code.shots = task.shots
        try:
            context = task.runtime.prog.steps[-2].context.copy()
        except:
            context = {}
        cmds, dataMap = task.runtime.arch.assembly_code(code, context)
        task.runtime.prog.steps[-1].circuit = circuit.copy()
        task.runtime.prog.steps[-1].code = code
        task.runtime.prog.steps[-1].data_map.update(dataMap)
        task.runtime.prog.steps[-1].context = context
        task.runtime.cmds.extend(cmds)
    return task.runtime.prog.steps[-1].fut


def expand_task(task: Task):
    task.runtime.compiled_step = 0
    task.runtime.finished_step = 0
    task.runtime.prog.side_effects = {}
    task.runtime.prog.steps = []
    task.runtime.prog.shots = task.shots
    task.runtime.prog.signal = task.signal

    kw = task.scan_range()
    kw['trackers'] = kw.get('trackers', [])
    kw['trackers'].append(task.runtime.storage)

    for step in scan_iters(**kw):
        try:
            if task.runtime._kill_event.is_set():
                break
        except AttributeError:
            pass

        if isinstance(step, StepStatus):
            task.runtime.prog.steps.append(ProgramFrame(step, fut=Future()))
            task.runtime.cmds = []
        yield step
        if isinstance(step, StepStatus):
            flush_task(task)


def flush_task(task):
    from waveforms import Waveform

    if len(task.runtime.prog.steps[-1].cmds) > 0:
        return

    task.trig()
    task.runtime.prog.steps[-1].cmds = task.runtime.cmds

    unused = list(task.runtime.prog.side_effects.keys())
    for cmd in task.runtime.cmds:
        if isinstance(cmd.value, Waveform):
            task.runtime.prog.side_effects.setdefault(cmd.address, 'zero()')
        else:
            task.runtime.prog.side_effects.setdefault(
                cmd.address, task.cfg._history.query(cmd.address))
        try:
            unused.remove(cmd.address)
        except ValueError:
            pass
    if not task.runtime.skip_compile:
        for addr in unused:
            if isinstance(
                    task.runtime.prog.side_effects[addr],
                    str) and task.runtime.prog.side_effects[addr] == 'zero()':
                task.runtime.prog.steps[-1].cmds.insert(
                    0, WRITE(addr, 'zero()'))
                task.runtime.prog.side_effects.pop(addr)

    task.runtime.compiled_step += 1
    task.runtime.progress.max = task.runtime.compiled_step
    time.sleep(0.001)
