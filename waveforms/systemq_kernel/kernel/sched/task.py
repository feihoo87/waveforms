from __future__ import annotations

import copy
import functools
import hashlib
import importlib
import inspect
import pickle
import sys
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import (Any, Callable, Generator, Iterable, Literal, Optional,
                    Sequence, Type, Union)

import blinker
import numpy as np

from lib.gates import stdlib
from qlisp import (NOTSET, READ, TRIG, WRITE, Config, ConfigProxy, Library,
                   Signal, get_arch)
from storage.crud import tag_it
from storage.fs import get_data_path
from storage.ipy_events import get_current_cell_id
from storage.models import Record, Report

from ..config import QuarkLocalConfig
from .base import Executor
from .base import Task as BaseTask
from .progress import JupyterProgressBar
from .scan import exec_circuit, expand_task, flush_task

_default_lib = stdlib


def set_default_lib(lib: Library):
    global _default_lib
    _default_lib = lib


def get_default_lib() -> Library:
    return _default_lib


@dataclass
class CalibrationResult():
    suggested_calibration_level: int = 0
    parameters: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


@dataclass
class DataFrame():
    pass


SIGNAL = Literal['trace', 'iq', 'state', 'count', 'diag', 'population',
                 'trace_avg', 'iq_avg']
QLisp = list[tuple]


def _form_signal(sig):
    sig_tab = {
        'trace': Signal.trace,
        'iq': Signal.iq,
        'state': Signal.state,
        'count': Signal.count,
        'diag': Signal.diag,
        'population': Signal.population,
        'trace_avg': Signal.trace_avg,
        'iq_avg': Signal.iq_avg,
        'remote_trace_avg': Signal.remote_trace_avg,
        'remote_iq_avg': Signal.remote_iq_avg,
        'remote_state': Signal.remote_state,
        'remote_population': Signal.remote_population,
        'remote_count': Signal.remote_count,
    }
    if isinstance(sig, str):
        if sig == 'raw':
            sig = 'iq'
        try:
            return sig_tab[sig]
        except KeyError:
            pass
    elif isinstance(sig, Signal):
        return sig
    raise ValueError(f'unknow type of signal "{sig}".'
                     f" optional signal types: {list(sig_tab.keys())}")


class TagSet(set):
    updated = blinker.signal('updated')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add(self, tag: str):
        if tag in self:
            return
        super().add(tag)
        self.updated.send(self, tag_text=tag)

    def __repr__(self) -> str:
        return super().__repr__()


def update_tags(sender: TagSet, tag_text, obj: Any, tag_set_id, db) -> None:
    if id(sender) == tag_set_id:
        tag_it(db, tag_text, obj)


class Task(BaseTask):

    def __init__(self,
                 signal: Union[SIGNAL, Signal] = Signal.state,
                 shots: int = 1024,
                 calibration_level: int = 0,
                 lib: Library | None = None,
                 arch: str = 'baqis',
                 task_priority=0,
                 **kw):
        """
        Args:
            signal: the signal to be measured
            shots: the number of shots to be measured
            calibration_level: calibration level (0~100)
        """
        super().__init__()
        self.parent = None
        self.container = None
        self.signal = signal
        self.shots = shots
        self.lib = get_default_lib() if lib is None else lib
        self.calibration_level = calibration_level
        self.no_record = False
        self.reshape_record = True
        self.debug_mode = False
        self.task_priority = task_priority
        self._tags: set = TagSet()
        self.__cfg = None
        self.__runtime.arch = get_arch(arch)
        self._hooks: list[Callable[[Task, int, Executor], None]] = []
        self._init_hooks: list[Callable[[Task, int, Executor], None]] = []
        self._final_hooks: list[Callable[[Task, int, Executor], None]] = []

    @property
    def signal(self):
        return self.__signal

    @signal.setter
    def signal(self, sig: Union[SIGNAL, Signal]):
        self.__signal = _form_signal(sig)

    def __repr__(self):
        try:
            return f"{self.name}({self.id}, calibration_level={self.calibration_level}, record_id={self.runtime.record.id})"
        except:
            return f"{self.name}({self.id}, calibration_level={self.calibration_level})"

    @property
    def db(self):
        if self.runtime.db is None:
            self.runtime.db = self.kernel.session()
        return self.runtime.db

    @property
    def cfg(self):
        if self.__cfg is None:
            self.__cfg = QuarkLocalConfig(
                copy.deepcopy(self.runtime.prog.snapshot))
            self.__cfg._history = QuarkLocalConfig(self.runtime.prog.snapshot)
        return self.__cfg

    @property
    def tags(self):
        return self._tags

    @property
    def record(self) -> Optional[Record]:
        return self.runtime.record

    @cached_property
    def task_hash(self):
        kwds = self.runtime.prog.task_arguments
        kwds = {k: kwds[k] for k in sorted(kwds.keys())}
        buf = pickle.dumps(kwds)
        return hashlib.sha256(buf).digest()

    def is_children_of(self, task: Task) -> bool:
        return self.parent is not None and self.parent == task.id

    def get_parent(self) -> Optional[Task]:
        if self.parent is not None:
            return self.kernel.get_task_by_id(self.parent)
        return None

    def set_record(self, dims: list[tuple[str, str]], vars: list[tuple[str,
                                                                       str]],
                   coords: dict[str, Sequence]) -> None:

        if self.no_record:
            return
        if self.runtime.record is not None:
            return
        dims, dims_units = list(zip(*dims))
        vars, vars_units = list(zip(*vars))
        self.runtime.record = self.create_record()
        self.db.add(self.runtime.record)
        self.db.commit()

    def create_record(self, db=None) -> Record:
        """Create a record"""

        if db is None:
            db = self.db

        file, key = self.data_path.split(':/')
        file = get_data_path() / (file + '.hdf5')

        record = Record(file=str(file), key=key)
        record.app = self.name
        record.task_hash = self.task_hash
        record.cell_id = get_current_cell_id()
        if self.parent is not None:
            record.parent_id = self.kernel.get_task_by_id(
                self.parent).runtime.record.id
        for tag_text in self.tags:
            tag_it(db, tag_text, record)

        receiver = functools.partial(update_tags,
                                     obj=record,
                                     db=self.db,
                                     tag_set_id=id(self.tags))
        self.tags.updated.connect(receiver)
        record._blinker_update_tag_receiver = receiver  # hold a reference
        return record

    def create_report(self, db=None) -> Report:
        """create a report"""
        if db is None:
            db = self.db

        file, key = self.data_path.split(':/')
        file = get_data_path() / (file + '.hdf5')

        rp = Report(file=str(file), key=key)
        rp.task_hash = self.task_hash
        for tag_text in self.tags:
            tag_it(db, tag_text, rp)

        receiver = functools.partial(update_tags,
                                     obj=rp,
                                     db=self.db,
                                     tag_set_id=id(self.tags))
        self.tags.updated.connect(receiver)
        rp._blinker_update_tag_receiver = receiver  # hold a reference
        return rp

    def set(self, key: str, value: Any, cache: bool = True) -> None:
        self.runtime.cmds.append(WRITE(key, value))
        self.cfg.update(key, value, cache=cache)

    def get(self, key: str, default: Any = NOTSET) -> Any:
        """
        return the value of the key in the kernel config
        """
        ret = self.cfg.query(key)
        if isinstance(ret, tuple) and ret[0] is NOTSET:
            if default is NOTSET:
                raise KeyError(f"key {key} not found")
            return default
        return ret

    def push(self, frame: dict):
        if 'data' not in frame:
            self.runtime.result['data'].append(None)
        for k, v in frame['data'].items():
            self.runtime.result[k].append(v)

    def exec(self,
             circuit: QLisp,
             lib: Optional[Library] = None,
             cfg: Union[Config, ConfigProxy, None] = None,
             skip_compile: bool = False):

        if lib is None:
            lib = self.lib
        if cfg is None:
            cfg = self.cfg

        self._collect_used_elements(circuit)
        return exec_circuit(self,
                            circuit,
                            lib=lib,
                            cfg=cfg,
                            signal=self.signal,
                            skip_compile=skip_compile)

    def flush(self):
        flush_task(self)

    def _collect_used_elements(self, circuit: QLisp) -> None:
        all_qubits = self.cfg._getAllQubitLabels()
        for _, qubits in circuit:
            if not isinstance(qubits, tuple):
                qubits = (qubits, )
            for qubit in qubits:
                if qubit not in all_qubits:
                    raise ValueError(f"{qubit} is not in the config, "
                                     f"please add it to the config")
                # self.tags.add(qubit)
                self.runtime.used_elements.add(qubit)
                q = self.cfg.query(qubit)
                try:
                    self.runtime.used_elements.add(q['probe'])
                except:
                    pass
                try:
                    for coupler in q['couplers']:
                        self.runtime.used_elements.add(coupler)
                except:
                    pass

    def measure(self, keys, labels=None):
        if labels is None:
            labels = keys
        dataMap = {'data': {label: key for key, label in zip(keys, labels)}}
        self.runtime.prog.data_maps[-1].update(dataMap)
        self.runtime.cmds.extend([READ(key) for key in keys])

    def trig(self) -> None:
        cmds = self.get('station.triggercmds')
        for cmd in cmds:
            self.runtime.cmds.append(TRIG(cmd))

    def scan(self) -> Generator:
        self.set_record(dims=[('index', ''), ('shots', ''), ('cbits', '')],
                        vars=[('data', ''), ('state', '')],
                        coords={
                            'shots': np.arange(self.shots),
                        })

        yield from expand_task(self)
        with self.runtime._status_lock:
            if self.status == 'compiling':
                self.runtime.status = 'pending'

    @cached_property
    def data_path(self) -> str:
        if self.parent:
            name = '/'.join([
                self.kernel.get_task_by_id(self.parent).data_path, 'sub_data',
                f"{self.name}_{self.runtime.sub_index}"
            ])
            return name
        else:
            file_name = self.get('station.sample')
            time_str = time.strftime('%Y-%m-%d-%H-%M-%S',
                                     time.localtime(self.runtime.started_time))
            name = f"{self.name}"
            return f"{file_name}:/{name}"

    def clean_side_effects(self) -> None:
        self.kernel.clean_side_effects(self, self.kernel.get_executor(), False)

    @abstractmethod
    def scan_range(self) -> dict:
        pass

    @abstractmethod
    def main(self):
        pass

    def depends(self) -> list[tuple[str, tuple, dict]]:
        """
        """
        return []

    def check(self, lastest: bool = False) -> bool:
        """
        If the latest scan was finished sucessfully in life time,
        then return the finished time, otherwise return -1
        """
        raise NotImplementedError()

    def analyze(self, data) -> CalibrationResult:
        """
        return a CalibrationResult object
        """
        raise NotImplementedError()

    def result(self, reshape: bool = True, skip=None) -> dict:
        self.kernel._fetch_data(self, self.kernel.get_executor())
        try:
            record_id = self.record.id
        except:
            record_id = None
        ret = {
            'calibration_level': self.calibration_level,
            'index': {},
            'meta': {
                'id': record_id,
                'pos': {},
                'iteration': {},
                'reshaped': reshape,
                'system': self.runtime.system_info,
            }
        }

        if skip is None or skip == 0:
            for key in self.runtime.storage._frozen_keys:
                ret['index'][key] = self.runtime.storage[key]
            ret['meta']['key_levels'] = self.runtime.storage._key_levels
            ret['meta']['pos'] = self.runtime.storage.pos
            ret['meta']['iteration'] = self.runtime.storage.iteration
            ret['meta']['arguments'] = self.runtime.prog.task_arguments
            ret['meta']['config'] = self.runtime.prog.snapshot
        ret['meta']['shape'] = self.runtime.storage.shape

        for key in self.runtime.storage.storage:
            if key not in self.runtime.storage._frozen_keys:
                if reshape:
                    ret[key] = self.runtime.storage[key]
                elif skip is None:
                    data, *_ = self.runtime.storage.get(key, skip=0)
                    try:
                        ret[key] = np.asarray(data)
                    except:
                        ret[key] = data
                else:
                    data, iteration, pos = self.runtime.storage.get(key,
                                                                    skip=skip)
                    ret['meta']['pos'][key] = pos
                    ret['meta']['iteration'][key] = iteration
                    ret[key] = data
        return ret

    def cancel(self):
        for fut, evt in self.runtime.threads.values():
            evt.set()
        self.runtime.progress.finish(False)
        self.kernel.get_executor().cancel(self.id)
        with self.runtime._status_lock:
            self.runtime.status = 'cancelled'

    def join(self, timeout=None):
        try:
            if timeout is None:
                while 'compile' not in self.runtime.threads:
                    time.sleep(0.1)
                self.runtime.threads['compile'][0].result()
                if self.runtime.dry_run:
                    return
                while 'run' not in self.runtime.threads:
                    time.sleep(0.1)
                self.runtime.threads['run'][0].result()
            else:
                start = time.time()
                while 'compile' not in self.runtime.threads:
                    used = time.time() - start
                    if used > timeout:
                        raise TimeoutError(f"timeout {timeout}")
                    time.sleep(0.1)
                used = time.time() - start
                self.runtime.threads['compile'][0].result(
                    max(timeout - used, 0.001))
                if self.runtime.dry_run:
                    return
                while 'run' not in self.runtime.threads:
                    used = time.time() - start
                    if used > timeout:
                        raise TimeoutError(f"timeout {timeout}")
                    time.sleep(0.1)
                used = time.time() - start
                self.runtime.threads['run'][0].result(
                    max(timeout - used, 0.001))
        finally:
            for fut, evt in self.runtime.threads.values():
                evt.set()

    def bar(self):
        bar = JupyterProgressBar(description=self.name.split('.')[-1])
        bar.listen(self.runtime.progress)
        bar.display()

    def check_level(self) -> int:
        return 90

    def plot_prog_frame(
        self,
        step,
        start=0,
        stop=99e-6,
        keys=None,
        sample_rate=None,
        fig=None,
        axis=None,
        stack_hight=2,
    ):
        import matplotlib.pyplot as plt
        from waveforms import Waveform

        waveforms = {}

        for cmd in self.runtime.prog.steps[step].cmds:
            if isinstance(cmd.value, Waveform):
                el, *_, ch = cmd.address.split('.')
                if keys is not None and el not in keys:
                    continue
                if el not in waveforms:
                    waveforms[el] = {}
                waveforms[el][ch] = cmd.value

        if sample_rate is None:
            t = np.linspace(start, stop, 1001)
        else:
            t = np.arange(start, stop, 1 / sample_rate)

        if fig is None:
            fig = plt.figure(
                figsize=(8,
                         max(1 + round(stack_hight * len(waveforms) / 4), 2)))
        if axis is None:
            axis = fig.add_subplot(1, 1, 1)
        colors = {}
        offset = 0
        for el in sorted(waveforms):
            group = waveforms[el]
            for ch, wav in group.items():
                color = colors.setdefault(ch, f'C{len(colors)%10}')
                axis.plot(t / 1e-6, offset + wav(t), color=color, lw=1)
            axis.text(start / 1e-6 - 0.07 * (stop - start) / 1e-6,
                      offset,
                      el,
                      va='center')
            offset -= stack_hight

        axis.set_axis_off()


class ContainerTask(Task):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elements = None


class App(Task):

    def plot(self, fig, result):
        raise NotImplementedError()


class UserInput(App):

    def __init__(self, *keys):
        super().__init__()
        self.elements = None
        self.keys = keys
        self.no_record = True

    @property
    def name(self):
        return 'UserInput'

    def check(self):
        # never pass
        return -1

    def analyze(self, data) -> CalibrationResult:
        # always pass
        return CalibrationResult(suggested_calibration_level=100)

    def scan_range(self) -> Union[Iterable, Generator]:
        return {}

    def main(self):
        for key in self.keys:
            self.set(key, input(f"{key} = "), cache=False)

    def result(self):
        return {'data': []}


class CalibrateGate(App):

    def __init__(self, gate_name, qubits, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_name = gate_name
        self.qubits = qubits

    @property
    def name(self):
        return 'CalibrateGate'

    def check(self):
        # never pass
        return -1

    def analyze(self, data) -> CalibrationResult:
        # always pass
        return CalibrationResult(suggested_calibration_level=100)

    def scan_range(self) -> Union[Iterable, Generator]:
        return {}

    def main(self):
        pass

    def result(self):
        return {'data': []}


class RunCircuits(App):

    def __init__(self,
                 circuits,
                 shots=1024,
                 signal='state',
                 arch='baqis',
                 lib=None,
                 cfg=None,
                 settings=None,
                 cmds=[],
                 calibration_mode=False,
                 **kw):
        super().__init__(signal=signal, shots=shots, arch=arch, **kw)
        self.circuits = circuits
        self.custom_lib = lib
        self.custom_cfg = cfg
        self.cmds = cmds
        self.settings = settings
        self.calibration_mode = calibration_mode

    @property
    def name(self):
        return 'RunCircuits'

    def depends(self) -> list[tuple[str, tuple, dict]]:
        from waveforms import compile
        from waveforms.quantum.circuit.qlisp.qlisp import gateName

        if self.calibration_mode:
            return []

        deps = set()

        for circuit in self.circuits:
            for st in compile(circuit,
                              no_link=True,
                              lib=self.custom_lib,
                              cfg=self.custom_cfg):
                deps.add((CalibrateGate, (gateName(st), st[1])))

        return list(deps)

    def check(self):
        # never pass
        return -1

    def analyze(self, data) -> CalibrationResult:
        # always pass
        return CalibrationResult(suggested_calibration_level=100)

    def scan_range(self):
        ret = {'iters': {'circuit': self.circuits}}
        if self.settings is not None:
            if isinstance(self.settings, dict):
                ret['iters']['settings'] = [self.settings] * len(self.circuits)
            else:
                ret['iters']['settings'] = self.settings
        return ret

    def main(self):
        for step in self.scan():
            self.runtime.cmds.extend(self.cmds)
            for k, v in step.kwds.get('settings', {}).items():
                self.set(k, v)
            self.exec(step.kwds['circuit'],
                      lib=self.custom_lib,
                      cfg=self.custom_cfg)


def _getAppClass(name: str) -> Type[App]:
    *module, name = name.split('.')
    if name in ['RunCircuits', 'CalibrateGate', 'UserInput']:
        return globals()[name],
    elif module == 'Scan':
        from kernel.terminal.scan import Scan
        return Scan, '.'.join(module[1:] + [name])
    elif len(module) == 0:
        module = sys.modules['__main__']
    else:
        module = '.'.join(module)
        module = importlib.import_module(module)
    return getattr(module, name),


def create_task(taskInfo: Union[tuple, Task]) -> Task:
    """
    create a task from a string or a class

    Args:
        taskInfo: a task or tuple of (app, [args, [kargs,]])

            app: a string or a subclass of Task
            args: arguments for the class
            kwds: keyword arguments for the class

    Returns:
        a task
    """
    if isinstance(taskInfo, Task):
        return copy_task(taskInfo)

    app, *other = taskInfo
    if isinstance(app, str):
        app, *name = _getAppClass(app)
        other = tuple(name) + tuple(other)
    if len(other) >= 1:
        args = other[0]
    else:
        args = ()
    if len(other) >= 2:
        kwds = other[1]
    else:
        kwds = {}
    task = app(*args, **kwds)
    bs = inspect.signature(app).bind(*args, **kwds)
    task.runtime.prog.task_arguments = bs.arguments
    task.runtime.prog.meta_info['arguments'] = bs.arguments
    return task


def copy_task(task: Task) -> Task:
    memo = {
        id(task.parent): None,
        id(task.container): None,
    }
    return copy.deepcopy(task, memo)
