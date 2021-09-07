from __future__ import annotations

import copy
import functools
import importlib
import sys
import time
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import (Any, Generator, Iterable, Literal, Optional, Sequence,
                    Type, Union)

import blinker
import numpy as np
from waveforms.math.fit import count_to_diag, countState
from waveforms.quantum.circuit.qlisp import get_arch
from waveforms.quantum.circuit.qlisp.config import Config, ConfigProxy
from waveforms.quantum.circuit.qlisp.library import Library
from waveforms.storage.crud import tag
from waveforms.storage.models import Record, Report

from .base import READ, TRIG, WRITE
from .base import Task as BaseTask
from .scan import exec_circuit, expand_task


@dataclass
class CalibrationResult():
    suggested_calibration_level: int = 0
    parameters: dict = field(default_factory=dict)


SIGNAL = Literal['trace', 'raw', 'state', 'count', 'diag']
QLisp = list[tuple]


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
        obj.tags.append(tag(db, tag_text))


class Task(BaseTask):
    def __init__(self,
                 signal: SIGNAL = 'count',
                 shots: int = 1024,
                 calibration_level: int = 0,
                 arch: str = 'baqis'):
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
        self.calibration_level = calibration_level
        self.no_record = False
        self._tags: set = TagSet()
        self.__cfg = None
        self.__runtime.arch = get_arch(arch)

    def __del__(self):
        try:
            self.kernel.executor.free(self.id)
        except:
            pass

    def __repr__(self):
        return f"{self.name}({self.id}, calibration_level={self.calibration_level})"

    @property
    def db(self):
        if self.runtime.db is None:
            self.runtime.db = self.kernel.session()
        return self.runtime.db

    @property
    def cfg(self):
        from waveforms.backends.quark.quarkconfig import QuarkLocalConfig

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

    def create_record(self) -> Record:
        """Create a record"""
        file, key = self.data_path.split(':/')
        file = self.kernel.data_path / (file + '.hdf5')

        record = Record(file=str(file), key=key)
        record.app = self.name
        record.base_path = file.parent
        for tag_text in self.tags:
            t = tag(self.db, tag_text)
            record.tags.append(t)

        receiver = functools.partial(update_tags,
                                     obj=record,
                                     db=self.db,
                                     tag_set_id=id(self.tags))
        self.tags.updated.connect(receiver)
        record._blinker_update_tag_receiver = receiver  # hold a reference
        return record

    def create_report(self) -> Report:
        """create a report"""
        file, key = self.data_path.split(':/')
        file = self.kernel.data_path / (file + '.hdf5')

        rp = Report(file=str(file), key=key)
        rp.base_path = file.parent
        for tag_text in self.tags:
            t = tag(self.db, tag_text)
            rp.tags.append(t)

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

    def get(self, key: str) -> Any:
        """
        return the value of the key in the kernel config
        """
        return self.cfg.query(key)

    def exec(self,
             circuit: QLisp,
             lib: Optional[Library] = None,
             cfg: Union[Config, ConfigProxy, None] = None,
             compile_once: bool = None,
             skip_compile: bool = False):
        from waveforms import stdlib

        if lib is None:
            lib = stdlib
        if cfg is None:
            cfg = self.cfg

        if compile_once is not None:
            if self.runtime.step == 0:
                warnings.warn(
                    "compile_once is deprecated, use skip_compile instead",
                    DeprecationWarning, 2)
            if compile_once and self.runtime.step != 0:
                skip_compile = True
            else:
                skip_compile = False

        self._collect_used_elements(circuit)
        return exec_circuit(self,
                            circuit,
                            lib=lib,
                            cfg=cfg,
                            signal=self.signal,
                            skip_compile=skip_compile)

    def _collect_used_elements(self, circuit: QLisp) -> None:
        for _, qubits in circuit:
            if not isinstance(qubits, tuple):
                qubits = (qubits, )
            for qubit in qubits:
                self.tags.add(qubit)
                self.runtime.used_elements.add(qubit)
                q = self.cfg.query(qubit)
                self.runtime.used_elements.add(q['probe'])
                for coupler in q['couplers']:
                    self.runtime.used_elements.add(coupler)

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
        self.kernel.clean_side_effects(self)

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

    def check(self) -> bool:
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

    def _fetch_result(self):
        i = len(self.runtime.result['data'])
        additional = self.kernel.executor.fetch(self.id, i)
        if isinstance(additional, str):
            additional = []
        for step, (result,
                   prog_frame) in enumerate(zip(additional,
                                                self.runtime.prog.steps[i:]),
                                            start=i):
            if prog_frame.data_map['signal'] in ['count', 'diag']:
                result['count'] = countState(result['state'])
            if prog_frame.data_map['signal'] == 'diag':
                result['diag'] = count_to_diag(result['count'])
            self.runtime.result['data'].append(result['data'])
            self.runtime.result['states'].append(result.get('state', None))
            self.runtime.result['counts'].append(result.get('count', None))
            self.runtime.result['diags'].append(result.get('diag', None))
            if self.runtime.record is not None:
                cbits = result['data'].shape[-1]
                if 'cbits' in self.runtime.record.dims and 'cbits' not in self.runtime.record.coords:
                    self.runtime.record.coords['cbits'] = np.arange(cbits)

        return {
            'calibration_level': self.calibration_level,
            'index': self.runtime.result['index'],
            'data': np.asarray(self.runtime.result['data']),
            'states': np.asarray(self.runtime.result['states']),
            'counts': self.runtime.result['counts'],
            'diags': np.asarray(self.runtime.result['diags'])
        }

    def result(self):
        return {
            'calibration_level': self.calibration_level,
            'index': self.runtime.result['index'],
            'data': np.asarray(self.runtime.result['data']),
            'states': np.asarray(self.runtime.result['states']),
            'counts': self.runtime.result['counts'],
            'diags': np.asarray(self.runtime.result['diags'])
        }

    def cancel(self):
        for t in self.runtime.threads.values():
            t.kill()
        with self.runtime._status_lock:
            self.runtime.status = 'cancelled'

    def join(self, timeout=None):
        try:
            start = time.time()
            while True:
                if self.runtime.status in ['cancelled', 'finished']:
                    break
                if timeout is not None and time.time() - start > timeout:
                    raise TimeoutError(f"{self.name}(id={self.id}) timeout.")
                time.sleep(0.1)
        finally:
            for t in self.runtime.threads.values():
                t.kill()


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


class RunCircuits(App):
    def __init__(self,
                 circuits,
                 shots=1024,
                 signal='state',
                 arch='baqis',
                 lib=None,
                 cfg=None,
                 settings=None,
                 cmds=[]):
        super().__init__(signal=signal, shots=shots, arch=arch)
        self.circuits = circuits
        self.custom_lib = lib
        self.custom_cfg = cfg
        self.cmds = cmds
        self.settings = settings

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
    if len(module) == 0:
        module = sys.modules['__main__']
    else:
        module = '.'.join(module)
        module = importlib.import_module(module)
    return getattr(module, name)


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
        app = _getAppClass(app)
    if len(other) >= 1:
        args = other[0]
    else:
        args = ()
    if len(other) >= 2:
        kwds = other[1]
    else:
        kwds = {}
    task = app(*args, **kwds)
    return task


def copy_task(task: Task) -> Task:
    memo = {
        id(task.parent): None,
        id(task.container): None,
        id(task._db_sessions): {},
    }
    return copy.deepcopy(task, memo)
