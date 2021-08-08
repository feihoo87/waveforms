from __future__ import annotations

import copy
import importlib
import logging
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import (Any, Generator, Iterable, Literal, Optional, Sequence,
                    Type, Union)

import numpy as np
from waveforms.quantum.circuit.qlisp.config import Config, ConfigProxy
from waveforms.quantum.circuit.qlisp.library import Library
from waveforms.storage.models import Record


class COMMAND():
    """Commands for the scheduler"""
    __slots__ = ('key', 'value')

    def __init__(self, key: str, value: Any):
        self.key = key
        self.value = value


class READ(COMMAND):
    """Read a value from the scheduler"""
    def __init__(self, key: str):
        super().__init__(key, 'READ')

    def __repr__(self) -> str:
        return f"READ({self.key})"


class WRITE(COMMAND):
    def __repr__(self) -> str:
        return f"WRITE({self.key}, {self.value})"


class TRIG(COMMAND):
    """Trigger the system"""
    def __init__(self, key: str):
        super().__init__(key, 0)

    def __repr__(self) -> str:
        return f"TRIG({self.key})"


@dataclass
class TaskRuntime():
    step: int = 0
    sub_index: int = 0
    status: str = 'not submited'
    created_time: float = field(default_factory=time.time)
    started_time: float = field(default_factory=time.time)
    finished_time: float = field(default=-1)
    dataMaps: list = field(default_factory=list)
    data: list = field(default_factory=list)
    cmds: list = field(default_factory=list)
    cmds_list: list = field(default_factory=list)
    feedback_buffer: Any = None
    side_effects: dict = field(default_factory=dict)
    result: dict = field(default_factory=lambda: {
        'index': [],
        'states': [],
        'counts': [],
        'diags': []
    })
    record: Optional[Record] = None


@dataclass
class CalibrationResult():
    suggested_calibration_level: int = 0
    parameters: dict = field(default_factory=dict)


SIGNAL = Literal['trace', 'raw', 'state', 'count', 'diag']
QLisp = list[tuple]


class Task(ABC):
    def __init__(self,
                 signal: SIGNAL = 'count',
                 shots: int = 1024,
                 calibration_level: int = 0):
        """
        Args:
            signal: the signal to be measured
            shots: the number of shots to be measured
            calibration_level: calibration level (0~100)
        """
        self.parent = None
        self.container = None
        self.id = None
        self.kernel = None
        self.db = None
        self.signal = signal
        self.shots = shots
        self.calibration_level = calibration_level
        self._runtime = TaskRuntime()

    def __del__(self):
        try:
            self.kernel.executor.free(self.id)
        except:
            pass

    def __repr__(self):
        return f"{self.app_name}({self.id}, calibration_level={self.calibration_level})"

    @property
    def app_name(self):
        return f"{self.__class__.__module__}.{self.__class__.__name__}"

    @property
    def log(self):
        return logging.getLogger(f"{self.app_name}")

    @property
    def status(self):
        return self._runtime.status

    @property
    def cfg(self) -> Config:
        return self.kernel.cfg

    def is_children_of(self, task: Task) -> bool:
        return self.parent is not None and self.parent == task.id

    def get_parent(self) -> Optional[Task]:
        if self.parent is not None:
            return self.kernel.get_task_by_id(self.parent)
        return None

    def set_record(self, dims: list[tuple[str, str]], vars: list[tuple[str,
                                                                       str]],
                   coords: dict[str, Sequence]) -> None:
        if self._runtime.record is not None:
            return
        dims, dims_units = list(zip(*dims))
        vars, vars_units = list(zip(*vars))
        file, key = self.data_path.split(':/')
        # file = self.kernel.data_path / (file + '.hdf5')
        file = self.kernel.data_path / (file + '.zip')
        self._runtime.record = Record(str(file), key, dims, vars, dims_units,
                                      vars_units, coords)
        self._runtime.record.app = self.app_name
        self.db.add(self._runtime.record)
        self.db.commit()

    def set(self, key: str, value: Any, cache: bool = True) -> None:
        self._runtime.cmds.append(WRITE(key, value))
        self.cfg.update(key, value, cache=cache)

    def get(self, key: str) -> Any:
        """
        return the value of the key in the kernel config
        """
        return self.kernel.query(key)

    def exec(self,
             circuit: QLisp,
             lib: Optional[Library] = None,
             cfg: Union[Config, ConfigProxy, None] = None,
             compile_once: bool = False):
        step = self.kernel._exec(self,
                                 circuit,
                                 lib=lib,
                                 cfg=cfg,
                                 signal=self.signal,
                                 compile_once=compile_once)
        return step

    def measure(self, keys, labels=None):
        self.kernel._measure(self, keys, labels)

    def trig(self) -> None:
        cmds = self.get('station.triggercmds')
        for cmd in cmds:
            self._runtime.cmds.append(TRIG(cmd))

    def scan(self) -> Generator:
        self.set_record(dims=[('index', ''), ('shots', ''), ('cbits', '')],
                        vars=[('data', ''), ('state', '')],
                        coords={
                            'shots': np.arange(self.shots),
                        })
        yield from self.kernel.scan(self)

    def feedback(self, obj: Any) -> None:
        self.kernel.feedback(self, obj)

    def get_feedback(self) -> Any:
        return self.kernel.get_feedback(self)

    @cached_property
    def data_path(self) -> str:
        if self.parent:
            name = '/'.join([
                self.kernel.get_task_by_id(self.parent).data_path, 'sub_data',
                f"{self.__class__.__name__}_{self._runtime.sub_index}"
            ])
            return name
        else:
            file_name = self.get('station.sample')
            time_str = time.strftime(
                '%Y-%m-%d-%H-%M-%S',
                time.localtime(self._runtime.started_time))
            name = f"{self.__class__.__name__}_{time_str}_{self.id}"
            return f"{file_name}:/{name}"

    def clean_side_effects(self) -> None:
        self.kernel.clean_side_effects(self)

    @abstractmethod
    def scan_range(self) -> Union[Iterable, Generator]:
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

    def run_subtask(self, subtask):
        subtask.parent = self.id
        subtask._runtime.sub_index = self._runtime.step
        self.kernel.submit(subtask)
        self.kernel.join(subtask)
        ret = subtask.result()
        self._runtime.data.append(ret['data'])
        return ret

    def standard_result(self):
        from waveforms.backends.quark.executable import assymblyData

        try:
            i = len(self._runtime.data)
            a = self.kernel.fetch(self, i)
            for step, (raw_data,
                       dataMap) in enumerate(zip(a,
                                                 self._runtime.dataMaps[i:]),
                                             start=i):
                result = assymblyData(raw_data, dataMap, self.signal)
                self._runtime.data.append(result['data'])
                self._runtime.result['states'].append(result.get(
                    'state', None))
                self._runtime.result['counts'].append(result.get(
                    'count', None))
                self._runtime.result['diags'].append(result.get('diag', None))
                if self._runtime.record is not None:
                    cbits = result['data'].shape[-1]
                    if 'cbits' in self._runtime.record.dims and 'cbits' not in self._runtime.record.coords:
                        self._runtime.record.coords['cbits'] = np.arange(cbits)
                    # self._runtime.record.append(
                    #     [(self._runtime.result['index'][step], )],
                    #     [(result['data'], result.get('state', None))])
        except:
            pass

        return {
            'calibration_level': self.calibration_level,
            'index': self._runtime.result['index'],
            'data': np.asarray(self._runtime.data),
            'states': np.asarray(self._runtime.result['states']),
            'counts': self._runtime.result['counts'],
            'diags': np.asarray(self._runtime.result['diags'])
        }

    def result(self):
        return self.standard_result()


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

    def check(self):
        # never pass
        return -1

    def analyze(self, data) -> CalibrationResult:
        # always pass
        return CalibrationResult(suggested_calibration_level=100)

    def scan_range(self) -> Union[Iterable, Generator]:
        return []

    def main(self):
        for key in self.keys:
            self.set(key, input(f"{key} = "), cache=False)

    def result(self):
        return None


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
        id(task._runtime): TaskRuntime(),
        id(task.kernel): None,
        id(task.parent): None,
        id(task.container): None,
        id(task.id): None,
        id(task.db): None,
    }
    return copy.deepcopy(task, memo)
