from __future__ import annotations

import copy
import functools
import importlib
import logging
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import (Any, Generator, Iterable, Literal, Optional, Sequence,
                    Type, Union)

import blinker
import numpy as np
from waveforms.quantum.circuit.qlisp.config import Config, ConfigProxy
from waveforms.quantum.circuit.qlisp.library import Library
from waveforms.storage.crud import tag
from waveforms.storage.models import Record, Report

from .base import TRIG, WRITE
from .base import Task as BaseTask
from .base import TaskRuntime


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
                 calibration_level: int = 0):
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

    # def set_frame(self, dims: list[tuple[str, str]],
    #                   vars: list[tuple[str, str]],
    #                   coords: dict[str, Sequence]) -> None:
    #     '''
    #     set_frame(dims=['shots', 'cbits'], vars=['IQ', 'state'], coords={'shots': [0, 1, 2], 'cbits': [0, 1, 2]})
    #     '''
    #     pass

    # def set_record(self, save_raw: bool = False, **kwargs) -> None:
    #     """
    #     Define the record for the task.

    #     Args:
    #         save_raw: whether to store raw data or not
    #         kwargs: the data to be stored

    #     Example:
    #         def_record(save_raw=True,
    #                    sigma_z={'mean': ['shots']},
    #                    S_z={'mean': ['shots'], 'sum': ['cbits']},
    #                    S_p={'std': ['shots']},)
    #     """
    #     pass

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
        return self.kernel.query(key)

    def exec(self,
             circuit: QLisp,
             lib: Optional[Library] = None,
             cfg: Union[Config, ConfigProxy, None] = None,
             compile_once: bool = False):

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
            self.runtime.cmds.append(TRIG(cmd))

    def scan(self) -> Generator:
        self.set_record(dims=[('index', ''), ('shots', ''), ('cbits', '')],
                        vars=[('data', ''), ('state', '')],
                        coords={
                            'shots': np.arange(self.shots),
                        })
        yield from self.kernel.scan(self)

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
            name = f"{self.name}_{time_str}_{self.id}"
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

    def _fetch_result(self):
        from waveforms.backends.quark.executable import assymblyData

        i = len(self.runtime.result['data'])
        additional = self.kernel.fetch(self, i)
        if isinstance(additional, str):
            additional = []
        for step, (raw_data,
                   dataMap) in enumerate(zip(additional,
                                             self.runtime.prog.data_maps[i:]),
                                         start=i):
            result = assymblyData(raw_data, dataMap, self.signal)
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
        return []

    def main(self):
        for key in self.keys:
            self.set(key, input(f"{key} = "), cache=False)

    def result(self):
        return {'data': []}


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
