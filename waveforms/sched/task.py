import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


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


@dataclass
class CalibrationResult():
    success: bool = False
    bad_data: bool = True
    in_spec: bool = False
    parameters: dict = field(default_factory=dict)


def _is_feedable(key):
    return re.match(r'[QCM]\d+\.(setting|waveform)\..+', key)


class Task(ABC):
    def __init__(self, signal='count', shots=1024, calibration_level=0):
        self.parent = None
        self.id = None
        self.kernel = None
        self.db = None
        self.signal = signal
        self.shots = shots
        self.calibration_level = calibration_level
        self._runtime = TaskRuntime()

    def __del__(self):
        try:
            self.kernel.excuter.free(self.id)
        except:
            pass

    @property
    def app_name(self):
        return f"{self.__class__.__module__}.{self.__class__.__name__}"

    def is_children_of(self, task):
        return self.parent is not None and self.parent == task.id

    def get_parent(self):
        if self.parent is not None:
            return self.kernel.get_task_by_id(self.parent)
        return None

    def set(self, key: str, value: Any, cache: bool = False):
        if not cache and _is_feedable(key):
            self._runtime.cmds.append(WRITE(key, value))
        self.kernel.get_config().update(key, value, cache=cache)

    def get(self, key: str):
        """
        return the value of the key in the kernel config
        """
        return self.kernel.query(key)

    def exec(self, circuit, lib=None, cfg=None, compile_once=False):
        self.kernel._exec(self,
                          circuit,
                          lib=lib,
                          cfg=cfg,
                          signal=self.signal,
                          compile_once=compile_once)

    def measure(self, keys, labels=None):
        self.kernel._measure(self, keys, labels)

    def trig(self):
        cmds = self.get('station.triggercmds')
        for cmd in cmds:
            self._runtime.cmds.append(TRIG(cmd))

    def scan(self):
        yield from self.kernel.scan(self)

    def feedback(self, obj):
        self.kernel.feedback(self, obj)

    def get_feedback(self):
        return self.kernel.get_feedback(self)

    def data_path(self):
        if self.parent:
            name = '/'.join([
                self.kernel.get_task_by_id(self.parent).data_path(),
                'sub_data',
                f"{self.__class__.__name__}_{self._runtime.sub_index}"
            ])
            return name
        else:
            file_name = self.get('station.sample')
            file_name = 'Test'
            time_str = time.strftime(
                '%Y-%m-%d-%H-%M-%S',
                time.localtime(self._runtime.started_time))
            name = f"{self.__class__.__name__}_{time_str}_{self.id}"
            return f"{file_name}:/{name}"

    def clean_side_effects(self):
        self.kernel.clean_side_effects(self)

    @abstractmethod
    def scan_range(self):
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
        return the last sucessful calibration time,
        otherwise return -1
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
            for raw_data, dataMap in zip(a, self._runtime.dataMaps[i:]):
                result = assymblyData(raw_data, dataMap, self.signal)
                self._runtime.data.append(result['data'])
                self._runtime.result['states'].append(result.get(
                    'state', None))
                self._runtime.result['counts'].append(result.get(
                    'count', None))
                self._runtime.result['diags'].append(result.get('diag', None))
            return {
                'index': self._runtime.result['index'],
                'data': self._runtime.data,
                'states': self._runtime.result['states'],
                'counts': self._runtime.result['counts'],
                'diags': self._runtime.result['diags']
            }
        except:
            return {
                'index': [],
                'data': [],
                'states': [],
                'counts': [],
                'diags': []
            }

    def result(self):
        return self.standard_result()


class App(Task):
    def plot(self, fig, result):
        raise NotImplementedError()
