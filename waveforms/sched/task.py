import time
from dataclasses import dataclass, field
from typing import Any
from abc import ABC, abstractmethod


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


class Task(ABC):
    def __init__(self, signal='count', calibration_level=0):
        self.parent = None
        self.id = None
        self.kernel = None
        self.signal = signal
        self.calibration_level = calibration_level
        self._runtime = TaskRuntime()

    def __del__(self):
        try:
            self.kernel.excuter.free(self.id)
        except:
            pass

    def is_children_of(self, task):
        return self.parent is not None and self.parent == task.id

    def get_parent(self):
        if self.parent is not None:
            return self.kernel.get_task_by_id(self.parent)
        return None

    def set(self, key, value, cache=True):
        self._runtime.cmds.append((key, value))
        self.kernel.get_config().update(key, value, cache=cache)

    def get(self, key):
        return self.kernel.query(key)

    def exec(self, circuit, lib=None, cfg=None):
        self.kernel._exec(self, circuit, lib=lib, cfg=cfg, signal=self.signal)

    def measure(self, keys, labels=None):
        self.kernel._measure(self, keys, labels)

    def trigger(self):
        pass

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
            file_name = self.get('sampleID')
            time_str = time.strftime('%Y-%m-%d-%H-%M-%S',
                                     self._runtime.started_time)
            name = f"{self.__class__.__name__}_{time_str}"
            return f"{file_name}:/{name}"

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

    def check_state(self) -> bool:
        """
        return True only if the task is finished successfully
        in ttl and the result is valid, otherwise return False.
        """
        raise NotImplementedError()

    def analyze(self, data) -> CalibrationResult:
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
            a = self.kernel.result(self, i)
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
