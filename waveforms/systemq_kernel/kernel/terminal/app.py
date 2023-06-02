from abc import ABC, abstractmethod
import numpy as np
from waveforms.scan_iter import scan_iters


class App(ABC):

    def __init__(self):
        self.shots = 1024
        self._step_setting = {}
        self._measures = {}
        self._circuit = []
        self._task_info = {'shape': (), 'steps': []}
        self.libs = ['std']

    def set(self, key, value):
        self._step_setting[key] = value

    def get(self, key):
        pass

    def exec(self, circuit, lib=None, skip_compile: bool = False):
        if lib is None:
            lib = self.libs
        self._circuit = circuit, lib, skip_compile

    def measure(self, key, label=None):
        if label is None:
            label = key
        self._measures[label] = key

    def scan(self):

        def extend_shape(shape, pos):
            ret = []
            for a, b in zip(shape, pos):
                ret.append(max(a, b))
            return ret

        shape = None

        for step in scan_iters(**self.scan_range()):
            if shape is None:
                shape = step.pos
            else:
                shape = extend_shape(shape, step.pos)
            yield step
            self._task_info['steps'].append({
                'pos':
                step.pos,
                'kwds': {
                    k: v
                    for k, v in step.kwds.items()
                    if not k.startswith('__tmp_') and k != 'circuit'
                },
                'setting':
                self._step_setting.copy(),
                'circuit':
                self._circuit[0],
                'measure':
                self._measures.copy(),
                'lib':
                self._circuit[1]
            })
            self._step_setting.clear()
            self._measures.clear()
        self._task_info['shape'] = tuple([i + 1 for i in shape])

    @abstractmethod
    def scan_range(self):
        pass

    @abstractmethod
    def main(self):
        pass

    def task(self):
        self.main()
        return self._task_info
