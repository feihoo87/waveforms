from waveforms.dicttree import NOTSET
from waveforms.qlisp import (COMMAND, FREE, PUSH, READ, SYNC, TRIG, WRITE,
                             ABCCompileConfigMixin, ADChannel, Architecture,
                             AWGChannel, CommandList, Config, ConfigProxy,
                             DataMap, GateConfig, Library, MeasurementTask,
                             MultADChannel, MultAWGChannel, QLispCode, RawData,
                             Result, get_arch, libraries, register_arch,
                             stdlib)

__all__ = [
    'COMMAND', 'FREE', 'NOTSET', 'PUSH', 'READ', 'SYNC', 'TRIG', 'WRITE',
    'ABCCompileConfigMixin', 'ADChannel', 'Architecture', 'AWGChannel',
    'CommandList', 'Config', 'ConfigProxy', 'DataMap', 'GateConfig', 'Library',
    'MeasurementTask', 'MultADChannel', 'MultAWGChannel', 'QLispCode',
    'RawData', 'Result', 'Signal', 'get_arch', 'libraries', 'register_arch',
    'stdlib'
]


from enum import Flag, auto


class Signal(Flag):
    trace = auto()
    iq = auto()
    state = auto()

    _avg_trace = auto()
    _avg_iq = auto()
    _avg_state = auto()
    _count = auto()

    trace_avg = trace | _avg_trace

    iq_avg = iq | _avg_iq

    population = state | _avg_state
    count = state | _count
    diag = state | _count | _avg_state

    _remote = auto()

    remote_trace_avg = trace_avg | _remote
    remote_iq_avg = iq_avg | _remote
    remote_state = state | _remote
    remote_population = population | _remote
    remote_count = count | _remote