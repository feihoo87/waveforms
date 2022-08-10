from enum import Flag, auto
from pathlib import Path

from ..dicttree import NOTSET
from ..quantum import stdlib
from ..quantum.circuit.qlisp import QLispError, get_arch
from ..quantum.circuit.qlisp.arch import register_arch
from ..quantum.circuit.qlisp.arch.base import (COMMAND, FREE, PUSH, READ, SYNC,
                                               TRIG, WRITE, Architecture,
                                               CommandList, DataMap,
                                               MeasurementTask, QLispCode,
                                               RawData, Result)
from ..quantum.circuit.qlisp.config import Config, ConfigProxy
from ..quantum.circuit.qlisp.library import Library, Parameter, libraries
from ..quantum.circuit.qlisp.qlisp import (ABCCompileConfigMixin, ADChannel,
                                           AWGChannel, GateConfig,
                                           MultADChannel, MultAWGChannel,
                                           Signal)

stdlib.qasmLib = {
    'qelib1.inc': Path(__file__).parent / 'qasm' / 'libs' / 'qelib1.inc'
}


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


__all__ = [
    'COMMAND', 'FREE', 'NOTSET', 'PUSH', 'READ', 'SYNC', 'TRIG', 'WRITE',
    'ABCCompileConfigMixin', 'ADChannel', 'Architecture', 'AWGChannel',
    'CommandList', 'Config', 'ConfigProxy', 'DataMap', 'GateConfig', 'Library',
    'MeasurementTask', 'MultADChannel', 'MultAWGChannel', 'Parameter',
    'QLispCode', 'QLispError', 'RawData', 'Result', 'Signal', 'get_arch',
    'libraries', 'register_arch', 'stdlib'
]
