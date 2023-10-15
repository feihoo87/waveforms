import warnings

from .arch import (COMMAND, FREE, PUSH, READ, SYNC, TRIG, WRITE, Architecture,
                   get_arch, register_arch)
from .arch.base import CommandList, DataMap, RawData, Result
from .compiler import compile
from .library import Library, libraries
from .qlisp import QLispCode, QLispError
from .stdlib import std

warnings.warn(
    'waveforms.quantum.circuit.qlisp is deprecated, please use waveforms.qlisp instead',
    DeprecationWarning,
    stacklevel=2,
)