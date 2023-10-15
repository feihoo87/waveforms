"""Multiple architecture support"""
from waveforms.qlisp.arch import get_arch, register_arch

from .base import (COMMAND, FREE, PUSH, READ, SYNC, TRIG, WRITE, Architecture,
                   general_architecture)
import warnings

warnings.warn(
    'waveforms.quantum.circuit.qlisp is deprecated, please use waveforms.qlisp instead',
    DeprecationWarning,
    stacklevel=2,
)