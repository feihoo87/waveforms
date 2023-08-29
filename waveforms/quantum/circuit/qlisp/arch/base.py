from typing import Any, Callable, NamedTuple

from waveforms.dicttree import flattenDict
from waveforms.qlisp.commands import (COMMAND, FREE, PUSH, READ, SYNC, TRIG,
                                      WRITE, CommandList, DataMap, RawData,
                                      Result)

from waveforms.qlisp import Capture, QLispCode

from waveforms.qlisp.arch import Architecture, general_architecture
