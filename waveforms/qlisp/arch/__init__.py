"""Multiple architecture support"""
from typing import Callable, NamedTuple, Optional

from waveforms.dicttree import flattenDict

from ..base import QLispCode
from ..commands import CommandList, DataMap, RawData, Result


class Architecture(NamedTuple):
    name: str
    description: str
    assembly_code: Callable[[QLispCode, Optional[dict]], tuple[CommandList,
                                                               DataMap]]
    assembly_data: Callable[[RawData, DataMap], Result]


general_architecture = Architecture(
    name='general',
    description='General architecture',
    assembly_code=lambda code, context: (
        [],
        {
            'arch': 'general'
        },
    ),
    assembly_data=lambda data, data_map: flattenDict(data),
)

__regested_architectures = {}


def get_arch(name: str = 'general') -> Architecture:
    return __regested_architectures[name]


def register_arch(arch: Architecture):
    __regested_architectures[arch.name] = arch


register_arch(general_architecture)

__all__ = ['Architecture', 'get_arch', 'register_arch']
