"""Multiple architecture support"""
from .base import (COMMAND, FREE, PUSH, READ, SYNC, TRIG, WRITE, Architecture,
                   general_architecture)

__regested_architectures = {}


def get_arch(name: str = 'baqis') -> Architecture:
    return __regested_architectures[name]


def register_arch(arch: Architecture):
    __regested_architectures[arch.name] = arch


register_arch(general_architecture)
