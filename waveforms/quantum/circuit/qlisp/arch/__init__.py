"""Multiple architecture support"""
from .baqis import baqisArchitecture
from .base import COMMAND, FREE, PUSH, READ, SYNC, TRIG, WRITE, Architecture

__regested_architectures = {}


def get_arch(name: str = 'baqis') -> Architecture:
    return __regested_architectures[name]


def register_arch(arch: Architecture):
    __regested_architectures[arch.name] = arch


register_arch(baqisArchitecture)
