from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from waveforms.waveform import Waveform, zero

from .config import Config, getConfig


def gateName(st):
    if isinstance(st[0], str):
        return st[0]
    else:
        return st[0][0]


class QLispError(Exception):
    pass


class MeasurementTask(NamedTuple):
    qubit: str
    cbit: int
    time: float
    signal: str
    params: dict
    hardware: dict


class _ChannelGetter():
    __slots__ = ('ctx')

    def __init__(self, ctx):
        self.ctx = ctx

    def __getitem__(self, key):
        return self.ctx.raw_waveforms.__getitem__(key)

    def __setitem__(self, key, wav):
        self.ctx.raw_waveforms.__setitem__(key, wav)


@dataclass
class Context():
    cfg: Config = field(default_factory=getConfig)
    scopes: list[dict[str, Any]] = field(default_factory=lambda: [dict()])
    qlisp: list = field(default_factory=list)
    time: dict[str,
               float] = field(default_factory=lambda: defaultdict(lambda: 0))
    waveforms: dict[str, Waveform] = field(
        default_factory=lambda: defaultdict(zero))
    raw_waveforms: dict[tuple[str, ...], Waveform] = field(
        default_factory=lambda: defaultdict(zero))
    measures: dict[int, list[MeasurementTask]] = field(
        default_factory=lambda: defaultdict(list))
    phases: dict[str,
                 float] = field(default_factory=lambda: defaultdict(lambda: 0))
    end: float = 0

    @property
    def channel(self):
        return _ChannelGetter(self)

    @property
    def vars(self):
        return self.scopes[-1]


@dataclass
class QLispCode():
    cfg: Config = field(repr=False)
    qlisp: list = field(repr=True)
    waveforms: dict[str, Waveform] = field(repr=True)
    measures: dict[int, list[MeasurementTask]] = field(repr=True)
    end: float = field(default=0, repr=True)
