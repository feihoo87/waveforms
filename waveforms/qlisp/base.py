from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Any, Literal, NamedTuple, Optional, Union

from ..waveform import Waveform, zero


class Signal(Flag):
    trace = auto()
    iq = auto()
    state = auto()

    _avg_trace = auto()
    _avg_iq = auto()
    _avg_state = auto()
    _count = auto()
    _remote = auto()

    trace_avg = trace | _avg_trace

    iq_avg = iq | _avg_iq

    population = state | _avg_state
    count = state | _count
    diag = state | _count | _avg_state

    remote_trace_avg = trace_avg | _remote
    remote_iq_avg = iq_avg | _remote
    remote_state = state | _remote
    remote_population = population | _remote
    remote_count = count | _remote


def gateName(st):
    if isinstance(st[0], str):
        return st[0]
    else:
        return st[0][0]


class QLispError(SyntaxError):
    pass


class MeasurementTask(NamedTuple):
    qubit: str
    cbit: int
    time: float
    signal: Signal
    params: dict
    hardware: Union[ADChannel, MultADChannel] = None
    shift: float = 0


class AWGChannel(NamedTuple):
    name: str
    sampleRate: float
    size: int = -1
    amplitude: Optional[float] = None
    offset: Optional[float] = None
    commandAddresses: tuple = ()


class MultAWGChannel(NamedTuple):
    I: Optional[AWGChannel] = None
    Q: Optional[AWGChannel] = None
    LO: Optional[str] = None
    lo_freq: float = -1
    lo_power: Optional[float] = None


class ADChannel(NamedTuple):
    name: str
    sampleRate: float = 1e9
    trigger: str = ''
    triggerDelay: float = 0
    triggerClockCycle: float = 8e-9
    commandAddresses: tuple = ()


class MultADChannel(NamedTuple):
    I: Optional[ADChannel] = None
    Q: Optional[ADChannel] = None
    IQ: Optional[ADChannel] = None
    Ref: Optional[ADChannel] = None
    LO: Optional[str] = None
    lo_freq: float = -1
    lo_power: Optional[float] = None


class GateConfig(NamedTuple):
    name: str
    qubits: tuple
    type: str = 'default'
    params: dict = {}


class ABCCompileConfigMixin(ABC):
    """
    Mixin for configs that can be used by compiler.
    """

    @abstractmethod
    def _getAWGChannel(self, name,
                       *qubits) -> Union[AWGChannel, MultAWGChannel]:
        """
        Get AWG channel by name and qubits.
        """
        pass

    @abstractmethod
    def _getADChannel(self, qubit) -> Union[ADChannel, MultADChannel]:
        """
        Get AD channel by qubit.
        """
        pass

    @abstractmethod
    def _getGateConfig(self, name, *qubits) -> GateConfig:
        """
        Return the gate config for the given qubits.

        Args:
            name: Name of the gate.
            qubits: Qubits to which the gate is applied.
        
        Returns:
            GateConfig for the given qubits.
            if the gate is not found, return None.
        """
        pass

    @abstractmethod
    def _getAllQubitLabels(self) -> list[str]:
        """
        Return all qubit labels.
        """
        pass


__config_factory = None


def set_config_factory(factory):
    global __config_factory
    __config_factory = factory


def getConfig() -> ABCCompileConfigMixin:
    if __config_factory is None:
        raise FileNotFoundError(
            'set_config_factory(factory) must be run first.')
    else:
        return __config_factory()


@dataclass
class Context():
    cfg: ABCCompileConfigMixin = field(default_factory=getConfig)
    scopes: list[dict[str, Any]] = field(default_factory=lambda: [dict()])
    qlisp: list = field(default_factory=list)
    time: dict[str,
               float] = field(default_factory=lambda: defaultdict(lambda: 0))
    addressTable: dict = field(default_factory=dict)
    waveforms: dict[str, Waveform] = field(
        default_factory=lambda: defaultdict(zero))
    raw_waveforms: dict[tuple[str, ...], Waveform] = field(
        default_factory=lambda: defaultdict(zero))
    measures: dict[int, MeasurementTask] = field(default_factory=dict)
    phases_ext: dict[str, dict[Union[int, str], float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    biases: dict[str,
                 float] = field(default_factory=lambda: defaultdict(lambda: 0))
    end: float = 0

    @property
    def channel(self):
        return self.raw_waveforms

    @property
    def phases(self):

        class D():
            __slots__ = ('ctx', )

            def __init__(self, ctx):
                self.ctx = ctx

            def __getitem__(self, qubit):
                return self.ctx.phases_ext[qubit][1] - self.ctx.phases_ext[
                    qubit][0]

            def __setitem__(self, qubit, phase):
                self.ctx.phases_ext[qubit][
                    1] = phase + self.ctx.phases_ext[qubit][0]

        return D(self)

    @property
    def params(self):
        return self.scopes[-1]

    @property
    def vars(self):
        return self.scopes[-2]

    @property
    def globals(self):
        return self.scopes[0]

    def qubit(self, q):
        return self.addressTable[q]


@dataclass
class QLispCode():
    cfg: ABCCompileConfigMixin = field(repr=False)
    qlisp: list = field(repr=True)
    waveforms: dict[str, Waveform] = field(repr=True)
    measures: dict[int, list[MeasurementTask]] = field(repr=True)
    end: float = field(default=0, repr=True)
    signal: Signal = Signal.state
    shots: int = 1024
    arch: str = 'general'
    cbit_alias: dict[int, tuple[int, int]] = field(default_factory=dict)
    sub_code_count: int = 0


def set_context_factory(factory):
    warnings.warn('set_context_factory is deprecated', DeprecationWarning, 2)


def create_context(ctx: Optional[Context] = None, **kw) -> Context:
    if ctx is None:
        return Context(**kw)
    else:
        if 'cfg' not in kw:
            kw['cfg'] = ctx.cfg
        sub_ctx = Context(**kw)
        sub_ctx.time.update(ctx.time)
        #sub_ctx.phases.update(ctx.phases)
        sub_ctx.biases.update(ctx.biases)
        for k, v in ctx.phases_ext.items():
            sub_ctx.phases_ext[k].update(v)

        return sub_ctx
