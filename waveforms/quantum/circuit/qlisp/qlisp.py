from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional, Union

from waveforms.waveform import Waveform, zero

from .config import Config, getConfig


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
    signal: str
    params: dict
    hardware: dict


@dataclass
class Context():
    cfg: Config = field(default_factory=getConfig)
    scopes: list[dict[str, Any]] = field(default_factory=lambda: [dict()])
    qlisp: list = field(default_factory=list)
    time: dict[str,
               float] = field(default_factory=lambda: defaultdict(lambda: 0))
    addressTable: dict = field(default_factory=dict)
    waveforms: dict[str, Waveform] = field(
        default_factory=lambda: defaultdict(zero))
    raw_waveforms: dict[tuple[str, ...], Waveform] = field(
        default_factory=lambda: defaultdict(zero))
    measures: dict[int, list[MeasurementTask]] = field(
        default_factory=lambda: defaultdict(list))
    phases: dict[str,
                 float] = field(default_factory=lambda: defaultdict(lambda: 0))
    biases: dict[str,
                 float] = field(default_factory=lambda: defaultdict(lambda: 0))
    end: float = 0

    @property
    def channel(self):
        return self.raw_waveforms

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

    def _getAWGChannel(self, name, *qubits) -> Union[str, dict]:
        def _getSharedCoupler(qubits):
            s = set(qubits[0]['couplers'])
            for qubit in qubits[1:]:
                s = s & set(qubit['couplers'])
            return s

        qubits = [self.cfg.getQubit(q) for q in qubits]

        if name.startswith('readoutLine.'):
            #name = name.removeprefix('readoutLine.')
            name = name[len('readoutLine.'):]
            rl = self.cfg.getReadoutLine(qubits[0]['readoutLine'])
            chInfo = rl.query('channels.' + name)
        elif name.startswith('coupler.'):
            #name = name.removeprefix('coupler.')
            name = name[len('coupler.'):]
            c = _getSharedCoupler(qubits).pop()
            c = self.cfg.getCoupler(c)
            chInfo = c.query('channels.' + name)
        else:
            chInfo = qubits[0].query('channels.' + name)
        return chInfo

    def _getReadoutADLO(self, qubit):
        rl = self.cfg.getReadoutLine(self.cfg.getQubit(qubit).readoutLine)
        lo = self.cfg.getChannel(rl.channels.AD.LO).status.frequency
        return lo

    def _getADChannel(self, qubit) -> Union[str, dict]:
        rl = self.cfg.getQubit(qubit).readoutLine
        rl = self.cfg.getReadoutLine(rl)
        return rl.channels.AD

    def _getLOFrequencyOfChannel(self, chInfo) -> float:
        lo = self.cfg.getChannel(chInfo['LO'])
        lofreq = lo.status.frequency
        return lofreq

    def _getADChannelDetails(self, chInfo) -> dict:
        def _getADSampleRate(ctx, channel):
            return ctx.cfg.getChannel(channel).params.sampleRate

        hardware = {'channel': {}, 'params': {}}
        if isinstance(chInfo, dict):
            if 'LO' in chInfo:
                loFreq = self._getLOFrequencyOfChannel(chInfo)
                hardware['channel']['LO'] = chInfo['LO']
                hardware['params']['LOFrequency'] = loFreq

            hardware['params']['sampleRate'] = {}
            for ch in ['I', 'Q', 'IQ', 'Ref']:
                if ch in chInfo:
                    hardware['channel'][ch] = chInfo[ch]
                    sampleRate = _getADSampleRate(self, chInfo[ch])
                    hardware['params']['sampleRate'][ch] = sampleRate
        elif isinstance(chInfo, str):
            hardware['channel'] = chInfo

        return hardware

    def _getGateConfig(self, name, *qubits) -> dict:
        try:
            gate = self.cfg.getGate(name, *qubits)
        except:
            return {'type': 'default', 'params': {}}
        params = gate['params']
        type = gate.get('type', 'default')
        return {'type': type, 'params': params}

    def _getAllQubitLabels(self) -> list[str]:
        return sorted(self.cfg['chip']['qubits'].keys(),
                      key=lambda s: int(s[1:]))


@dataclass
class QLispCode():
    cfg: Config = field(repr=False)
    qlisp: list = field(repr=True)
    waveforms: dict[str, Waveform] = field(repr=True)
    measures: dict[int, list[MeasurementTask]] = field(repr=True)
    end: float = field(default=0, repr=True)


__context_factory = Context


def set_context_factory(factory):
    global __context_factory
    __context_factory = factory


def create_context(ctx: Optional[Context] = None, **kw) -> Context:
    if ctx is None:
        return __context_factory(**kw)
    else:
        if 'cfg' not in kw:
            kw['cfg'] = ctx.cfg
        sub_ctx = __context_factory(**kw)
        sub_ctx.time.update(ctx.time)
        sub_ctx.phases.update(ctx.phases)
        sub_ctx.biases.update(ctx.biases)

        return sub_ctx
