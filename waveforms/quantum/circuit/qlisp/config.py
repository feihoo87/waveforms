import pickle
import warnings
from itertools import chain, permutations
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from waveforms.baseconfig import BaseConfig, Trait


class Config(BaseConfig):
    def __init__(self, path: Optional[Union[str, Path]] = None):
        super().__init__(path)
        if 'station' not in self:
            self.update({
                'station': {
                    'name': "",
                    'instruments': {},
                    'channels': {},
                    'params': {},
                    'status': {},
                    'calibrations': {}
                },
                'chip': {
                    'qubits': {},
                    'couplers': {},
                    'readoutLines': {},
                    'params': {},
                    'status': {},
                    'calibrations': {
                        'fluxCrosstalk': [],
                        'rfCrosstalk': []
                    }
                },
                'gates': {
                    'Measure': {},
                    'Reset': {},
                    'rfUnitary': {},
                    'CZ': {},
                    'iSWAP': {},
                    'CR': {
                        '__order_senstive__': True
                    }
                }
            })

    def newObject(self,
                  section,
                  key,
                  template=None,
                  properties=('params', 'status', 'calibrations')):
        try:
            d = self.query('.'.join([section, key]))
            return d
        except:
            pass
        d = self
        for k in chain(section.split('.'), key.split('.')):
            if k not in d:
                d[k] = {}
            d = d[k]
        if template is not None:
            d.update(template)
        for k in properties:
            if k not in d:
                d[k] = {}
        return d

    def newInstrument(self, name, driver, address, **params):
        d = self.newObject('station.instruments', name, {
            'dirver': driver,
            'address': address,
            'params': params
        })
        return d

    def newChannel(self, name, instrument='', channel=1, **params):
        try:
            self.getInstrument(instrument)
        except:
            warnings.warn(f'instrument {instrument} not exist.', )
        d = self.newObject(
            'station.channels', name, {
                'port': {
                    'instrument': instrument,
                    'channel': channel,
                },
                'params': params
            })
        return d

    def newQubit(self, name, couplers=[], readoutLine='', **params):
        for coupler in couplers:
            try:
                c = self.getCoupler(coupler)
            except:
                c = self.newCoupler(coupler, [])
            c['qubits'].append(name)
        try:
            rl = self.getReadoutLine(readoutLine)
        except:
            rl = self.newReadoutLine(readoutLine, [])
        rl['qubits'].append(name)

        d = self.newObject(
            'chip.qubits', name, {
                'params': params,
                'couplers': couplers,
                'readoutLine': readoutLine,
                'channels': {}
            })
        return d

    def newCoupler(self, name, qubits, **params):
        for qubit in qubits:
            try:
                q = self.getQubit(qubit)
            except:
                q = self.newQubit(qubit)
            q['couplers'].append(name)
        d = self.newObject('chip.couplers', name, {
            'params': params,
            'qubits': qubits,
            'channels': {}
        })
        return d

    def newReadoutLine(self, name, qubits, **params):
        for qubit in qubits:
            try:
                q = self.getQubit(qubit)
            except:
                q = self.newQubit(qubit)
            q['readoutLine'] = name
        d = self.newObject('chip.readoutLines', name, {
            'params': params,
            'qubits': qubits,
            'channels': {}
        })
        return d

    def newGate(self, name, *qubits, type='default', **params):
        for qubit in qubits:
            try:
                self.getQubit(qubit)
            except:
                self.newQubit(qubit)
        key = ','.join(qubits)
        template = {'type': type, 'params': params}
        g = self.newObject(f'gates.{name}',
                           key,
                           template,
                           properties=('type', 'params'))
        return g

    def getObject(self, q, cls=None):
        if cls is None:
            cls = 'ConfigObject'
        return self.query(q + ':' + cls)

    def getQubit(self, name):
        return self.getObject(f"chip.qubits.{name}", cls='Qubit')

    def getCoupler(self, name):
        return self.getObject(f"chip.couplers.{name}")

    def getReadoutLine(self, name):
        return self.getObject(f"chip.readoutLines.{name}")

    def getChannel(self, name):
        return self.getObject(f"station.channels.{name}")

    def getInstrument(self, name):
        return self.getObject(f"station.instruments.{name}")

    def getGate(self, name, *qubits):
        if name not in self['gates']:
            raise KeyError(f'"{name}" gate not defined.')
        if name == 'rfUnitary':
            return self.getObject(f"gates.{name}.{','.join(qubits)}",
                                  cls='rfUnitary')
        elif name == 'Measure':
            return self.getObject(f"gates.{name}.{','.join(qubits)}",
                                  cls='Measure')
        if ('__order_senstive__' in self['gates'][name]
                and self['gates'][name]['__order_senstive__']):
            return self.getObject(f"gates.{name}.{','.join(qubits)}",
                                  cls='Gate')
        else:
            for qlist in permutations(qubits):
                try:
                    return self.getObject(f"gates.{name}.{','.join(qlist)}",
                                          cls='Gate')
                except:
                    pass
            else:
                raise KeyError(f'Could not find "{name}" gate for {qubits}')


class ConfigObject(Trait):
    def setParams(self, **params):
        self['params'].update(params)

    def setStatus(self, **status):
        self['status'].update(status)

    def _channelDetails(self):
        if 'channels' not in self:
            return {}

        def _getChannels(dct):
            ret = {}
            for c, v in dct.items():
                if isinstance(v, str):
                    try:
                        ret[c] = self._cfg_.getChannel(v)
                    except:
                        ret[c] = v
                elif isinstance(v, dict):
                    ret[c] = _getChannels(v)
                else:
                    ret[c] = v
            return ret

        return _getChannels(self.channels)

    def _getSectionDetails(self, section, skip):
        if section not in self:
            return None

        getMethod = {
            'qubits': self._cfg_.getQubit,
            'couplers': self._cfg_.getCoupler,
            'readoutLine': self._cfg_.getReadoutLine
        }[section]

        if isinstance(self[section], str):
            return getMethod(self[section]).details(skip + (section, ))
        else:
            return {
                k: getMethod(k).details(skip + (section, ))
                for k in self[section]
            }

    def details(self, skip=('channels', 'qubits', 'couplers', 'readoutLine')):
        ret = {}
        ret.update(self)
        if 'channels' not in skip:
            ret['channels'] = self._channelDetails()
        if 'qubits' not in skip:
            ret['qubits'] = self._getSectionDetails('qubits', skip)
        if 'couplers' not in skip:
            ret['couplers'] = self._getSectionDetails('couplers', skip)
        if 'readoutLine' not in skip:
            ret['readoutLine'] = self._getSectionDetails('readoutLine', skip)
        return BaseConfig.fromdict(ret)


class Qubit(ConfigObject):
    def details(self):
        return super().details(skip=('qubits', ))


class Gate(ConfigObject):
    def details(self):
        ret = {}
        ret.update(self)
        ret['qubits'] = {
            q: self._cfg_.getQubit(q).details()
            for q in self.qubits
        }
        return BaseConfig.fromdict(ret)

    @property
    def name(self):
        return self._key_.split('.')[1]

    @property
    def qubits(self):
        return self._key_.split('.')[-1].split(',')

    def reset(self):
        qubit = self._cfg_.getQubit(self.qubits[0])
        if self.name == 'rfUnitary':
            self.setParams(frequency=qubit.params()['f01'])
        elif self.name == 'measure':
            self.setParams(frequency=qubit.params()['fr'])
        elif self.name == 'iSWAP':
            qubit2 = self._cfg_.getQubit(self.qubits[1])
            self.setParams(frequency=abs(qubit.params()['f01'] -
                                         qubit2.params()['f01']))


class rfUnitary(Gate):
    def setAmp(self, theta, amp):
        t = theta / np.pi
        dct = {k: v for k, v in zip(*self['params']['amp'])}
        dct[t] = amp
        self['params']['amp'] = [
            sorted(dct.keys()), [dct[k] for k in sorted(dct.keys())]
        ]

    def setDuration(self, theta, duration):
        t = theta / np.pi
        dct = {k: v for k, v in zip(*self['params']['duration'])}
        dct[t] = duration
        self['params']['duration'] = [
            sorted(dct.keys()), [dct[k] for k in sorted(dct.keys())]
        ]

    def setPhase(self, phi, phase):
        p = phi / np.pi
        dct = {k: v for k, v in zip(*self['params']['phase'])}
        dct[p] = phase / np.pi
        self['params']['phase'] = [
            sorted(dct.keys()), [dct[k] for k in sorted(dct.keys())]
        ]

    def amp(self, theta):
        return np.interp(theta / np.pi, *self['params']['amp'])

    def duration(self, theta):
        return np.interp(theta / np.pi, *self['params']['duration'])

    def phase(self, phi):
        return np.pi * np.interp(phi / np.pi, *self['params']['phase'])

    def reset(self):
        qubit = self._cfg_.getQubit(self.qubits[0])
        self.setParams(frequency=qubit.params.f01)
        self['params']['amp'] = [[0, 1], [self.amp(0), self.amp(np.pi)]]
        self['params']['duration'] = [[0, 1],
                                      [self.duration(0),
                                       self.duration(np.pi)]]
        self['params']['phase'] = [[-1, 1], [-1, 1]]

    def shape(self, theta, phi=0):
        return {
            'shape': self['params'].get('shape', 'Gaussian'),
            'amp': self.amp(theta),
            'duration': self.duration(theta),
            'phase': self.phase(phi),
            'frequency': self['params']['frequency'],
            'DRAGScaling': self['params'].get('DRAGScaling', 0)
        }


class Measure(Gate):
    @property
    def wfile(self):
        return self._cfg__._path_.parent / 'Measure' / self.qubits[0]+'.pic'

    def setW(self, w):
        with open(self.wfile, 'wb') as f:
            pickle.dump(w, f)

    def W(self):
        with open(self.wfile, 'rb') as f:
            return pickle.load(f)


__configFilePath = None


def setConfig(path: Union[Path, str]) -> None:
    global __configFilePath
    __configFilePath = Path(path)


def getConfig() -> Config:
    if __configFilePath is None:
        raise FileNotFoundError('setConfig(path) must be run first.')
    return Config(__configFilePath)


__all__ = ['Config', 'getConfig', 'setConfig']
