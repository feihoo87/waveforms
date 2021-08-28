import copy
import re
import warnings
from itertools import permutations
from typing import Union

from waveforms.baseconfig import _flattenDictIter, _foldDict, _query, _update
from waveforms.namespace import DictDriver
from waveforms.quantum.circuit.qlisp.config import ConfigProxy
from waveforms.quantum.circuit.qlisp.qlisp import (ABCCompileConfigMixin,
                                                   ADChannel, AWGChannel,
                                                   MultADChannel,
                                                   MultAWGChannel)


def _getSharedCoupler(qubitsDict: dict) -> set[str]:
    s = set(qubitsDict[0]['couplers'])
    for qubit in qubitsDict[1:]:
        s = s & set(qubit['couplers'])
    return s


def _makeAWGChannelInfo(section: str, cfgDict: dict,
                        name: str) -> Union[str, dict]:
    ret = {}
    if name == 'RF':
        # if cfgDict['channel']['DDS'] is not None:
        #     return f"{section}.waveform.DDS"
        if cfgDict['channel']['I'] is not None:
            ret['I'] = f"{section}.waveform.RF.I"
        if cfgDict['channel']['Q'] is not None:
            ret['Q'] = f"{section}.waveform.RF.Q"
        ret['lofreq'] = cfgDict['setting']['LO']
        return ret
    elif name == 'AD.trigger':
        return f"{section}.waveform.TRIG"
    else:
        return f"{section}.waveform.{name}"


class CompileConfigMixin(ABCCompileConfigMixin):
    def _getAWGChannel(self, name,
                       *qubits) -> Union[AWGChannel, MultAWGChannel]:

        qubitsDict = [self.getQubit(q) for q in qubits]

        if name.startswith('readoutLine.'):
            name = name.removeprefix('readoutLine.')
            section = qubitsDict[0]['probe']
            cfgDict = self.getReadout(section)
        elif name.startswith('coupler.'):
            name = name.removeprefix('coupler.')
            section = _getSharedCoupler(qubitsDict).pop()
            cfgDict = self.getCoupler(section)
        else:
            section = qubits[0]
            cfgDict = qubitsDict[0]

        chInfo = _makeAWGChannelInfo(section, cfgDict, name)

        if isinstance(chInfo, str):
            return AWGChannel(chInfo, -1)
        else:
            info = {'lo_freq': chInfo['lofreq']}
            if 'I' in chInfo:
                info['I'] = AWGChannel(chInfo['I'], -1)
            if 'Q' in chInfo:
                info['Q'] = AWGChannel(chInfo['Q'], -1)
            return MultAWGChannel(**info)

    def _getADChannel(self, qubit) -> Union[str, dict]:
        rl = self.getQubit(qubit)['probe']
        rlDict = self.getReadout(rl)
        chInfo = {
            'IQ': rlDict['channel']['ADC'],
            'LO': rlDict['channel']['LO'],
            'TRIG': rlDict['channel']['TRIG'],
            'lofreq': rlDict['setting']['LO'],
            'trigger': f'{rl}.waveform.TRIG',
            'sampleRate': rlDict['adcsr'],
            'triggerDelay': rlDict['setting']['TRIGD'],
            'triggerSkew': rlDict.get('trigger_skew', 0),
        }

        hardware = {
            'channel': {},
            'params': {
                'triggerDelay': chInfo['triggerDelay'],
                'triggerSkew': chInfo['triggerSkew'],
            }
        }
        if isinstance(chInfo, dict):
            if 'lofreq' in chInfo:
                hardware['params']['LOFrequency'] = chInfo['lofreq']

            hardware['params']['sampleRate'] = {}
            for ch in ['I', 'Q', 'IQ', 'Ref', 'TRIG']:
                if ch in chInfo:
                    hardware['channel'][ch] = chInfo[ch]
                    sampleRate = chInfo['sampleRate']
                    hardware['params']['sampleRate'][ch] = sampleRate
        elif isinstance(chInfo, str):
            hardware['channel'] = chInfo

        return hardware

    def _getLOFrequencyOfChannel(self, chInfo) -> float:
        return chInfo['lofreq']

    def _getGateConfig(self, name, *qubits) -> dict:
        try:
            gate = self.getGate(name, *qubits)
            if not isinstance(gate, dict):
                return {'type': 'default', 'params': {}}
            params = gate['params']
            type = gate.get('type', 'default')
        except:
            type = 'default'
            params = {}
        return {'type': type, 'params': params}

    def _getAllQubitLabels(self) -> list[str]:
        return self.keys('Q*')


class QuarkConfig(ConfigProxy, CompileConfigMixin):
    def __init__(self, host='127.0.0.1'):
        self.host = host
        self._cache = {}
        self._history = {}
        self._cached_keys = set()
        self.connect()
        self.init_namespace()

    def connect(self):
        """Connect to the quark server."""
        from quark import connect
        if self.host is None:
            self.conn = None
        else:
            self.conn = connect('QuarkServer', host=self.host, verbose=False)

    def init_namespace(self):
        self.conn.create('dev', {})
        self.conn.create(
            'etc', {
                'checkpoint_path': './checkpoint.dat',
                'data_path': './data',
                'driver_paths': [],
            })
        self.conn.create('station', {'sample': 'Test', 'triggercmds': []})
        self.conn.create('tmp', {})
        self.conn.create('apps', {})
        self.conn.create('gate', {})
        self.conn.create('home', {})

    def newGate(self, name, *qubits):
        """Create a new gate."""
        qubits = '_'.join(qubits)
        self.conn.create(f"gate.{name}.{qubits}", {
            'type': 'default',
            'params': {}
        })

    def newQubit(self, q):
        """Create a new qubit."""
        self.conn.create(
            f"{q}", {
                'index': [-9, -9],
                'color': 'green',
                'probe': 'M0',
                'couplers': [],
                'qubit': {
                    'Ej': 10000000000.0,
                    'Ec': 250000000.0,
                    'f01': 5000000000.0,
                    'f12': 4750000000.0,
                    'fr': 6000000000.0,
                    'T1': 1e-05,
                    'Tr': 5000000.0,
                    'Te': 1.5e-05,
                    'test': 100,
                    'testdep': 200
                },
                'setting': {
                    'LO': 4350000000.0,
                    'POW': 21,
                    'OFFSET': 0.0
                },
                'waveform': {
                    'SR': 2000000000.0,
                    'LEN': 9.9e-05,
                    'SW': 'zero()',
                    'TRIG': 'zero()',
                    'RF': 'zero()',
                    'Z': 'zero()'
                },
                'channel': {
                    'I': 'AWG23.CH1',
                    'Q': None,
                    'LO': 'PSG105.CH1',
                    'DDS': None,
                    'SW': None,
                    'TRIG': None,
                    'Z': None
                },
                'calibration': {
                    'I': {
                        'delay': 0,
                        'distortion': 0
                    },
                    'Q': {
                        'delay': 0,
                        'distortion': 0
                    },
                    'Z': {
                        'delay': 0,
                        'distortion': 0
                    },
                    'DDS': {
                        'delay': 0,
                        'distortion': 0
                    },
                    'TRIG': {
                        'delay': 0,
                        'distortion': 0
                    }
                }
            })

    def newCoupler(self, c):
        """Create a new coupler."""
        self.conn.create(
            f"{c}", {
                'index': [-9, -9],
                'color': 'green',
                'qubits': [],
                'setting': {
                    'LO': 0,
                    'POW': 0,
                    'OFFSET': 0.0
                },
                'waveform': {
                    'SR': 2000000000.0,
                    'LEN': 9.9e-05,
                    'SW': 'zero()',
                    'TRIG': 'zero()',
                    'RF': 'zero()',
                    'Z': 'zero()'
                },
                'channel': {
                    'I': None,
                    'Q': None,
                    'LO': None,
                    'DDS': None,
                    'SW': None,
                    'TRIG': None,
                    'Z': 'AWG68.CH2'
                },
                'calibration': {
                    'I': {
                        'delay': 0,
                        'distortion': 0
                    },
                    'Q': {
                        'delay': 0,
                        'distortion': 0
                    },
                    'Z': {
                        'delay': 0,
                        'distortion': 0
                    },
                    'DDS': {
                        'delay': 0,
                        'distortion': 0
                    },
                    'TRIG': {
                        'delay': 0,
                        'distortion': 0
                    }
                },
                'qubit': {
                    'test': 100,
                    'testdep': 200
                }
            })

    def newReadout(self, r):
        """Create a new readout."""
        self.conn.create(
            f"{r}", {
                'index': [-9, -9],
                'color': 'green',
                'qubits': [],
                'adcsr': 1000000000.0,
                'setting': {
                    'LO': 6758520000.0,
                    'POW': 19,
                    'PNT': 4096,
                    'SHOT': 1024,
                    'TRIGD': 5e-07
                },
                'waveform': {
                    'SR': 2000000000.0,
                    'LEN': 9.9e-05,
                    'SW': 'zero()',
                    'TRIG': 'zero()',
                    'RF': 'zero()'
                },
                'channel': {
                    'I': 'AWG142.CH3',
                    'Q': 'AWG142.CH4',
                    'LO': 'PSG128.CH1',
                    'DDS': None,
                    'SW': None,
                    'TRIG': 'AWG142.CH3.Marker1',
                    'ADC': 'AD3.CH1'
                },
                'calibration': {
                    'I': {
                        'delay': 0,
                        'distortion': 0
                    },
                    'Q': {
                        'delay': 0,
                        'distortion': 0
                    },
                    'Z': {
                        'delay': 0,
                        'distortion': 0
                    },
                    'DDS': {
                        'delay': 0,
                        'distortion': 0
                    },
                    'TRIG': {
                        'delay': 0,
                        'distortion': 0
                    },
                    'drange': [100, 2000]
                }
            })

    def getQubit(self, q):
        """Get a qubit."""
        return self.query(q)

    def getCoupler(self, c):
        """Get a coupler."""
        return self.query(c)

    def getReadout(self, r):
        """Get a readout line."""
        return self.query(r)

    def getReadoutLine(self, r):
        """Get a readout line. (deprecated)"""
        warnings.warn(
            '`getReadoutLine` is no longer used and is being '
            'deprecated, use `getReadout` instead.', DeprecationWarning, 2)
        return self.getReadout(r)

    def getGate(self, name, *qubits):
        """Get a gate."""
        order_senstive = self.query(f"gate.{name}.__order_senstive__")
        if order_senstive is None:
            order_senstive = True
        if len(qubits) == 1 or order_senstive:
            qubits = '_'.join(qubits)
            ret = self.query(f"gate.{name}.{qubits}")
            if isinstance(ret, dict):
                return ret
            else:
                raise Exception(f"gate {name} of {qubits} not calibrated.")
        else:
            for qlist in permutations(qubits):
                try:
                    ret = self.query(f"gate.{name}.{'_'.join(qlist)}")
                    if isinstance(ret, dict):
                        return ret
                except:
                    break
            raise Exception(f"gate {name} of {qubits} not calibrated.")

    def getChannel(self, name):
        return {}

    def clear_buffer(self):
        """Clear the cache."""
        self._cache.clear()
        self._history.clear()
        self._cached_keys.clear()

    def commit(self):
        pass

    def rollback(self):
        pass

    def query(self, q):
        """Query the quark server."""
        u = {}
        if q in self._cache:
            return self._cache[q]
        elif q in self._cached_keys:
            u = _foldDict(_query(q, self._cache))
        ret, error = self.conn.query(q)
        #if error != 'None':
        #    raise KeyError(f"{q} not found")
        if isinstance(ret, dict):
            _update(ret, u)
        self._cache_result(q, ret)
        return ret

    def keys(self, pattern='*'):
        """Get keys."""
        if pattern == '*' or pattern == '.*':
            namespace = '.'
            keyword = '*'
        else:
            *namespace, keyword = pattern.split('.')
            if keyword[-1] == '*' and keyword != '*':
                keyword = keyword[:-1]
            else:
                keyword = '*'
            namespace = '.'.join(namespace)
        if namespace == '':
            namespace = '.'
        return self.conn.query(namespace, keyword=keyword)[0]

    def _cache_result(self, q, ret, record_history=False):
        """Cache the result."""
        return
        if isinstance(ret, dict):
            for k, v in _flattenDictIter(ret):
                key = f'{q}.{k}'
                if record_history and key not in self._history:
                    self._history[key] = self.query(key)
                self._cache[key] = v
                buffered_key = key.split('.')
                for i in range(len(buffered_key)):
                    self._cached_keys.add('.'.join([q, *buffered_key[:i]]))
        else:
            if record_history and q not in self._history:
                self._history[q] = self.query(q)
            self._cache[q] = ret

    def update(self, q, v, cache=False):
        """Update config."""
        self._cache_result(q, v, record_history=True)
        if not cache:
            self.conn.update(q, v)

    def update_all(self, data, cache=False):
        """Update all config."""
        for k, v in data:
            self._cache_result(k, v, record_history=True)
        if not cache:
            self.conn.batchup(data)

    def checkpoint(self):
        """Checkpoint."""
        return self.conn.checkpoint()

    def export(self):
        """Export."""
        return self.conn.snapshot()

    def load(self, data):
        """Load."""
        self.conn.clear()
        for k, v in data.items():
            self.conn.create(k, v)


class QuarkLocalConfig(ConfigProxy, CompileConfigMixin):
    def __init__(self, data) -> None:
        self._history = None
        self.__driver = DictDriver(copy.deepcopy(data))

    def query(self, q):
        return self.__driver.query(q)

    def keys(self, pattern='*'):
        return self.__driver.keys(pattern)

    def update(self, q, v, cache=False):
        self.__driver.update_many({q: v})

    def getQubit(self, name):
        return self.query(name)

    def getCoupler(self, name):
        return self.query(name)

    def getReadout(self, name):
        return self.query(name)

    def getReadoutLine(self, name):
        return self.query(name)

    def getGate(self, name, *qubits):
        order_senstive = self.query(f"gate.{name}.__order_senstive__")
        if order_senstive is None:
            order_senstive = True
        if len(qubits) == 1 or order_senstive:
            qubits = '_'.join(qubits)
            ret = self.query(f"gate.{name}.{qubits}")
            if isinstance(ret, dict):
                return ret
            else:
                raise Exception(f"gate {name} of {qubits} not calibrated.")
        else:
            for qlist in permutations(qubits):
                try:
                    ret = self.query(f"gate.{name}.{'_'.join(qlist)}")
                    if isinstance(ret, dict):
                        return ret
                except:
                    break
            raise Exception(f"gate {name} of {qubits} not calibrated.")

    def clear_buffer(self):
        pass

    def export(self):
        return copy.deepcopy(self.__driver.dct)
