import warnings
from itertools import permutations

from waveforms.baseconfig import _flattenDictIter, _foldDict, _query, _update
from waveforms.quantum.circuit.qlisp.config import ConfigProxy


class QuarkConfig(ConfigProxy):
    def __init__(self, host='127.0.0.1'):
        self.host = host
        self._cache = {}
        self._history = {}
        self._cached_keys = set()
        self.connect()

    def connect(self):
        """Connect to the quark server."""
        from quark import connect
        self.conn = connect('QuarkServer', host=self.host, verbose=False)

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
                'probe': 'M0',
                'couplers': [],
                'setting': {
                    'LO': 4350000000.0,
                    'POW': 21,
                    'OFFSET': 0.0
                },
                'waveform': {
                    'SR': 2000000000.0,
                    'LEN': 99e-06,
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
                }
            })

    def newCoupler(self, c):
        """Create a new coupler."""
        self.conn.create(f"{c}", {})

    def newReadout(self, r):
        """Create a new readout."""
        self.conn.create(f"{r}", {})

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
        ret = self.conn.query(q)[0]
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
