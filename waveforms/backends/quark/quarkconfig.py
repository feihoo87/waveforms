import warnings
from itertools import permutations

from waveforms.baseconfig import _flattenDictIter, _foldDict, _query, _update
from waveforms.quantum.circuit.qlisp.config import ConfigProxy


class QuarkConfig(ConfigProxy):
    def __init__(self, host='127.0.0.1'):
        self.host = host
        self._cache = {}
        self._cached_keys = set()
        self.connect()

    def connect(self):
        """Connect to the quark server."""
        from quark import connect
        self.conn = connect('QuarkServer', host=self.host)

    def newGate(self, name, *qubits):
        """Create a new gate."""
        qubits = '_'.join(qubits)
        self.conn.alter(f"+gate.{name}.{qubits}")

    def newQubit(self, q):
        """Create a new qubit."""
        self.conn.alter(f"+{q}")

    def newCoupler(self, c):
        """Create a new coupler."""
        self.conn.alter(f"+{c}")

    def newReadout(self, r):
        """Create a new readout."""
        self.conn.alter(f"+{r}")

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
        # if name not in self['gates']:
        #     raise KeyError(f'"{name}" gate not defined.')
        # if name == 'rfUnitary':
        #     return self.getObject(f"gates.{name}.{','.join(qubits)}",
        #                           cls='rfUnitary')
        # elif name == 'Measure':
        #     return self.getObject(f"gates.{name}.{','.join(qubits)}",
        #                           cls='Measure')
        # if ('__order_senstive__' in self['gates'][name]
        #         and self['gates'][name]['__order_senstive__']):
        #     return self.getObject(f"gates.{name}.{','.join(qubits)}",
        #                           cls='Gate')
        # else:
        #     for qlist in permutations(qubits):
        #         try:
        #             return self.getObject(f"gates.{name}.{','.join(qlist)}",
        #                                   cls='Gate')
        #         except:
        #             pass
        #     else:
        #         raise KeyError(f'Could not find "{name}" gate for {qubits}')

        qubits = '_'.join(qubits)
        ret = self.query(f"gate.{name}.{qubits}")
        if isinstance(ret, dict):
            return ret
        else:
            raise Exception(f"gate {name} of {qubits} not calibrated.")

    def getChannel(self, name):
        return {}

    def clear_buffer(self):
        """Clear the cache."""
        self._cache.clear()
        self._cached_keys.clear()

    def commit(self):
        pass

    def query(self, q):
        """Query the quark server."""
        u = {}
        if q in self._cache:
            return self._cache[q]
        elif q in self._cached_keys:
            u = _foldDict(_query(q, self._cache))
        ret = self.conn.query(q)[0]
        self._cache_result(q, ret)
        _update(ret, u)
        return ret

    def _cache_result(self, q, ret):
        """Cache the result."""
        if isinstance(ret, dict):
            for k, v in _flattenDictIter(ret):
                key = f'{q}.{k}'
                self._cache[key] = v
                buffered_key = key.split('.')
                for i in range(len(buffered_key)):
                    self._cached_keys.add('.'.join([q, *buffered_key[:i]]))
        else:
            self._cache[q] = ret

    def update(self, q, v, cache=False):
        """Update config."""
        self._cache_result(q, v)
        if not cache:
            self.conn.update(q, v)
