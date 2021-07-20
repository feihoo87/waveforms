import warnings
from functools import lru_cache
from itertools import permutations

from waveforms.quantum.circuit.qlisp.config import ConfigProxy


class QuarkConfig(ConfigProxy):
    def __init__(self, host='127.0.0.1'):
        self.host = host
        self._cache = {}
        self.connect()

    def connect(self):
        from quark import connect
        self.conn = connect('QuarkServer', host=self.host)

    def newGate(self, name, *qubits):
        qubits = '_'.join(qubits)
        self.conn.alter(f"+gate.{name}.{qubits}")

    def newQubit(self, q):
        self.conn.alter(f"+{q}")

    def newCoupler(self, c):
        self.conn.alter(f"+{c}")

    def newReadout(self, r):
        self.conn.alter(f"+{r}")

    def getQubit(self, q):
        return self.query(q)[0]

    def getCoupler(self, c):
        return self.query(c)[0]

    def getReadout(self, r):
        return self.query(r)[0]

    def getReadoutLine(self, r):
        warnings.warn(
            '`getReadoutLine` is no longer used and is being '
            'deprecated, use `getReadout` instead.', DeprecationWarning, 2)
        return self.getReadout(r)

    def getGate(self, name, *qubits):
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
        ret = self.query(f"gate.{name}.{qubits}")[0]
        if isinstance(ret, dict):
            return ret
        else:
            raise Exception(f"gate {name} of {qubits} not calibrated.")

    def getChannel(self, name):
        return {}

    def clear_buffer(self):
        self.query.cache_clear()

    def commit(self):
        pass

    @lru_cache()
    def query(self, q):
        return self.conn.query(q)

    def update(self, q, v):
        self.conn.update(q, v)
