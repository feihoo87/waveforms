from types import MappingProxyType

import numpy as np
import scipy.sparse as sp


def freeze(x):
    """Freeze a mutable object.
    """
    if isinstance(x, (int, float, complex, str, bytes, type(None))):
        pass
    elif isinstance(x, (list, tuple)):
        return tuple([freeze(y) for y in x])
    elif isinstance(x, dict):
        return MappingProxyType({k: freeze(v) for k, v in x.items()})
    elif isinstance(x, set):
        return frozenset([freeze(y) for y in x])
    elif isinstance(x, (np.ndarray, np.matrix)):
        x.flags.writeable = False
    elif isinstance(x, sp.spmatrix):
        x.data.flags.writeable = False
        if x.format in {'csr', 'csc', 'bsr'}:
            x.indices.flags.writeable = False
            x.indptr.flags.writeable = False
        elif x.format == 'coo':
            x.row.flags.writeable = False
            x.col.flags.writeable = False
    elif isinstance(x, bytearray):
        x = bytes(x)
    return x
