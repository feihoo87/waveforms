import json
from itertools import count
from pathlib import Path


def queryKey(q, dct, prefix=None):
    if prefix is None:
        prefix = []

    keys = q.split('.', maxsplit=1)

    if not isinstance(dct, dict):
        k = '.'.join(prefix)
        raise KeyError(
            f"Query {k}.{q} error, type '{k}' is {type(dct)}, not dict.")
    try:
        sub = dct[keys[0]]
    except KeyError:
        k = '.'.join([*prefix, keys[0]])
        raise KeyError(
            f"Query {'.'.join([*prefix, q])} error, key '{k}' not found.")

    if len(keys) == 1:
        return sub

    else:
        return queryKey(keys[1], sub, [*prefix, keys[0]])


def query(q, dct, prefix=None):
    if isinstance(q, str):
        return queryKey(q, dct, prefix)
    elif isinstance(q, list):
        return [query(sub_q, dct, prefix) for sub_q in q]
    elif isinstance(q, tuple):
        return tuple([query(sub_q, dct, prefix) for sub_q in q])
    elif isinstance(q, set):
        return {sub_q: query(sub_q, dct, prefix) for sub_q in q}
    elif isinstance(q, dict):
        if prefix is None:
            prefix = []
        ret = {}
        for k, sub_q in q.items():
            if sub_q is None:
                ret[k] = queryKey(k, dct, prefix)
            else:
                ret[k] = query(sub_q, queryKey(k, dct, prefix), [*prefix, k])
        return ret
    else:
        raise TypeError


def setKey(q, value, dct, prefix=None):
    if prefix is None:
        prefix = []

    keys = q.split('.', maxsplit=1)

    if len(keys) == 1:
        if keys[0] in dct and isinstance(dct[keys[0]],
                                         dict) and not isinstance(value, dict):
            k = '.'.join([*prefix, keys[0]])
            raise ValueError(f'try to set a dict {k} to {type(value)}')
        else:
            dct[keys[0]] = value
    else:
        if keys[0] in dct and isinstance(dct[keys[0]], dict):
            sub = dct[keys[0]]
        elif keys[0] in dct and not isinstance(dct[keys[0]], dict):
            k = '.'.join([*prefix, keys[0]])
            raise ValueError(f'try to set a dict {k} to {type(value)}')
        else:
            sub = {}
            dct[keys[0]] = sub
        setKey(keys[1], sub, [*prefix, keys[0]])


class Config(dict):
    def __init__(self, path, backup=True):
        self.path = Path(path)
        self.backup = backup
        if self.path.exists():
            self.reload()
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.commit()

    def reload(self):
        with self.path.open('r') as f:
            dct = json.load(f)
            self.clear()
            self.update(dct)

    def commit(self):
        if self.backup and self.path.exists():
            for i in count():
                bk = self.path.parent / (self.path.stem + f"_{i}" +
                                         self.path.suffix)
                if not bk.exists():
                    break
            self.path.rename(bk)
        with self.path.open('w') as f:
            json.dump(self, f, indent=4)

    def rollback(self):
        self.reload()

    def update(self, other):
        def update(d, u):
            for k in u:
                if isinstance(u[k], dict):
                    if k not in d:
                        d[k] = u[k]
                    elif isinstance(d[k], dict):
                        update(d[k], u[k])
                    else:
                        raise TypeError()
                else:
                    d[k] = u[k]

        update(self, other)

    def query(self, q):
        return query(q, self)

    def set(self, q, value):
        setKey(q, value, self, prefix=None)
