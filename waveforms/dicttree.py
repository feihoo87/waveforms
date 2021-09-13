import copy
import sys
from typing import Any, Generator


def flattenDictIter(d: dict,
                    prefix: list = []
                    ) -> Generator[tuple[str, Any], None, None]:
    for k in d:
        if isinstance(d[k], dict):
            yield from flattenDictIter(d[k], prefix=[*prefix, k])
        else:
            yield '.'.join(prefix + [k]), d[k]


def flattenDict(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in flattenDictIter(d)}


def foldDict(d: dict[str, Any]) -> dict[str, Any]:
    ret = {}
    for k, v in d.items():
        keys = k.split('.')
        d = ret
        for key in keys[:-1]:
            if key not in d:
                d[key] = dict()
            d = d[key]
        d[keys[-1]] = v
    return ret


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton,
                                        cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class _NOTSET(metaclass=Singleton):
    __slots__ = ()

    def __repr__(self):
        return 'N/A'


class _UNKNOW(metaclass=Singleton):
    __slots__ = ()

    def __repr__(self) -> str:
        return "Unknow"


NOTSET = _NOTSET()
UNKNOW = _UNKNOW()


class Delete(metaclass=Singleton):
    __slots__ = ()

    def __repr__(self):
        return 'Delete'


DELETE = Delete()


class Update():
    __slots__ = ('o', 'n')

    def __init__(self, o, n):
        self.o = o
        self.n = n

    def __repr__(self):
        return f"Update: {self.o!r} ==> {self.n!r}"

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Update) and self.n == o.n


class Create():
    __slots__ = ('n', 'delete')

    def __init__(self, n, delete=False):
        self.n = n
        self.delete = delete

    def __repr__(self):
        if self.delete:
            return f"Delete and Create: {self.n!r}"
        else:
            return f"Create: {self.n!r}"

    def __eq__(self, o: object) -> bool:
        return isinstance(o,
                          Create) and self.n == o.n and self.delete == o.delete


def diff(d1: dict, d2: dict) -> dict:
    ret = {}
    for k in d2:
        if k in d1:
            if isinstance(d2[k], type(d1[k])) and d1[k] == d2[k]:
                pass
            elif isinstance(d1[k], dict) and isinstance(d2[k], dict):
                ret[k] = diff(d1[k], d2[k])
            else:
                ret[k] = Update(d1[k], d2[k])
        else:
            ret[k] = Create(d2[k])
    for k in d1:
        if k not in d2:
            ret[k] = DELETE
    return ret


def patch(source, diff, in_place=False):
    if in_place:
        ret = source
    else:
        ret = copy.copy(source)
    for k, v in diff.items():
        if isinstance(v, dict):
            ret[k] = patch(source[k], v, in_place=in_place)
        else:
            if isinstance(v, Update):
                ret[k] = v.n
            elif isinstance(v, Create):
                if v.delete or k not in ret:
                    ret[k] = v.n
                else:
                    update_tree(ret[k], v.n)
            elif v is DELETE:
                del ret[k]
            else:
                raise ValueError(f"Unsupported patch: {v!r}")
    return ret


def merge(diff1, diff2, origin=None):
    if origin is not None:
        updated = patch(patch(origin, diff1), diff2)
        return diff(origin, updated)

    ret = {}
    for k, v in diff1.items():
        if k in diff2:
            v2 = diff2[k]
            if isinstance(v, dict) and isinstance(v2, dict):
                d = merge(v, v2)
                if d:
                    ret[k] = d
            else:
                if isinstance(v, Update) and isinstance(v2, Update):
                    ret[k] = Update(v.o, v2.n)
                elif isinstance(v, Create) and isinstance(v2, dict):
                    ret[k] = Create(patch(copy.copy(v.n), v2, True))
                elif isinstance(v, Create) and isinstance(v2, Update):
                    ret[k] = Create(v2.n)
                elif isinstance(v, Create) and v2 is DELETE:
                    pass
                elif v2 is DELETE:
                    ret[k] = DELETE
                elif v is DELETE and isinstance(v2, Create):
                    if isinstance(v2.n, dict):
                        ret[k] = Create(v2.n, delete=True)
                    else:
                        ret[k] = Update(UNKNOW, v2.n)
                elif isinstance(v2, Create) and v2.delete:
                    ret[k] = v2
                else:
                    raise ValueError(f"Unsupported merge: {v!r} {v2!r}")
        else:
            ret[k] = v
    for k, v in diff2.items():
        if k not in diff1:
            ret[k] = v
    return ret


def print_diff(d, lim=None, offset=0, file=sys.stdout):
    count = 0
    for i, (k, v) in enumerate(flattenDictIter(d)):
        if i >= offset:
            print(f"{k:40}", v, file=file)
            count += 1
            if lim is not None and count >= lim:
                break


def update_tree(result, updates):
    for k, v in updates.items():
        if isinstance(v, dict):
            if k not in result or not isinstance(result[k], dict):
                result[k] = {}
            update_tree(result[k], v)
        else:
            result[k] = v
    return result


def query_tree(q, dct, prefix=[]):
    keys = q.split('.')
    for i, key in enumerate(keys):
        if key not in dct:
            return (NOTSET, '.'.join(prefix + keys[:i + 1]))
        dct = dct[key]
    return dct


def sorted_tree(dct, *, keys=None):
    if keys is None or callable(keys):
        key = keys
    elif isinstance(keys, list):
        key = keys[0]
        if len(keys) > 1:
            keys = keys[1:]
        else:
            keys = None
    elif (isinstance(keys, tuple) and len(keys) == 2
          and isinstance(keys[1], dict)):
        key = keys[0]
        keys = keys[1]
    else:
        raise Exception(f"Unsupported keys: {keys!r}")

    if isinstance(dct, dict):
        if isinstance(keys, dict):
            default = keys.get('default', None)
            return {
                k: sorted_tree(dct[k], keys=keys.get(k, default))
                for k in sorted(dct.keys(), key=key)
            }
        else:
            return {
                k: sorted_tree(dct[k], keys=keys)
                for k in sorted(dct.keys(), key=key)
            }
    elif isinstance(dct, set):
        if isinstance(keys, dict):
            default = keys.get('default', None)
            return set([
                sorted_tree(v, keys=keys.get(v, default))
                for v in sorted(list(dct), key=key)
            ])
        else:
            return set([
                sorted_tree(v, keys=keys) for v in sorted(list(dct), key=key)
            ])
    else:
        return dct
