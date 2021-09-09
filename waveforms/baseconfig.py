from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Optional, Union

from .dicttree import flattenDict as _flattenDict
from .dicttree import flattenDictIter as _flattenDictIter
from .dicttree import foldDict as _foldDict
from .dicttree import update_tree as _update


def _query(q, dct):
    return {
        k.removeprefix(q + '.'): v
        for k, v in dct.items() if k.startswith(q + '.')
    }


def randomStr(n):
    s = ('abcdefghijklmnopqrstuvwxyz'
         'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
         '0123456789')
    return ''.join(random.choices(s, k=n))


def mixin(o: object, *traits: type) -> object:
    bases = (o.__class__, *traits)
    name = '_'.join([cls.__name__ for cls in bases])
    name = '_'.join([name, randomStr(6)])
    cls = type(name, bases, {})
    o.__class__ = cls
    return o


class TraitMeta(type):
    _traits = {}

    def __new__(cls, name, bases, namespace):
        cls = super().__new__(cls, name, bases, namespace)
        if name != 'Trait':
            TraitMeta._traits[name] = cls
        return cls


class Trait(metaclass=TraitMeta):
    pass


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


class ConfigSection(dict):
    def __init__(self, cfg: BaseConfig, key: str):
        self._cfg_ = cfg
        self._key_ = key

    def __setitem__(self, key: str, value: Any) -> None:
        if self._cfg_ is None:
            self._modified_ = True
        else:
            self._cfg_._modified_ = True
        if isinstance(value, dict) and not isinstance(value, ConfigSection):
            key, *traits = key.split(':')
            if self._cfg_ is None:
                cfg = self
                k = key
            else:
                cfg = self._cfg_
                k = '.'.join([self._key_, key])
            d = ConfigSection(cfg, k)
            d.update(value)
            value = d
        elif isinstance(value, ConfigSection):
            value.__class__ = ConfigSection
        super().__setitem__(key, value)

    def __getitem__(self, key: str) -> ValueType:
        key, *traits = key.split(':')
        if self._cfg_ is not None:
            d = self._cfg_.query(self._key_)
            if self is not d:
                self.update(d)
        ret = super().__getitem__(key)
        if isinstance(ret, ConfigSection) and len(traits) > 0:
            traits = [
                TraitMeta._traits[n] for n in traits if n in TraitMeta._traits
            ]
            mixin(ret, *traits)
        return ret

    def __delitem__(self, key: str) -> None:
        if self._cfg_ is None:
            self._modified_ = True
        else:
            self._cfg_._modified_ = True
        return super().__delitem__(key)

    def __delattr__(self, name: str) -> None:
        if name in self:
            return self.__delitem__(name)
        else:
            return super().__delattr__(name)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.__getattribute__(name)
        except:
            try:
                return self.__getitem__(name)
            except:
                raise AttributeError(f'Not Find Attr: {name}')

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self:
            self.__setitem__(name, value)
        else:
            super().__setattr__(name, value)

    def __deepcopy__(self, memo):
        dct = {k: copy.deepcopy(v) for k, v in self.items()}
        return dct

    def query(self, q: Union[str, set, tuple, list,
                             dict]) -> Union[dict, ValueType]:
        if self._key_ is None:
            prefix = []
        else:
            prefix = self._key_.split('.')
        return query(q, self, prefix=prefix)

    def set(self, q, value):
        if self._key_ is None:
            prefix = []
        else:
            prefix = self._key_.split('.')
        setKey(q, value, self, prefix=prefix)

    def update(self, other):
        _update(self, other)


ValueType = Union[str, int, float, list, ConfigSection]


class BaseConfig(ConfigSection):
    def __init__(self,
                 path: Optional[Union[str, Path]] = None,
                 backup: bool = False):
        super().__init__(None, None)
        if isinstance(path, str):
            path = Path(path)
        self._path_ = path
        self._backup_ = backup
        self._modified_ = False

        if self._path_ is None:
            return
        if self._path_.exists():
            self.reload()
            if '__version__' not in self:
                self['__version__'] = 1
        else:
            self._path_.parent.mkdir(parents=True, exist_ok=True)
            self['__version__'] = 1
            self._modified_ = True
            self.commit()

    def commit(self):
        if not self._modified_ or self._path_ is None:
            return
        if self._backup_ and self._path_.exists():
            v = self['__version__']
            bk = self._path_.with_stem(self._path_.stem + f"_{v}")
            self._path_.rename(bk)
        with self._path_.open('w') as f:
            self['__version__'] = self['__version__'] + 1
            json.dump(self, f, indent=4)
            self._modified_ = False

    def rollback(self):
        if not self._modified_ or self._path_ is None:
            return
        self.reload()

    def reload(self):
        with self._path_.open('r') as f:
            dct = json.load(f)
            self.clear()
            self.update(dct)
            self._modified_ = False

    @classmethod
    def fromdict(cls, d: dict) -> BaseConfig:
        ret = cls()
        ret.update(d)
        return ret
