import json
import pickle
from abc import ABC, abstractmethod
from threading import Lock
from typing import Any

import portalocker

from .dicttree import NOTSET, query_tree, update_tree


class NamespaceDriver(ABC):
    @abstractmethod
    def query(self, key: str) -> Any:
        pass

    @abstractmethod
    def keys(self, pattern: str = '*') -> list[str]:
        pass

    @abstractmethod
    def update_many(self, key_value_pairs: dict[str, Any]):
        pass

    @abstractmethod
    def create_many(self, key_value_pairs: dict[str, Any]):
        pass

    @abstractmethod
    def delete_many(self, keys: list[str]):
        pass

    def close(self):
        pass

    def commit(self):
        pass


class DictDriver(NamespaceDriver):
    def __init__(self, dct):
        self.dct = dct

    def query(self, key: str) -> Any:
        return query_tree(key, self.dct)

    def keys(self, pattern: str = '*') -> list[str]:
        if not pattern.endswith('*'):
            pattern += '.*'
        keys = pattern.split('.')
        if keys[0].endswith('*'):
            d = self.dct
        else:
            d = query_tree('.'.join(keys[:-1]), self.dct)
        if not isinstance(d, dict):
            return []
        ret = d.keys()
        if keys[-1] == '*':
            return ret
        else:
            s = keys[-1].removesuffix('*')
            return [k for k in ret if k.startswith(s)]

    def update_many(self, key_value_pairs: dict[str, Any]):
        for key, value in key_value_pairs.items():
            keys = list(reversed(key.split('.')))
            for k in keys[:-1]:
                value = {k: value}
            update_tree(self.dct, {keys[-1]: value})

    def create_many(self, key_value_pairs: dict[str, Any]):
        self.update_many(key_value_pairs)

    def delete_many(self, keys: list[str]):
        for key in keys:
            ks = key.split('.')
            d = self.dct
            for k in ks[:-1]:
                if k in d:
                    d = d[k]
                else:
                    break
            else:
                try:
                    del d[ks[-1]]
                except KeyError:
                    pass


class JSONDriver(DictDriver):
    def __init__(self, path):
        super().__init__(dict())
        self.path = path
        self.load()

    def load(self):
        try:
            with open(self.path, 'r') as f:
                self.dct = json.load(f)
        except:
            self.dct = {}

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.dct, f)

    def commit(self):
        self.save()


class PickleDriver(DictDriver):
    def __init__(self, path):
        super().__init__(dict())
        self.path = path
        self.load()

    def load(self):
        try:
            with portalocker.Lock(self.path, 'rb') as f:
                self.dct = pickle.load(f)
        except:
            self.dct = {}

    def save(self):
        with portalocker.Lock(self.path, 'wb') as f:
            pickle.dump(self.dct, f)

    def commit(self):
        self.save()


class NamespaceCache():
    def __init__(self):
        self.caches = {}
        self.notset_caches = set()
        self.fully_cached_keys = set()
        self.keys_caches = {}
        self.lock = Lock()

    def cache(self, key, value):
        with self.lock:
            self._cache(key, value)

    def delete(self, key):
        with self.lock:
            self._delete(key)

    def query(self, key):
        with self.lock:
            return self._query(key)

    def keys(self, pattern='*'):
        with self.lock:
            return self._keys(pattern)

    def clear(self):
        with self.lock:
            self._clear()

    def _cache(self, key: str, value: Any):
        if isinstance(value, tuple) and value[0] is NOTSET:
            self._cache_notset(value[1])
            return
        self._cache_keys(key, value)
        if key == '.':
            self.caches = value
            return
        keys = key.split('.')
        for k in reversed(keys):
            value = {k: value}
        update_tree(self.caches, value)

    def _fully_cached(self, key: str) -> bool:
        if key in self.fully_cached_keys:
            return True
        for k in self.notset_caches:
            if key.startswith(k + '.') or key == k:
                return True
        return False

    def _cache_notset(self, key: str):
        self.notset_caches.add(key)

    def _cache_keys(self, key: str, value: Any):
        s = set()
        for k in self.notset_caches:
            if not key.startswith(k + '.') and key != k:
                s.add(k)
        self.notset_caches = s
        self.fully_cached_keys.add(key)
        if isinstance(value, dict):
            for k, v in value.items():
                if key == '.':
                    self._cache_keys(k, v)
                else:
                    self._cache_keys(key + '.' + k, v)

    def _delete(self, key: str):
        if key == '.':
            self._clear()
            return
        keys = key.split('.')
        parent = self.caches
        d = self.caches
        for k in keys:
            try:
                parent = d
                d = d[k]
            except KeyError:
                return
        del parent[keys[-1]]
        self._delete_keys(key)

    def _delete_keys(self, key: str):
        self.notset_caches.add(key)
        s = set()
        for k in self.fully_cached_keys:
            if not k.startswith(key + '.') and k != key:
                s.add(k)
        self.fully_cached_keys = s

    def _query(self, key: str) -> tuple[Any, bool]:
        for k in self.notset_caches:
            if key.startswith(k + '.') or key == k:
                return (NOTSET, k), True
        ret = query_tree(key, self.caches)
        if isinstance(ret, tuple) and ret[0] is NOTSET:
            return None, False
        return ret, self._fully_cached(key)

    def _keys(self, pattern='*'):
        if not pattern.endswith('*'):
            pattern += '.*'
        keys = pattern.split('.')
        d, fully_cached = self._query('.'.join(keys[:-1]))
        if not fully_cached:
            return None
        if not isinstance(d, dict):
            return []
        ret = d.keys()
        if keys[-1] == '*':
            return ret
        else:
            s = keys[-1].removesuffix('*')
            return [k for k in ret if k.startswith(s)]

    def _clear(self):
        self.caches.clear()
        self.fully_cached_keys.clear()
        self.keys_caches.clear()
        self.notset_caches.clear()

    def __contains__(self, key: str):
        return key in self.fully_cached_keys


class NamespaceEventLog():
    def __init__(self):
        self.updates = NamespaceCache()
        self.creates = NamespaceCache()
        self.deletes = NamespaceCache()
