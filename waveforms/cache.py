import functools
import hashlib
import pathlib
import pickle
import time
from typing import Any, Callable, Hashable, KeysView, Optional, Union

import portalocker

cache_dir = pathlib.Path.home() / '.waveforms' / 'cache'
cache_dir.mkdir(parents=True, exist_ok=True)

MAXVALUESIZE = 1024
Decorator = Callable[[Callable], Callable]


class Cache(dict):
    def __init__(self, name: str, path: pathlib.Path = cache_dir):
        name = name.split('.')
        self.path: pathlib.Path = path.joinpath(*name[:-1]) / (name[-1] +
                                                               '.cache.d')
        self._buffer = {}
        self._index = {}
        self._mtime = 0
        (self.path / 'values').mkdir(parents=True, exist_ok=True)

    @property
    def index(self) -> pathlib.Path:
        return self.path / 'index'

    @staticmethod
    def _hash(key: Hashable) -> str:
        return hashlib.md5(pickle.dumps(key)).hexdigest()

    def _syncIndex(self) -> None:
        if not self.index.exists() or self._mtime > self.index.stat().st_mtime:
            with portalocker.Lock(self.index, 'wb') as fh:
                pickle.dump(self._index, fh)
        elif self._mtime < self.index.stat().st_mtime:
            with portalocker.Lock(self.index, 'rb') as fh:
                self._index = pickle.load(fh)
        self._mtime = self.index.stat().st_mtime

    def _valuePath(self, key: Hashable) -> pathlib.Path:
        hashedKey = self._hash(key)
        return self.path / 'values' / hashedKey

    def _storeValue(self, k: Hashable, buf: bytes) -> None:
        with portalocker.Lock(self._valuePath(k), 'wb') as fh:
            fh.write(buf)

    def _loadValue(self, k: Hashable) -> None:
        with portalocker.Lock(self._valuePath(k), 'rb') as fh:
            buf = fh.read()
        hashstr = self._hash(buf)
        self._index[k] = self._valuePath(k).stat().st_mtime, None, hashstr
        self._buffer[k] = pickle.loads(buf)

    def __setitem__(self, k: Hashable, v: Any) -> None:
        buf = pickle.dumps(v)
        if len(buf) <= MAXVALUESIZE:
            mtime = time.time()
            self._index[k] = mtime, buf, ''
            self._mtime = mtime
            self._valuePath(k).unlink(missing_ok=True)
        else:
            hashstr = self._hash(buf)
            if k not in self._index or hashstr != self._index[k][2]:
                self._storeValue(k, buf)
                mtime = self._valuePath(k).stat().st_mtime
                self._index[k] = mtime, None, hashstr
                self._mtime = mtime
        self._syncIndex()
        self._buffer[k] = v

    def __getitem__(self, k: Hashable) -> Any:
        self._syncIndex()
        mtime, buf, hashstr = self._index[k]
        if k not in self._buffer and buf is not None:
            self._buffer[k] = pickle.loads(buf)
        elif k not in self._buffer:
            self._loadValue(k)
        return self._buffer[k]

    def __contains__(self, x: Hashable) -> bool:
        self._syncIndex()
        return x in self._index

    def keys(self) -> KeysView[Hashable]:
        self._syncIndex()
        return self._index.keys()

    def clear(self) -> None:
        self._buffer.clear()
        self._index.clear()
        self.index.unlink()
        for f in (self.path / 'values').iterdir():
            f.unlink()
        self._mtime = 0


__caches = {}


def _getCache(name: str) -> Cache:
    try:
        return __caches[name]
    except:
        __caches[name] = Cache(name)
        return __caches[name]


def cache(storage: Optional[Union[str, Cache]] = None) -> Decorator:
    def decorator(func: Callable,
                  storage: Optional[Union[str, Cache]] = storage) -> Callable:
        if storage is None:
            storage = _getCache('.'.join([func.__module__, func.__name__]))
        elif isinstance(storage, str):
            storage = _getCache(storage)
        elif not isinstance(storage, Cache):
            raise Exception(f'storage type {type(storage)} not understand!')

        @functools.wraps(func)
        def wrapper(*args, **kwds):
            kwds = {k: kwds[k] for k in sorted(kwds)}
            try:
                ret = storage[pickle.dumps((args, kwds))]
            except:
                ret = func(*args, **kwds)
                storage[pickle.dumps((args, kwds))] = ret
            return ret

        wrapper.cache = storage
        return wrapper

    return decorator


def clear(name: str) -> None:
    cache = _getCache(name)
    cache.clear()
