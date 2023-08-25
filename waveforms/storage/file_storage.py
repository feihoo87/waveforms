import bisect
import pathlib
from io import BufferedRandom
from itertools import chain

import dill

from .base_storage import Storage


class _ListFile():

    def __init__(self, file, *args, cached=False):
        self._filename = file
        self._file: BufferedRandom = open(self._filename, 'a+b')
        self._cache = []
        self._cached = cached

        for x in args:
            self.append(x)

    def __del__(self):
        self._file.close()

    def unlink(self):
        try:
            self._file.close()
        except:
            pass
        try:
            pathlib.Path(self._filename).unlink()
        except:
            pass

    @staticmethod
    def _append_file(file: BufferedRandom, x):
        file.seek(0, 2)
        dill.dump(x, file)

    @staticmethod
    def _iter_file(file: BufferedRandom):
        file.seek(0)
        while True:
            try:
                x = dill.load(file)
            except EOFError:
                break
            yield x

    def append(self, x):
        self._append_file(self._file, x)
        if self._cached:
            self._cache.append(x)

    def __iter__(self):
        if self._cached:
            return iter(self._cache)
        yield from self._iter_file(self._file)

    def __getstate__(self) -> dict:
        return {
            '_filename': self._filename,
            '_file': None,
            '_cache': [],
            '_cached': self._cached
        }

    def __setstate__(self, state: dict):
        self.__dict__.update(state)
        self._file = open(self._filename, 'a+b')

    def __repr__(self):
        return f'_ListFile({self._filename})'


class FileStorage(Storage):

    def __init__(self, path):
        super().__init__()
        self._path = pathlib.Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        if (self._path / 'index').exists():
            with open(self._path / 'index', 'rb') as f:
                obj = dill.load(f)
            self.storage = obj.storage
            self.pos = obj.pos
            self.timestamps = obj.timestamps
            self.iteration = obj.iteration
            self.depends = obj.depends
            self.shape = obj.shape
            self.dims = obj.dims
            self.vars_dims = obj.vars_dims
            self.ctime = obj.ctime
            self.mtime = obj.mtime
            self._init_keys = obj._init_keys
            self._frozen_keys = obj._frozen_keys
            self._key_levels = obj._key_levels
            self.save_kwds = obj.save_kwds

    def unlink(self):
        for file in pathlib.Path(self._path).iterdir():
            if file.name.startswith('storage_'):
                file.unlink()
            elif file.name.startswith('pos_'):
                file.unlink()
            elif file.name.startswith('timestamp_'):
                file.unlink()
            elif file.name.startswith('iteration_'):
                file.unlink()
            elif file.name == 'index':
                file.unlink()
        try:
            pathlib.Path(self._path).unlink()
        except:
            pass

    def _append(self, iteration, pos, dataframe, kwds, now):
        for k, v in chain(kwds.items(), dataframe.items()):
            if k in self._frozen_keys:
                continue
            if k.startswith('__'):
                continue
            if self.vars_dims.get(k, ()) == () and k not in dataframe:
                continue
            self.count += 1
            if k not in self.storage:
                self.storage[k] = _ListFile(self._path / f'storage_{k}', v)
                if k in self.vars_dims:
                    self.pos[k] = tuple(
                        _ListFile(self._path / f'pos_{k}_{j}', pos[i])
                        for j, i in enumerate(self.vars_dims[k]))
                else:
                    self.pos[k] = tuple(
                        _ListFile(self._path / f'pos_{k}_{j}', i)
                        for j, i in enumerate(pos))
                self.timestamps[k] = _ListFile(self._path / f'timestamp_{k}',
                                               now.timestamp())
                self.iteration[k] = _ListFile(self._path / f'iteration_{k}',
                                              iteration)
            else:
                if k in self.vars_dims:
                    pos_k = tuple(pos[i] for i in self.vars_dims[k])
                    if k not in dataframe and pos_k in zip(
                            *[list(l) for l in self.pos[k]]):
                        continue
                    for i, l in zip(pos_k, self.pos[k]):
                        l.append(i)
                else:
                    for i, l in zip(pos, self.pos[k]):
                        l.append(i)
                self.timestamps[k].append(now.timestamp())
                self.iteration[k].append(iteration)
                self.storage[k].append(v)
        with open(self._path / 'index', 'wb') as f:
            dill.dump(self, f)

    def _get_array(self, key, shape, count):
        import numpy as np

        if key in self.vars_dims:
            shape = tuple([shape[i] for i in self.vars_dims[key]])

        data, data_shape, data_count = self.cache.get(key, (None, (), 0))
        if (data_shape, data_count) == (shape, count):
            return data
        try:
            tmp = np.asarray(list(self.storage[key]))
            if data_shape != shape:
                data = np.full(shape + tmp.shape[1:], np.nan, dtype=tmp.dtype)
        except:
            tmp = list(self.storage[key])
            if data_shape != shape:
                data = np.full(shape, np.nan, dtype=object)
        try:
            pos = tuple([list(l) for l in self.pos[key]])
            data[pos] = tmp
        except:
            print(key)
            print(data)
            print(pos)
            print(tmp)
            raise
        self.cache[key] = (data, shape, count)
        return data

    def _get_part(self, key, skip):
        i = bisect.bisect_left(list(self.iteration[key]), skip)
        pos = tuple(list(p)[i:] for p in self.pos[key])
        iteration = list(self.iteration[key])[i:]
        data = list(self.storage[key])[i:]
        return data, iteration, pos
