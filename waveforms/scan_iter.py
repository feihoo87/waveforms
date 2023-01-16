import bisect
import inspect
import warnings
from abc import ABC, abstractclassmethod
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain, count
from queue import Queue
from typing import Any, Callable, Iterable, Optional, Sequence, Type, Union

_NODEFAULT = object()


class BaseOptimizer(ABC):

    @abstractclassmethod
    def ask(self) -> tuple:
        pass

    @abstractclassmethod
    def tell(self, suggested: Sequence, value: Any):
        pass

    @abstractclassmethod
    def get_result(self):
        pass


@dataclass
class OptimizerConfig():
    cls: Type[BaseOptimizer]
    dimensions: list = field(default_factory=list)
    args: tuple = ()
    kwds: dict = field(default_factory=dict)
    max_iters: int = 100


class FeedbackPipe():
    __slots__ = (
        'keys',
        '_queue',
    )

    def __init__(self, keys):
        self.keys = keys
        self._queue = deque()

    def __iter__(self):
        while True:
            try:
                yield self._queue.popleft()
            except:
                break

    def __call__(self):
        return self.__iter__()

    def send(self, obj):
        self._queue.append(obj)

    def __repr__(self):
        if not isinstance(self.keys, tuple):
            return f'FeedbackProxy({repr(self.keys)})'
        else:
            return f'FeedbackProxy{self.keys}'


class FeedbackProxy():

    def feedback(self, keywords, obj, suggested=None):
        if keywords in self._pipes:
            if suggested is None:
                suggested = [self.kwds[k] for k in keywords]
            self._pipes[keywords].send((suggested, obj))
        else:
            warnings.warn(f'No feedback pipe for {keywords}', RuntimeWarning,
                          2)

    def feed(self, obj, **options):
        for tracker in self._trackers:
            tracker.feed(self, obj, **options)

    def store(self, obj, **options):
        self.feed(obj, store=True, **options)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_pipes']
        del state['_trackers']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._pipes = {}
        self._trackers = []


@dataclass
class StepStatus(FeedbackProxy):
    iteration: int = 0
    pos: tuple = ()
    index: tuple = ()
    kwds: dict = field(default_factory=dict)

    _pipes: dict = field(default_factory=dict, repr=False)
    _trackers: list = field(default_factory=list, repr=False)


@dataclass
class Begin(FeedbackProxy):
    level: int = 0
    iteration: int = 0
    pos: tuple = ()
    index: tuple = ()
    kwds: dict = field(default_factory=dict)

    _pipes: dict = field(default_factory=dict, repr=False)
    _trackers: list = field(default_factory=list, repr=False)

    def __repr__(self):
        return f'Begin(level={self.level}, kwds={self.kwds})'


@dataclass
class End(FeedbackProxy):
    level: int = 0
    iteration: int = 0
    pos: tuple = ()
    index: tuple = ()
    kwds: dict = field(default_factory=dict)

    _pipes: dict = field(default_factory=dict, repr=False)
    _trackers: list = field(default_factory=list, repr=False)

    def __repr__(self):
        return f'End(level={self.level}, kwds={self.kwds})'


class Tracker():

    def init(self, iter_dict: dict):
        pass

    def update(self, kwds: dict):
        return kwds

    def feed(self, step: StepStatus, obj: Any, **options):
        pass


def _call_func_with_kwds(func, args, kwds):
    sig = inspect.signature(func)
    for p in sig.parameters.values():
        if p.kind == p.VAR_KEYWORD:
            return func(*args, **kwds)
    kw = {
        k: v
        for k, v in kwds.items()
        if k in list(sig.parameters.keys())[len(args):]
    }
    return func(*args, **kw)


def _try_to_call(x, args, kwds):
    if callable(x):
        return _call_func_with_kwds(x, args, kwds)
    return x


def _get_current_iters(loops, level, kwds, pipes):
    keys, current = loops[level]
    limit = -1

    if isinstance(keys, str):
        keys = (keys, )
        current = (current, )
    elif isinstance(keys, tuple) and isinstance(
            current, tuple) and len(keys) == len(current):
        keys = tuple(k if isinstance(k, tuple) else (k, ) for k in keys)
    elif isinstance(keys, tuple) and not isinstance(current, tuple):
        current = (current, )
        if isinstance(keys[0], str):
            keys = (keys, )
    else:
        raise TypeError(f'Illegal keys {keys} on level {level}.')

    if not isinstance(keys, tuple):
        keys = (keys, )
    if not isinstance(current, tuple):
        current = (current, )

    iters = []
    for k, it in zip(keys, current):
        pipe = FeedbackPipe(k)
        if isinstance(it, OptimizerConfig):
            if limit < 0 or limit > it.max_iters:
                limit = it.max_iters
            it = it.cls(it.dimensions, *it.args, **it.kwds)
        else:
            it = iter(_try_to_call(it, (), kwds))

        iters.append((it, pipe))
        pipes[k] = pipe

    return keys, iters, pipes, limit


def _generate_kwds(keys, iters, kwds, iteration, limit):
    ret = {}
    for ks, it in zip(keys, iters):
        if isinstance(ks, str):
            ks = (ks, )
        if hasattr(it[0], 'ask') and hasattr(it[0], 'tell') and hasattr(
                it[0], 'get_result'):
            if limit > 0 and iteration >= limit - 1:
                value = _call_func_with_kwds(it[0].get_result, (), kwds).x
            else:
                value = _call_func_with_kwds(it[0].ask, (), kwds)
        else:
            value = next(it[0])
            if len(ks) == 1:
                value = (value, )
        ret.update(zip(ks, value))
    return ret


def _send_feedback(generator, feedback):
    if hasattr(generator, 'ask') and hasattr(generator, 'tell') and hasattr(
            generator, 'get_result'):
        generator.tell(*feedback)


def _feedback(iters):
    for generator, pipe in iters:
        for feedback in pipe():
            _send_feedback(generator, feedback)


def _args_generator(iters, kwds: dict, level: int, pos: tuple,
                    filter: Optional[callable], additional_kwds: dict,
                    trackers: list[Tracker], pipes: dict):
    if len(iters) == level and level > 0:
        kwds.update(
            {k: _try_to_call(v, (), kwds)
             for k, v in additional_kwds.items()})
        for tracker in trackers:
            kwds = tracker.update(kwds)
        if filter is None or _call_func_with_kwds(filter, (), kwds):
            yield StepStatus(pos=pos,
                             kwds=kwds,
                             _pipes=pipes,
                             _trackers=trackers)
        return

    keys, current_iters, pipes, limit = _get_current_iters(
        iters, level, kwds, pipes)

    for i in count():
        if limit > 0 and i >= limit:
            break
        try:
            kw = _generate_kwds(keys, current_iters, kwds, i, limit)
        except StopIteration:
            break
        yield Begin(level=level,
                    pos=pos + (i, ),
                    kwds=kwds | kw,
                    _pipes=pipes,
                    _trackers=trackers)
        yield from _args_generator(iters, kwds | kw, level + 1, pos + (i, ),
                                   filter, additional_kwds, trackers, pipes)
        yield End(level=level,
                  pos=pos + (i, ),
                  kwds=kwds | kw,
                  _pipes=pipes,
                  _trackers=trackers)
        _feedback(current_iters)


def _find_diff_pos(a: tuple, b: tuple):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i


def scan_iters(iters: dict[Union[str, tuple[str, ...]],
                           Union[Iterable, Callable, tuple[Iterable,
                                                           ...]]] = {},
               filter: Optional[Callable[..., bool]] = None,
               additional_kwds: dict = {},
               trackers: list = [],
               level_marker: bool = False) -> Iterable[StepStatus]:
    """
    Scan the given iterable of iterables.

    Parameters
    ----------
    iters : dict
        The map of iterables.
    filter : Callable[..., bool]
        A filter function that is called for each step.
        If it returns False, the step is skipped.
    additional_kwds : dict
        Additional keyword arguments that are passed to the iterables.

    Returns
    -------
    Iterable[StepStatus]
        An iterable of StepStatus objects.

    Examples
    --------
    >>> iters = {
    ...     'a': range(2),
    ...     'b': range(3),
    ... }
    >>> list(scan_iters(iters))
    [StepStatus(iteration=0, pos=(0, 0), index=(0, 0), kwds={'a': 0, 'b': 0}),
     StepStatus(iteration=1, pos=(0, 1), index=(0, 1), kwds={'a': 0, 'b': 1}),
     StepStatus(iteration=2, pos=(0, 2), index=(0, 2), kwds={'a': 0, 'b': 2}),
     StepStatus(iteration=3, pos=(1, 0), index=(1, 0), kwds={'a': 1, 'b': 0}),
     StepStatus(iteration=4, pos=(1, 1), index=(1, 1), kwds={'a': 1, 'b': 1}),
     StepStatus(iteration=5, pos=(1, 2), index=(1, 2), kwds={'a': 1, 'b': 2})]

    >>> iters = {
    ...     'a': range(2),
    ...     'b': range(3),
    ... }
    ... list(scan_iters(iters, lambda a, b: a < b))
    [StepStatus(iteration=0, pos=(0, 1), index=(0, 0), kwds={'a': 0, 'b': 1}),
     StepStatus(iteration=1, pos=(0, 2), index=(0, 1), kwds={'a': 0, 'b': 2}),
     StepStatus(iteration=2, pos=(1, 2), index=(1, 0), kwds={'a': 1, 'b': 2})]
    """

    if len(iters) == 0:
        return

    for tracker in trackers:
        tracker.init(iters)

    last_step = None
    index = ()
    iteration = count()

    for step in _args_generator(list(iters.items()),
                                kwds={},
                                level=0,
                                pos=(),
                                filter=filter,
                                additional_kwds=additional_kwds,
                                trackers=trackers,
                                pipes={}):
        if isinstance(step, (Begin, End)):
            if level_marker:
                if last_step is not None:
                    step.iteration = last_step.iteration
                yield step
            continue

        if last_step is None:
            i = 0
            index = (0, ) * len(step.pos)
        else:
            i = _find_diff_pos(last_step.pos, step.pos)
            index = tuple((j <= i) * n + (j == i) for j, n in enumerate(index))
        step.iteration = next(iteration)
        step.index = index
        yield step
        last_step = step


class Storage(Tracker):
    """
    A tracker that stores the results of the steps.

    Parameters
    ----------
    storage : dict
        The storage of the results.
    shape : tuple
        The shape of the results.
    ctime : datetime.datetime
        The creation time of the tracker.
    mtime : datetime.datetime
        The modification time of the tracker.
    """

    def __init__(self,
                 storage: dict = None,
                 shape: tuple = (),
                 save_kwds: Union[bool, Sequence[str]] = True,
                 frozen_keys: tuple = ()):
        self.ctime = datetime.utcnow()
        self.mtime = datetime.utcnow()
        self.storage = storage if storage is not None else {}
        self.cache = {}
        self.pos = {}
        self.timestamps = {}
        self.iteration = {}
        self._init_keys = list(self.storage.keys())
        self._frozen_keys = frozen_keys
        self._key_levels = ()
        self.shape = shape
        self.count = 0
        self.save_kwds = save_kwds
        self.queue = Queue()
        self._queue_buffer = None

    def init(self, iter_dict: dict):
        """
        Initialize the tracker.

        Parameters
        ----------
        iter_dict : dict
            The map of iterables.
        """
        from numpy import ndarray

        for level, (keys, iters) in enumerate(iter_dict.items()):
            self._key_levels = self._key_levels + ((keys, level), )
            if isinstance(keys, str):
                keys = (keys, )
                iters = (iters, )
            if (len(keys) > 1 and len(iters) == 1
                    and isinstance(iters[0], ndarray) and iters[0].ndim == 2
                    and iters[0].shape[1] == len(keys)):
                iters = iters[0]
                for i, key in enumerate(keys):
                    self.storage[key] = iters[:, i]
                    self._frozen_keys = self._frozen_keys + (key, )
                    self._init_keys.append(key)
                continue
            if not isinstance(iters, tuple) or len(keys) != len(iters):
                continue
            for key, iter in zip(keys, iters):
                if key not in self.storage and isinstance(
                        iter, (list, range, ndarray)):
                    self.storage[key] = iter
                    self._frozen_keys = self._frozen_keys + (key, )
                    self._init_keys.append(key)

    def feed(self, step: StepStatus, dataframe: dict, store=False, **options):
        """
        Feed the results of the step to the storage.

        Parameters
        ----------
        step : StepStatus
            The step.
        dataframe : dict
            The results of the step.
        """
        import numpy as np

        if not store:
            return
        self.mtime = datetime.utcnow()
        if not self.shape:
            self.shape = tuple([i + 1 for i in step.pos])
        else:
            self.shape = tuple(
                [max(i + 1, j) for i, j in zip(step.pos, self.shape)])
        if self.save_kwds:
            if isinstance(self.save_kwds, bool):
                kwds = step.kwds
            else:
                kwds = {
                    key: step.kwds.get(key, np.nan)
                    for key in self.save_kwds
                }
        else:
            kwds = {}
        self.queue.put_nowait(
            (step.iteration, step.pos, dataframe, kwds, self.mtime))

    def _append(self, iteration, pos, dataframe, kwds, now):
        for k, v in chain(kwds.items(), dataframe.items()):
            if k in self._frozen_keys:
                continue
            self.count += 1
            if k not in self.storage:
                self.storage[k] = [v]
                self.pos[k] = tuple([i] for i in pos)
                self.timestamps[k] = [now.timestamp()]
                self.iteration[k] = [iteration]
            else:
                self.storage[k].append(v)
                for i, l in zip(pos, self.pos[k]):
                    l.append(i)
                self.timestamps[k].append(now.timestamp())
                self.iteration[k].append(iteration)

    def _flush(self, block=False):
        if self._queue_buffer is not None:
            iteration, pos, fut, kwds, now = self._queue_buffer
            if fut.done() or block:
                self._append(iteration, pos, fut.result(), kwds, now)
                self._queue_buffer = None
            else:
                return
        while not self.queue.empty():
            iteration, pos, dataframe, kwds, now = self.queue.get()
            if isinstance(dataframe, Future):
                if not dataframe.done() and not block:
                    self._queue_buffer = (iteration, pos, dataframe, kwds, now)
                    return
                else:
                    self._append(iteration, pos, dataframe.result(), kwds, now)
            else:
                self._append(iteration, pos, dataframe, kwds, now)

    def _get_array(self, key, shape, count):
        import numpy as np

        data, data_shape, data_count = self.cache.get(key, (None, (), 0))
        if (data_shape, data_count) == (shape, count):
            return data
        tmp = np.asarray(self.storage[key])
        if data_shape != shape:
            data = np.full(shape + tmp.shape[1:], np.nan, dtype=tmp.dtype)
        data[self.pos[key]] = tmp
        self.cache[key] = (data, shape, count)
        return data

    def _get_part(self, key, skip):
        i = bisect.bisect_left(self.iteration[key], skip)
        pos = tuple(p[i:] for p in self.pos[key])
        iteration = self.iteration[key][i:]
        data = self.storage[key][i:]
        return data, iteration, pos

    def keys(self):
        """
        Get the keys of the storage.
        """
        self._flush()
        return self.storage.keys()

    def values(self):
        """
        Get the values of the storage.
        """
        self._flush()
        return [self[k] for k in self.storage]

    def items(self):
        """
        Get the items of the storage.
        """
        self._flush()
        return list(zip(self.keys(), self.values()))

    def get(self, key, default=_NODEFAULT, skip=None, block=False):
        """
        Get the value of the storage.
        """
        self._flush(block)
        if key in self._init_keys:
            return self.storage[key]
        elif key in self.storage:
            if skip is None:
                return self._get_array(key, self.shape, self.count)
            else:
                return self._get_part(key, skip)
        elif default is _NODEFAULT:
            raise KeyError(key)
        else:
            return default

    def __getitem__(self, key):
        return self.get(key)

    def __getstate__(self):
        self._flush()
        storage = dict(self.items())
        return {
            'storage': storage,
            'pos': self.pos,
            'timestamps': self.timestamps,
            'iteration': self.iteration,
            'shape': self.shape,
            'ctime': self.ctime,
            'mtime': self.mtime,
            '_init_keys': self._init_keys,
            '_frozen_keys': self._frozen_keys,
            '_key_levels': self._key_levels,
            'save_kwds': self.save_kwds,
        }
