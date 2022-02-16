import inspect
from abc import ABC, abstractclassmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from itertools import count
from typing import Any, Callable, Iterable, Optional, Sequence, Type, Union


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


@dataclass
class StepStatus():
    iteration: int = 0
    level: int = 0
    pos: tuple = ()
    index: tuple = ()
    kwds: dict = field(default_factory=dict)

    _pipes: dict = field(default_factory=dict, repr=False)
    _trackers: list = field(default_factory=list, repr=False)

    def feedback(self, keywords, obj, suggested=None):
        if keywords in self._pipes:
            if suggested is None:
                suggested = [self.kwds[k] for k in keywords]
            self._pipes[keywords].send((suggested, obj))

    def feed(self, obj):
        for tracker in self._trackers:
            tracker.feed(self, obj)


class Tracker():

    def update(self, kwds: dict):
        return kwds

    def feed(self, step: StepStatus, obj: Any):
        pass


def _call_func_with_kwds(func, kwds):
    sig = inspect.signature(func)
    for p in sig.parameters.values():
        if p.kind == p.VAR_KEYWORD:
            return func(**kwds)
    kw = {k: v for k, v in kwds.items() if k in sig.parameters}
    return func(**kw)


def _try_to_call(x, kwds):
    if callable(x):
        return _call_func_with_kwds(x, kwds)
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
            it = iter(_try_to_call(it, kwds))

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
                value = it[0].get_result().x
            else:
                value = it[0].ask()
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
            {k: _try_to_call(v, kwds)
             for k, v in additional_kwds.items()})
        for tracker in trackers:
            kwds = tracker.update(kwds)
        if filter is None or _call_func_with_kwds(filter, kwds):
            yield StepStatus(level=level,
                             pos=pos,
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
        yield from _args_generator(iters, kwds | kw, level + 1, pos + (i, ),
                                   filter, additional_kwds, trackers, pipes)
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
               trackers: list = []) -> Iterable[StepStatus]:
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
    [StepStatus(iteration=0, level=1, pos=(0, 0), index=(0, 0), kwds={'a': 0, 'b': 0}),
     StepStatus(iteration=1, level=1, pos=(0, 1), index=(0, 1), kwds={'a': 0, 'b': 1}),
     StepStatus(iteration=2, level=1, pos=(0, 2), index=(0, 2), kwds={'a': 0, 'b': 2}),
     StepStatus(iteration=3, level=0, pos=(1, 0), index=(1, 0), kwds={'a': 1, 'b': 0}),
     StepStatus(iteration=4, level=1, pos=(1, 1), index=(1, 1), kwds={'a': 1, 'b': 1}),
     StepStatus(iteration=5, level=1, pos=(1, 2), index=(1, 2), kwds={'a': 1, 'b': 2})]

    >>> iters = {
    ...     'a': range(2),
    ...     'b': range(3),
    ... }
    ... list(scan_iters(iters, lambda a, b: a < b))
    [StepStatus(iteration=0, level=1, pos=(0, 1), index=(0, 0), kwds={'a': 0, 'b': 1}),
     StepStatus(iteration=1, level=1, pos=(0, 2), index=(0, 1), kwds={'a': 0, 'b': 2}),
     StepStatus(iteration=2, level=0, pos=(1, 2), index=(1, 0), kwds={'a': 1, 'b': 2})]
    """

    if len(iters) == 0:
        return

    last_pos = None
    index = ()
    for iteration, step in enumerate(
            _args_generator(list(iters.items()),
                            kwds={},
                            level=0,
                            pos=(),
                            filter=filter,
                            additional_kwds=additional_kwds,
                            trackers=trackers,
                            pipes={})):
        if last_pos is None:
            i = step.level - 1
            index = (0, ) * len(step.pos)
        else:
            i = _find_diff_pos(last_pos, step.pos)
            index = tuple((j <= i) * n + (j == i) for j, n in enumerate(index))
        step.iteration = iteration
        step.level = i
        step.index = index
        yield step
        last_pos = step.pos


class Storage(Tracker):

    def __init__(self, title):
        self.title = title
        self.ctime = datetime.utcnow()
        self.mtime = datetime.utcnow()
        self._storage = {}
        self.shape = ()

    def feed(self, step: StepStatus, dataframe: dict):
        self.mtime = datetime.utcnow()
        if not self.shape:
            self.shape = tuple([i + 1 for i in step.pos])
        else:
            self.shape = tuple(
                [max(i + 1, j) for i, j in zip(step.pos, self.shape)])
        for k, v in dataframe.items():
            if k not in self._storage:
                self._create_data(k, v, step.pos)
            else:
                self._fill_data(k, v, step.pos)

    def keys(self):
        return self._storage.keys()

    def values(self):
        slices = tuple(slice(None, i, None) for i in self.shape) + (..., )
        return [v[slices] for v in self._storage.values()]

    def items(self):
        return list(zip(self.keys(), self.values()))

    def __getitem__(self, key):
        slices = tuple(slice(None, i, None) for i in self.shape) + (..., )
        return self._storage[key][slices]

    def _create_data(self, key, value, pos):
        import numpy as np

        if isinstance(value, np.ndarray):
            size = self.shape + value.shape
            dtype = value.dtype
        else:
            size = self.shape
            if isinstance(value, (int, float, complex)):
                dtype = type(value)
            else:
                dtype = object
        self._storage[key] = np.full(size, np.nan, dtype=dtype)
        self._storage[key][pos + (..., )] = value

    def _fill_data(self, key, value, pos):
        import numpy as np

        shape = []
        slices = []
        data = self._storage[key]
        extend = False

        if isinstance(value, np.ndarray):
            frame_shape = value.shape
        else:
            frame_shape = ()

        for i, j in zip(data.shape, self.shape + frame_shape):
            slices.append(slice(None, i, None))
            while i < j:
                i *= 2
                extend = True
            shape.append(i)

        if extend:
            shape = tuple(shape)
            self._storage[key] = np.full(shape, np.nan, dtype=data.dtype)
            self._storage[key][tuple(slices)] = data
        if isinstance(value, np.ndarray):
            self._storage[key][pos + tuple(
                slice(None, i, None) for i in value.shape)] = value
        else:
            self._storage[key][pos] = value
