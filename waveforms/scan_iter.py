import inspect
from abc import ABC, abstractclassmethod
from collections import deque
from itertools import chain, count
from typing import (Any, Callable, Iterable, NamedTuple, Optional, Sequence,
                    Type, Union)


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


class OptimizerConfig(NamedTuple):
    cls: Type[BaseOptimizer]
    dimensions: list = []
    args: tuple = ()
    kwds: dict = {}
    max_iters: int = 100


class FeedbackPipe():
    __slots__ = (
        'opt_keys',
        '_queue',
    )

    def __init__(self, opt_keys):
        self.opt_keys = opt_keys
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
        return f'FeedbackProxy{self.opt_keys}'


class OptimizerStatus(NamedTuple):
    suggested_keys: tuple[tuple[str]] = ()
    suggested: tuple = ()
    optimized: bool = False
    iteration: int = 0
    pipe: FeedbackPipe = None


class StepStatus(NamedTuple):
    pos: tuple = ()
    kwds: dict = {}
    iteration: int = 0
    level: int = 0
    index: tuple = ()
    optimizer_status: tuple = ()

    def feedback(self,
                 keywords: tuple[str],
                 obj: Any,
                 suggested: Optional[Sequence] = None):
        for opt_st in self.optimizer_status:
            for i, k in enumerate(opt_st.suggested_keys):
                if k == keywords:
                    if suggested is None:
                        suggested = [self.kwds[name] for name in keywords]
                    opt_st.pipe.send((i, (suggested, obj)))
                    return


def _is_multi_step(keys, iters):
    return isinstance(keys, tuple) and not isinstance(iters, OptimizerConfig)


def _is_optimize_step(keys, iters):
    return isinstance(iters, OptimizerConfig) or isinstance(
        iters, tuple) and isinstance(iters[0], OptimizerConfig)


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


def _args_generator(iters: list[tuple[str, Iterable]],
                    filter: Optional[Callable[..., bool]] = None,
                    additional_kwds: dict = {},
                    kwds: dict = {},
                    pos=(),
                    optimizer_status=(),
                    level=0):

    if len(iters) == level:
        kwds.update(
            {k: _try_to_call(v, kwds)
             for k, v in additional_kwds.items()})
        if filter is None or _call_func_with_kwds(filter, kwds):
            yield StepStatus(pos, kwds, 0, level - 1, (), optimizer_status)
        return

    keys, current_iters = iters[level]

    if not _is_multi_step(keys, current_iters):
        current_iters = (current_iters, )
    if not isinstance(keys, tuple):
        keys = (keys, )
    current_iters = tuple(_try_to_call(it, kwds) for it in current_iters)

    if _is_optimize_step(keys, current_iters):
        opts = current_iters
        yield from _opt_generator(iters, filter, additional_kwds, kwds, pos,
                                  optimizer_status, keys, opts, level)
        return

    for i, values in enumerate(zip(*current_iters)):
        yield from _args_generator(iters, filter, additional_kwds,
                                   kwds | dict(zip(keys, values)), pos + (i, ),
                                   optimizer_status, level + 1)


def _opt_generator(iters: list[tuple[str, Iterable]],
                   filter: Optional[Callable[..., bool]],
                   additional_kwds: dict,
                   kwds: dict,
                   pos: tuple[int],
                   optimizer_status: tuple,
                   opt_keys: tuple[str, ...],
                   opt_configs: Union[OptimizerConfig, tuple[OptimizerConfig,
                                                             ...]],
                   level=0):

    opt_configs = (opt_configs, ) if isinstance(
        opt_configs, OptimizerConfig) else opt_configs
    max_opt_iter = max(o.max_iters for o in opt_configs)
    opt_key_groups, opts = [], []
    for o in opt_configs:
        opts.append(o.cls(o.dimensions, *o.args, **o.kwds))
        opt_key_groups.append(opt_keys[:len(o.dimensions)])
        opt_keys = opt_keys[len(o.dimensions):]
    opt_key_groups = tuple(opt_key_groups)
    pipe = FeedbackPipe(opt_key_groups)

    optimized = False

    for i in count():
        suggested = tuple(tuple(opt.ask()) for opt in opts)
        kw = {}
        for keys, values in zip(opt_key_groups, suggested):
            kw.update(dict(zip(keys, values)))
        if i >= max_opt_iter:
            optimized = True
            try:
                result = tuple(opt.get_result().x for opt in opts)
                kw = {}
                for keys, values in zip(opt_key_groups, result):
                    kw.update(dict(zip(keys, values)))
            except:
                pass
        o = OptimizerStatus(opt_key_groups, suggested, optimized, i, pipe)
        yield from _args_generator(iters,
                                   filter,
                                   additional_kwds,
                                   kwds | kw,
                                   pos + (i, ),
                                   optimizer_status + (o, ),
                                   level=level + 1)
        if optimized:
            break
        for i, feedback in pipe():
            suggested, fun = feedback
            opts[i].tell(suggested, fun)


def _find_diff_pos(a: tuple, b: tuple):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i


def scan_iters(iters: dict[Union[str, tuple[str, ...]],
                           Union[Iterable, Callable, tuple[Iterable,
                                                           ...]]] = {},
               filter: Optional[Callable[..., bool]] = None,
               additional_kwds: dict = {}) -> Iterable[StepStatus]:
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
    >>>     'a': range(2),
    >>>     'b': range(3),
    >>> }
    >>> list(scan_iters(iters))
    [StepStatus(pos=(0, 0), kwds={'a': 0, 'b': 0}, iteration=0, level=1,
                index=(0, 0), optimizer_status=()),
     StepStatus(pos=(0, 1), kwds={'a': 0, 'b': 1}, iteration=1, level=1,
                index=(0, 1), optimizer_status=()),
     StepStatus(pos=(0, 2), kwds={'a': 0, 'b': 2}, iteration=2, level=1,
                index=(0, 2), optimizer_status=()),
     StepStatus(pos=(1, 0), kwds={'a': 1, 'b': 0}, iteration=3, level=0,
                index=(1, 0), optimizer_status=()),
     StepStatus(pos=(1, 1), kwds={'a': 1, 'b': 1}, iteration=4, level=1,
                index=(1, 1), optimizer_status=()),
     StepStatus(pos=(1, 2), kwds={'a': 1, 'b': 2}, iteration=5, level=1,
                index=(1, 2), optimizer_status=())]

    >>> iters = {
    >>>     'a': range(2),
    >>>     'b': range(3),
    >>> }
    >>> list(scan_iters(iters, lambda a, b: a < b))
    [StepStatus(pos=(0, 1), kwds={'a': 0, 'b': 1}, iteration=0, level=1,
                index=(0, 0), optimizer_status=()),
     StepStatus(pos=(0, 2), kwds={'a': 0, 'b': 2}, iteration=1, level=1,
                index=(0, 1), optimizer_status=()),
     StepStatus(pos=(1, 2), kwds={'a': 1, 'b': 2}, iteration=2, level=0,
                index=(1, 0), optimizer_status=())]
    """

    if len(iters) == 0:
        return

    last_pos = None
    index = ()
    for iteration, step in enumerate(
            _args_generator(list(iters.items()),
                            filter=filter,
                            additional_kwds=additional_kwds)):
        if last_pos is None:
            i = step.level
            index = (0, ) * len(step.pos)
        else:
            i = _find_diff_pos(last_pos, step.pos)
            index = tuple((j <= i) * n + (j == i) for j, n in enumerate(index))
        yield StepStatus(step.pos, step.kwds, iteration, i, index,
                         step.optimizer_status)
        last_pos = step.pos
