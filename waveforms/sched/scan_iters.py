import inspect
from collections import deque
from itertools import chain, count
from typing import Callable, Iterable, NamedTuple, Optional, Type, Union

from skopt import Optimizer


class OptimizerConfig(NamedTuple):
    cls: Type[Optimizer] = Optimizer
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
    suggested_keys: tuple[str] = ()
    suggested: tuple = ()
    optimized: bool = False
    iteration: int = 0
    pipe: FeedbackPipe = None


class StepStatus(NamedTuple):
    pos: tuple = ()
    kwds: dict = {}
    iteration: int = 0
    index: tuple = ()
    optimizer_status: tuple = ()


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


def _args_generator(iters: dict,
                    kwds: dict = {},
                    pos=(),
                    optimizer_status=(),
                    filter=None):

    if len(iters) == 0:
        if filter is None or _call_func_with_kwds(filter, kwds):
            yield StepStatus(pos, kwds, 0, (), optimizer_status)
        return

    iters = iters.copy()
    keys, current_iters = iters.popitem()

    if _is_optimize_step(keys, current_iters):
        opts = current_iters
        yield from _opt_generator(iters, kwds, pos, optimizer_status, keys,
                                  opts, filter)
        return

    if not _is_multi_step(keys, current_iters):
        keys = (keys, )
        current_iters = (current_iters, )
    iter_list = [_try_to_call(it, kwds) for it in current_iters]
    for i, values in enumerate(zip(*iter_list)):
        yield from _args_generator(iters, kwds | dict(zip(keys, values)),
                                   pos + (i, ), optimizer_status, filter)


def _opt_generator(iters: dict,
                   kwds: dict,
                   pos: tuple[int],
                   optimizer_status: tuple,
                   opt_keys: tuple[str, ...],
                   opts: Union[OptimizerConfig, tuple[OptimizerConfig, ...]],
                   filter=None):

    pipe = FeedbackPipe(opt_keys)
    opts = (opts, ) if isinstance(opts, OptimizerConfig) else opts
    max_opt_iter = max(o.max_iters for o in opts)
    opts = [o.cls(o.dimensions, *o.args, **o.kwds) for o in opts]

    optimized = False

    for i in count():
        suggested = tuple(tuple(opt.ask()) for opt in opts)
        kw = dict(zip(opt_keys, chain(*suggested)))
        if i >= max_opt_iter:
            optimized = True
            try:
                result = tuple(opt.get_result().x for opt in opts)
                kw = dict(zip(opt_keys, chain(*result)))
            except:
                pass
        o = OptimizerStatus(opt_keys, suggested, optimized, i, pipe)
        yield from _args_generator(iters,
                                   kwds | kw,
                                   pos + (i, ),
                                   optimizer_status + (o, ),
                                   filter=filter)
        if optimized:
            break
        for feedback in pipe():
            if len(opts) == 1:
                suggested, fun = feedback
                opts[0].tell(suggested, fun)
            else:
                for opt, (suggested, fun) in zip(opts, feedback):
                    opt.tell(suggested, fun)


def _find_diff_pos(a: tuple, b: tuple):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i


def scan_iters(
        iters: dict[Union[str, tuple[str, ...]], Union[Iterable, Callable,
                                                       tuple[Iterable, ...]]],
        filter: Optional[Callable[..., bool]] = None) -> Iterable[StepStatus]:
    """
    Scan the given iterable of iterables.

    Parameters
    ----------
    iters : dict
        The map of iterables.
    filter : Callable[..., bool]
        A filter function that is called for each step.
        If it returns False, the step is skipped.

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
    [StepStatus(pos=(0, 0), kwds={'a': 0, 'b': 0}, iteration=0, index=(0, 0),
                optimizer_status=()),
     StepStatus(pos=(0, 1), kwds={'a': 0, 'b': 1}, iteration=0, index=(0, 1),
                optimizer_status=()),
     StepStatus(pos=(0, 2), kwds={'a': 0, 'b': 2}, iteration=0, index=(0, 2),
                optimizer_status=()),
     StepStatus(pos=(1, 0), kwds={'a': 1, 'b': 0}, iteration=1, index=(1, 0),
                optimizer_status=()),
     StepStatus(pos=(1, 1), kwds={'a': 1, 'b': 1}, iteration=1, index=(1, 1),
                optimizer_status=()),
     StepStatus(pos=(1, 2), kwds={'a': 1, 'b': 2}, iteration=1, index=(1, 2),
                optimizer_status=())]

    >>> iters = {
    >>>     'a': range(2),
    >>>     'b': range(3),
    >>> }
    >>> list(scan_iters(iters, lambda a, b: a < b))
    [StepStatus(pos=(0, 1), kwds={'a': 0, 'b': 1}, iteration=0, index=(0, 0),
                optimizer_status=()),
     StepStatus(pos=(0, 2), kwds={'a': 0, 'b': 2}, iteration=0, index=(0, 1),
                optimizer_status=()),
     StepStatus(pos=(1, 2), kwds={'a': 1, 'b': 2}, iteration=1, index=(1, 0),
                optimizer_status=())]
    """

    if len(iters) == 0:
        return

    iters = {k: iters[k] for k in reversed(iters)}

    last_pos = None
    index = ()
    for iteration, step in enumerate(_args_generator(iters, filter=filter)):
        if last_pos is None:
            index = step.pos
        else:
            i = _find_diff_pos(last_pos, step.pos)
            index = tuple((j <= i) * n + (j == i) for j, n in enumerate(index))
        yield StepStatus(step.pos, step.kwds, iteration, index,
                         step.optimizer_status)
        last_pos = step.pos
