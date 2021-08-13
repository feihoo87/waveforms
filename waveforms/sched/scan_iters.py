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


class FeedbackProxy():
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
    proxy: FeedbackProxy = None


class StepStatus(NamedTuple):
    pos: tuple = ()
    kwds: dict = {}
    iteration: int = 0
    index: tuple = ()
    optimizer_status: tuple = ()


def _is_multi_step(keys, iters):
    return isinstance(keys, tuple) and len(keys) > 1 and not isinstance(
        iters, OptimizerConfig)


def _is_optimize_step(keys, iters):
    return isinstance(iters, OptimizerConfig) or isinstance(
        iters, tuple) and isinstance(iters[0], OptimizerConfig)


def _call_func_with_kwds(func, kwds):
    sig = inspect.signature(func)
    kw = {k: v for k, v in kwds.items() if k in sig.parameters}
    return func(**kw)


def _args_generator(iters: dict,
                    kwds: dict = {},
                    pos=(),
                    counter=None,
                    optimizer_status=(),
                    filter=None):
    if counter is None:
        counter = count()

    if len(iters) == 0:
        if filter is None or _call_func_with_kwds(filter, kwds):
            yield StepStatus(pos, kwds, next(counter), (), optimizer_status)
        return

    iters = iters.copy()
    keys, current_iters = iters.popitem()
    if _is_optimize_step(keys, current_iters):
        opts = current_iters
        yield from _opt_generator(iters, kwds, pos, counter, optimizer_status,
                                  keys, opts, filter)
    else:
        if not _is_multi_step(keys, current_iters):
            keys = (keys, )
            current_iters = (current_iters, )
        iter_list = []
        for current_iter in current_iters:
            if callable(current_iter):
                iter_list.append(_call_func_with_kwds(current_iter, kwds))
            else:
                iter_list.append(current_iter)
        for i, a in enumerate(zip(*iter_list)):
            yield from _args_generator(iters, kwds | dict(zip(keys, a)),
                                       pos + (i, ), counter, optimizer_status,
                                       filter)


def _opt_generator(iters: dict,
                   kwds: dict,
                   pos: tuple[int],
                   counter: Iterable,
                   optimizer_status: tuple,
                   opt_keys: tuple[str, ...],
                   opts: Union[OptimizerConfig, tuple[OptimizerConfig, ...]],
                   filter=None):

    proxy = FeedbackProxy(opt_keys)
    opts = (opts, ) if isinstance(opts, OptimizerConfig) else opts
    max_opt_iter = max(o.max_iters for o in opts)
    opts = [o.cls(o.dimensions, *o.args, **o.kwds) for o in opts]

    for i in count():
        suggested = tuple(tuple(opt.ask()) for opt in opts)
        kw = dict(zip(opt_keys, chain(*suggested)))
        if i >= max_opt_iter:
            o = OptimizerStatus(opt_keys, suggested, True, i, proxy)
            try:
                kw = dict(
                    zip(opt_keys,
                        chain(*(opt.get_result().x for opt in opts))))
            except:
                pass
            yield from _args_generator(iters,
                                       kwds | kw,
                                       pos + (i, ),
                                       counter,
                                       optimizer_status + (o, ),
                                       filter=filter)
            break
        else:
            o = OptimizerStatus(opt_keys, suggested, False, i, proxy)
            yield from _args_generator(iters,
                                       kwds | kw,
                                       pos + (i, ),
                                       counter,
                                       optimizer_status + (o, ),
                                       filter=filter)
        for feedback in proxy():
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
        iters: dict,
        filter: Optional[Callable[..., bool]] = None) -> Iterable[StepStatus]:
    iters = {k: iters[k] for k in reversed(iters)}

    last_pos = None
    index = ()
    for step in _args_generator(iters, filter=filter):
        if last_pos is None:
            index = step.pos
        else:
            i = _find_diff_pos(last_pos, step.pos)
            index = tuple((j <= i) * n + (j == i) for j, n in enumerate(index))
        yield StepStatus(step.pos, step.kwds, step.iteration, index,
                         step.optimizer_status)
        last_pos = step.pos
