import inspect
from collections import deque
from itertools import chain, count
from typing import Iterable, NamedTuple, Type, Union

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
    index: tuple = ()
    kwds: dict = {}
    iteration: int = 0
    optimizer_status: tuple = ()


def _is_multi_step(keys, iters):
    return isinstance(keys, tuple) and len(keys) > 1 and not isinstance(
        iters, OptimizerConfig)


def _is_optimize_step(keys, iters):
    return isinstance(iters, OptimizerConfig) or isinstance(
        iters, tuple) and isinstance(iters[0], OptimizerConfig)


def _args_generator(iters: dict,
                    kwds: dict = {},
                    index=(),
                    counter=None,
                    optimizer_status=()):
    if counter is None:
        counter = count()
    if len(iters) == 0:
        yield StepStatus(index, kwds, next(counter), optimizer_status)
        return

    iters = iters.copy()
    keys, current_iters = iters.popitem()
    if _is_optimize_step(keys, current_iters):
        opts = current_iters
        yield from _opt_generator(iters, kwds, index, counter,
                                  optimizer_status, keys, opts)
    else:
        if not _is_multi_step(keys, current_iters):
            keys = (keys, )
            current_iters = (current_iters, )
        iter_list = []
        for current_iter in current_iters:
            if callable(current_iter):
                sig = inspect.signature(current_iter)
                kw = {k: v for k, v in kwds.items() if k in sig.parameters}
                current_iter = current_iter(**kw)
                iter_list.append(current_iter)
            else:
                iter_list.append(current_iter)
        for i, a in enumerate(zip(*iter_list)):
            yield from _args_generator(iters, kwds | dict(zip(keys, a)),
                                       index + (i, ), counter,
                                       optimizer_status)


def _opt_generator(iters: dict, kwds: dict, index: tuple[int],
                   counter: Iterable, optimizer_status: tuple,
                   opt_keys: tuple[str, ...],
                   opts: Union[OptimizerConfig, tuple[OptimizerConfig, ...]]):
    def _is_optimized(suggested, last_suggested, max_opt_iter):
        return (last_suggested is not None and suggested == last_suggested
                or i >= max_opt_iter)

    proxy = FeedbackProxy(opt_keys)
    opts = (opts, ) if isinstance(opts, OptimizerConfig) else opts
    max_opt_iter = max(o.max_iters for o in opts)
    opts = [o.cls(o.dimensions, *o.args, **o.kwds) for o in opts]

    last_suggested = None
    for i in count():
        suggested = tuple(tuple(opt.ask()) for opt in opts)
        if _is_optimized(suggested, last_suggested, max_opt_iter):
            o = OptimizerStatus(opt_keys, suggested, True, i, proxy)
            yield from _args_generator(
                iters, kwds | dict(
                    zip(opt_keys, chain(*(opt.get_result().x
                                          for opt in opts)))), index + (i, ),
                counter, optimizer_status + (o, ))
            break
        else:
            last_suggested = suggested

            o = OptimizerStatus(opt_keys, suggested, False, i, proxy)
            yield from _args_generator(
                iters, kwds | dict(zip(opt_keys, chain(*suggested))),
                index + (i, ), counter, optimizer_status + (o, ))
        for feedback in proxy():
            if len(opts) == 1:
                suggested, fun = feedback
                opts[0].tell(suggested, fun)
            else:
                for opt, (suggested, fun) in zip(opts, feedback):
                    opt.tell(suggested, fun)


def scan_iters(iters: dict) -> Iterable[StepStatus]:
    iters = {k: iters[k] for k in reversed(iters)}
    yield from _args_generator(iters)
