from importlib import reload, import_module
from functools import wraps


class AutoReload():

    def __new__(cls, *args, **kwargs):
        mod = reload(import_module(cls.__module__))
        cls = getattr(mod, cls.__name__)
        return super().__new__(cls, *args, **kwargs)


def autoreload(func):

    module_name = func.__module__
    func_name = func.__name__

    @wraps(func)
    def wrapper(*args, **kwargs):
        mod = reload(import_module(module_name))
        func = getattr(mod, func_name)
        return func._func_(*args, **kwargs)

    wrapper._func_ = func

    return wrapper