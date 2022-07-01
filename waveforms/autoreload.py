from importlib import reload, import_module
from functools import wraps


class AutoReload():

    def __new__(cls, *args, **kwargs):
        mod = reload(import_module(cls.__module__))
        cls = getattr(mod, cls.__name__)
        return super().__new__(cls, *args, **kwargs)


def autoreload(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        mod = reload(import_module(func.__module__))
        func = getattr(mod, func.__name__)
        return func(*args, **kwargs)

    return wrapper