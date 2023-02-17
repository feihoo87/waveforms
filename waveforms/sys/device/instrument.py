import itertools
import logging
import re
from functools import partial
from typing import Any, Callable, Literal

log = logging.getLogger(__name__)

Decorator = Callable[[Callable], Callable]


def action(key: str, method: Literal['get', 'set'] = 'get') -> Decorator:

    def decorator(func):
        func.__action__ = key, method
        return func

    return decorator


def get(key: str) -> Decorator:
    return action(key, 'get')


def set(key: str) -> Decorator:
    return action(key, 'set')


def _add_action(attrs: dict, key: str, method: str, func: Callable,
                doc: dict) -> None:
    if method == 'get':
        mapping = attrs['__get_actions__']
    elif method == 'set':
        mapping = attrs['__set_actions__']
    else:
        raise ValueError('Invalid method: ' + method)
    arguments = re.findall(r'\{(\w+)\}', key)
    doc[method][key] = func.__doc__
    for arg in arguments:
        if arg not in attrs:
            raise ValueError(
                f'Undefined section: {arg!r} in @action({key!r}, {method!r})')
    for values in itertools.product(*[attrs[arg] for arg in arguments]):
        kwds = dict(zip(arguments, values))
        mapping[key.format(**kwds)] = partial(func, **kwds)


def _build_docs(mapping: dict, attrs: dict) -> str:
    docs = []
    for key, doc in mapping.items():
        if not doc:
            doc = "No documentation."
        docs.append(f"key = \"{key}\"")
        lines = doc.strip().split('\n')
        docs.extend(lines)
        docs.append("")
    return '\n'.join(docs)


class InstrumentMeta(type):

    def __new__(cls, name, bases, attrs):
        attrs.setdefault('__get_actions__', {})
        attrs.setdefault('__set_actions__', {})
        doc = {'get': {}, 'set': {}}
        for attr in attrs.values():
            if hasattr(attr, '__action__'):
                key, method = attr.__action__
                _add_action(attrs, key, method, attr, doc)
        new_class = super().__new__(cls, name, bases, attrs)
        new_class.get.__doc__ = "Get\n\n" + _build_docs(doc['get'], attrs)
        new_class.set.__doc__ = "Set\n\n" + _build_docs(doc['set'], attrs)
        return new_class


class Instrument(metaclass=InstrumentMeta):
    __log__ = None

    def __init__(self):
        self.__status = {}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def log(self):
        if self.__log__ is None:
            return log
        else:
            return self.__log__

    def open(self, *args, **kwds) -> None:
        pass

    def close(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def get(self, key: str, default: Any = None) -> Any:
        self.log.info(f'Get {key!r}')
        if key in self.__get_actions__:
            result = self.__get_actions__[key](self)
            self.__status[key] = result
            return result
        else:
            return self.__status.get(key, default)

    def set(self, key: str, value: Any = None) -> None:
        self.log.info(f'Set {key!r} = {value!r}')
        self.__set_actions__[key](self, value)
        self.__status[key] = value


class VisaInstrument(Instrument):

    def open(self, resource_name) -> None:
        import pyvisa
        rm = pyvisa.ResourceManager()
        self.resource = rm.open_resource(resource_name)

    def close(self) -> None:
        self.resource.close()

    def reset(self) -> None:
        self.resource.write('*RST')

    @get('IDN')
    def get_idn(self) -> str:
        """Get instrument identification."""
        return self.resource.query('*IDN?')
