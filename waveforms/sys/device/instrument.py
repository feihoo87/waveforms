import itertools
import logging
import re
from functools import partial
from typing import Any, Callable, Literal, NamedTuple

log = logging.getLogger(__name__)

Decorator = Callable[[Callable], Callable]


def action(key: str,
           method: Literal['get', 'set'] = 'get',
           **kwds) -> Decorator:

    if any(c in key for c in ",()[]{}<>"):
        raise ValueError('Invalid key: ' + key)

    def decorator(func):
        func.__action__ = key, method, kwds
        return func

    return decorator


def get(key: str, **kwds) -> Decorator:
    return action(key, 'get', **kwds)


def set(key: str, **kwds) -> Decorator:
    return action(key, 'set', **kwds)


class _Exclusion(NamedTuple):
    keys: list


def exclude(sections: list):
    return _Exclusion(sections)


def _add_action(attrs: dict, key: str, method: str, func: Callable, doc: dict,
                sections: dict) -> None:
    if method == 'get':
        mapping = attrs['__get_actions__']
    elif method == 'set':
        mapping = attrs['__set_actions__']
    else:
        raise ValueError('Invalid method: ' + method)
    arguments = re.findall(r'\{(\w+)\}', key)
    doc[method][key] = func.__doc__
    matrix = {}
    for arg in arguments:
        if arg in attrs or arg in sections and not isinstance(
                sections[arg],
                _Exclusion) or arg in attrs and arg in sections and isinstance(
                    sections[arg], _Exclusion):
            raise ValueError(
                f'Undefined section: {arg!r} in @action({key!r}, {method!r})')
        if arg in sections and not isinstance(sections[arg], _Exclusion):
            matrix[arg] = sections[arg]
        else:
            if arg in sections:
                matrix[arg] = [
                    k for k in attrs[arg] if k not in sections[arg].keys
                ]
            else:
                matrix[arg] = attrs[arg]
    for values in itertools.product(*[matrix[arg] for arg in arguments]):
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
                key, method, kwds = attr.__action__
                _add_action(attrs, key, method, attr, doc, kwds)
        new_class = super().__new__(cls, name, bases, attrs)
        new_class.get.__doc__ = "Get\n\n" + _build_docs(doc['get'], attrs)
        new_class.set.__doc__ = "Set\n\n" + _build_docs(doc['set'], attrs)
        return new_class


class BaseInstrument(metaclass=InstrumentMeta):
    __log__ = None

    def __init__(self):
        self._status = {}

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

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def reset(self) -> None:
        self._status.clear()

    def get(self, key: str, default: Any = None) -> Any:
        self.log.info(f'Get {key!r}')
        if key in self.__get_actions__:
            result = self.__get_actions__[key](self)
            self._status[key] = result
            return result
        else:
            return self._status.get(key, default)

    def set(self, key: str, value: Any = None) -> None:
        self.log.info(f'Set {key!r} = {value!r}')
        self.__set_actions__[key](self, value)
        self._status[key] = value


class VisaInstrument(BaseInstrument):

    def __init__(self, resource_name):
        super().__init__()
        self.resource_name = resource_name

    def open(self) -> None:
        import pyvisa
        rm = pyvisa.ResourceManager()
        self.resource = rm.open_resource(self.resource_name)

    def close(self) -> None:
        self.resource.close()

    def reset(self) -> None:
        super().reset()
        self.resource.write('*RST')

    @get('idn')
    def get_idn(self) -> str:
        """Get instrument identification."""
        return self.resource.query('*IDN?')

    @get('opc')
    def get_opc(self) -> bool:
        """Get operation complete."""
        return bool(int(self.resource.query('*OPC?')))

    @get('errors')
    def get_errors(self) -> list[str]:
        """Get error queue."""
        errors = []
        while True:
            error = self.resource.query('SYST:ERR?')
            error_code = int(error.split(',')[0])
            if error_code == 0:
                break
            errors.append(error)
        return errors

    @set('timeout')
    def set_timeout(self, value: float) -> None:
        """Set timeout in seconds."""
        self.resource.timeout = round(value * 1000)

    @get('timeout')
    def get_timeout(self) -> float:
        """Get timeout in seconds."""
        return self.resource.timeout / 1000
