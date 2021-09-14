from typing import Any, Callable, NamedTuple

from waveforms.dicttree import flattenDict

from ..qlisp import MeasurementTask, QLispCode


class COMMAND():
    """Commands for the executor"""
    __slots__ = ('address', 'value')

    def __init__(self, address: str, value: Any):
        self.address = address
        self.value = value


class READ(COMMAND):
    """Read a value from the scheduler"""
    def __init__(self, address: str):
        super().__init__(address, 'READ')

    def __repr__(self) -> str:
        return f"READ({self.address})"


class WRITE(COMMAND):
    def __repr__(self) -> str:
        return f"WRITE({self.address}, {self.value})"


class TRIG(COMMAND):
    """Trigger the system"""
    def __init__(self, address: str):
        super().__init__(address, 0)

    def __repr__(self) -> str:
        return f"TRIG({self.address})"


class SYNC(COMMAND):
    """Synchronization command"""
    def __init__(self, delay: float = 0):
        super().__init__('SYNC', delay)

    def __repr__(self) -> str:
        return f"SYNC({self.value})"


class PUSH(COMMAND):
    def __init__(self):
        super().__init__('PUSH', 0)

    def __repr__(self) -> str:
        return f"PUSH()"


class FREE(COMMAND):
    def __init__(self):
        super().__init__('FREE', 0)

    def __repr__(self) -> str:
        return f"FREE()"


CommandList = list[COMMAND]
DataMap = dict[str, dict]
RawData = Any
Result = dict


class Architecture(NamedTuple):
    name: str
    description: str
    assembly_code: Callable[[QLispCode], tuple[CommandList, DataMap]]
    assembly_data: Callable[[RawData, DataMap], Result]


general_architecture = Architecture(
    name='general',
    description='General architecture',
    assembly_code=lambda code: (
        [],
        {
            'arch': 'general'
        },
    ),
    assembly_data=lambda data, _: flattenDict(data),
)
