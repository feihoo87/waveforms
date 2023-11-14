from typing import Any

__version__: str

version: tuple[int, int, int]

def pack(obj, fp, **options) -> None: ...
def packb(obj, **options) -> bytes: ...
def dump(obj, fp, **options) -> None: ...
def dumps(obj, **options) -> bytes: ...

def unpackb(s: bytes | bytearray, **options) -> Any: ...
def unpack(fp, **options) -> Any: ...
def loads(s: bytes | bytearray, **options) -> Any: ...
def load(fp, **options) -> Any: ...

class Ext:
    type: int
    data: bytes
    def __init__(self, type: int, data: bytes) -> None: ...
    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __hash__(self) -> int: ...

class InvalidString(bytes): ...

def ext_serializable(ext_type: int): ...

class PackException(Exception): ...
class UnpackException(Exception): ...
class UnsupportedTypeException(PackException): ...
class InsufficientDataException(UnpackException): ...
class InvalidStringException(UnpackException): ...
class UnsupportedTimestampException(UnpackException): ...
class ReservedCodeException(UnpackException): ...
class UnhashableKeyException(UnpackException): ...
class DuplicateKeyException(UnpackException): ...
KeyNotPrimitiveException = UnhashableKeyException
KeyDuplicateException = DuplicateKeyException

compatibility: bool