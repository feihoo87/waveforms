from typing import Callable

import numpy as np
from numpy import e, inf, pi

NDIGITS: int = ...
__TypeIndex: int = ...
_baseFunc: dict[int, Callable] = ...
_derivativeBaseFunc: dict[int, Callable] = ...
_baseFunc_latex: dict[int, Callable] = ...

_zero: tuple[tuple, tuple] = ...


def _const(c: int | float | complex) -> tuple:
    pass


_one: tuple[tuple, tuple] = ...
_half: tuple[tuple, tuple] = ...
_two: tuple[tuple, tuple] = ...
_pi: tuple[tuple, tuple] = ...
_two_pi: tuple[tuple, tuple] = ...
_half_pi: tuple[tuple, tuple] = ...


def is_const(x: tuple[tuple, tuple]) -> bool:
    pass


def basic_wave(Type, *args, shift=0) -> tuple[tuple, tuple]:
    pass


def mul(x: tuple[tuple, tuple], y: tuple[tuple, tuple]) -> tuple[tuple, tuple]:
    pass


def add(x: tuple[tuple, tuple], y: tuple[tuple, tuple]) -> tuple[tuple, tuple]:
    pass


def shift(x: tuple[tuple, tuple], time: float) -> tuple[tuple, tuple]:
    pass


def pow(x: tuple[tuple, tuple], n: int) -> tuple[tuple, tuple]:
    pass


def calc_parts(bounds: tuple,
               seq: tuple,
               x: np.ndarray,
               function_lib: dict,
               min=-inf,
               max=inf) -> tuple[list[np.ndarray], type]:
    pass


def wave_sum(waves: list[tuple[tuple, tuple]]) -> tuple[tuple, tuple]:
    pass


def merge_waveform(b1: tuple, s1: tuple, b2: tuple, s2: tuple,
                   oper) -> tuple[tuple, tuple]:
    pass


def _D(x: tuple[tuple, tuple]) -> tuple[tuple, tuple]:
    pass


def registerBaseFunc(func: Callable) -> int:
    pass


def packBaseFunc() -> bytes:
    pass


def updateBaseFunc(buf: bytes):
    pass


def registerDerivative(Type: int, dFunc: Callable):
    pass


def registerBaseFuncLatex(Type: int, dFunc: Callable):
    pass


LINEAR: int = ...
GAUSSIAN: int = ...
ERF: int = ...
COS: int = ...
SINC: int = ...
EXP: int = ...
INTERP: int = ...
LINEARCHIRP: int = ...
EXPONENTIALCHIRP: int = ...
HYPERBOLICCHIRP: int = ...
COSH: int = ...
SINH: int = ...
DRAG: int = ...


def simplify(expr: tuple[tuple, tuple], eps: float) -> tuple[tuple, tuple]:
    pass


def filter(expr: tuple[tuple, tuple], low: float, high: float,
           eps: float) -> tuple[tuple, tuple]:
    pass
