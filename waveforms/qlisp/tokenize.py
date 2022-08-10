from __future__ import annotations

import re
from typing import Any, NamedTuple


class Token(NamedTuple):
    type: str
    value: str
    line: int
    column: int


class Symbol():
    __slots__ = ('name', )

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        if isinstance(self.name, str):
            return self.name
        return repr(self.name)

    def __eq__(self, other):
        return isinstance(other, Symbol) and other.name == self.name


class Expression(tuple):
    pass


def atom(token):
    "Numbers become numbers; every other token is a symbol."
    for kind in [int, float, complex, Symbol]:
        try:
            return kind(token)
        except ValueError:
            continue
    raise ValueError()


Number = (int, float, complex
          )  # A Lisp Number is implemented as a Python int or float


def tokenize(code):
    keywords = {}
    token_specification = [
        ('BRACKET', r'[\(\)]'),
        ('STRING', r'\"([^\\\"]|\\.)*\"'),
        ('NEWLINE', r'\n'),  # Line endings
        ('SKIP', r'[ \t]+'),  # Skip over spaces and tabs
        ('ATOM', r'[A-Za-z0-9\.!@#$%^&\*/_\-\+:\?\|\<\>=]+'),  # Atom
        ('COMMENT', r';'),  # Comment
        ('QUOTE', r"'"),
        ('MISMATCH', r'.'),  # Any other character
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    line_num = 1
    line_start = 0
    in_comment = False
    for mo in re.finditer(tok_regex, code):
        kind = mo.lastgroup
        value = mo.group()
        column = mo.start() - line_start

        if kind == 'NEWLINE':
            line_start = mo.end()
            line_num += 1
            in_comment = False
            continue
        elif kind == 'COMMENT':
            in_comment = True
            continue
        elif in_comment or kind == 'SKIP':
            continue
        elif kind == 'STRING':
            value = value[1:-1]
        elif kind == 'BRACKET':
            kind = value
        elif kind == 'ATOM':
            value = atom(value)
        elif kind == 'MISMATCH':
            raise RuntimeError(f'{value!r} unexpected on line {line_num}')
        yield Token(kind, value, line_num, column)
