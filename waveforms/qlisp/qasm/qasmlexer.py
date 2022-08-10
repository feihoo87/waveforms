# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
OPENQASM Lexer.
This is a wrapper around the PLY lexer to support the "include" statement
by creating a stack of lexers.
"""

import os

import numpy as np
import ply.lex as lex

from . import node
from .exceptions import QasmError


class QasmLexer:
    """OPENQASM Lexer.
    This is a wrapper around the PLY lexer to support the "include" statement
    by creating a stack of lexers.
    """

    # pylint: disable=invalid-name,missing-docstring
    # pylint: disable=attribute-defined-outside-init,bad-docstring-quotes

    def __mklexer__(self, filename):
        """Create a PLY lexer."""
        self.lexer = lex.lex(module=self, debug=False)
        self.filename = filename
        self.lineno = 1

        if filename:
            with open(filename) as ifile:
                self.data = ifile.read()
            self.lexer.input(self.data)

    def __init__(self, filename, lib=None):
        """Create the OPENQASM lexer."""
        self.__mklexer__(filename)
        self.stack = []
        self.lib = lib

    def input(self, data):
        """Set the input text data."""
        self.data = data
        self.lexer.input(data)

    def token(self):
        """Return the next token."""
        ret = self.lexer.token()
        return ret

    def pop(self):
        """Pop a PLY lexer off the stack."""
        self.lexer = self.stack.pop()
        self.filename = self.lexer.qasm_file
        self.lineno = self.lexer.qasm_line

    def push(self, filename):
        """Push a PLY lexer on the stack to parse filename."""
        self.lexer.qasm_file = self.filename
        self.lexer.qasm_line = self.lineno
        self.stack.append(self.lexer)
        self.__mklexer__(filename)

    # ---- Beginning of the PLY lexer ----
    literals = r'=()[]{};<>,.+-/*^"'
    reserved = {
        'barrier': 'BARRIER',
        'creg': 'CREG',
        'gate': 'GATE',
        'if': 'IF',
        'measure': 'MEASURE',
        'opaque': 'OPAQUE',
        'qreg': 'QREG',
        'pi': 'PI',
        'reset': 'RESET',
    }
    tokens = [
        'NNINTEGER',
        'REAL',
        'FORMAT',
        'ASSIGN',
        'MATCHES',
        'ID',
        'STRING',
    ] + list(reserved.values())

    def t_REAL(self, t):
        r'(([0-9]+|([0-9]+)?\.[0-9]+|[0-9]+\.)[eE][+-]?[0-9]+)|(([0-9]+)?\.[0-9]+|[0-9]+\.)'
        if np.iscomplex(t):
            return t.real
        else:
            return t

    def t_NNINTEGER(self, t):
        r'[1-9]+[0-9]*|0'
        t.value = int(t.value)
        return t

    def t_ASSIGN(self, t):
        '->'
        return t

    def t_MATCHES(self, t):
        '=='
        return t

    def t_STRING(self, t):
        r'\"([^\\\"]|\\.)*\"'
        return t

    def t_INCLUDE(self, _):
        'include'
        # Now eat up the next two tokens which must be
        # 1 - the name of the include file, and
        # 2 - a terminating semicolon
        #
        # Then push the current lexer onto the stack, create a new one from
        # the include file, and push it onto the stack.
        #
        # When we hit eof (the t_eof) rule, we pop.
        next_token = self.lexer.token()
        lineno = next_token.lineno
        if isinstance(next_token.value, str):
            incfile = next_token.value.strip('"')
        else:
            raise QasmError("Invalid include: must be a quoted string.")

        if self.lib is not None:
            fullpath = self.lib.getQasmLib(incfile)
            if fullpath is not None:
                incfile = fullpath

        next_token = self.lexer.token()
        if next_token is None or next_token.value != ';':
            raise QasmError('Invalid syntax, missing ";" at line', str(lineno))

        if not os.path.exists(incfile):
            raise QasmError(
                'Include file %s cannot be found, line %s, file %s' %
                (incfile, str(next_token.lineno), self.filename))
        self.push(incfile)
        return self.lexer.token()

    def t_FORMAT(self, t):
        r'OPENQASM\s+(\d+)\.(\d+)'
        return t

    def t_COMMENT(self, _):
        r'//.*'
        pass

    def t_ID(self, t):
        r'[a-zA-Z_][a-zA-Z0-9_]*'

        t.type = self.reserved.get(t.value, 'ID')
        if t.type == 'ID':
            t.value = node.Id(t.value, self.lineno, self.filename)
        return t

    def t_newline(self, t):
        r'\n+'
        self.lineno += len(t.value)
        t.lexer.lineno = self.lineno

    def t_eof(self, _):
        if self.stack:
            self.pop()
            return self.lexer.token()
        return None

    t_ignore = ' \t\r'

    def t_error(self, t):
        raise QasmError(
            "Unable to match any token rule, got -->%s<-- "
            "Check your OPENQASM source and any include statements." %
            t.value[0])
