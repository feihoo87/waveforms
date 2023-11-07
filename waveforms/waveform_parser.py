import tempfile
from ast import literal_eval
from functools import lru_cache

import ply.lex as lex
import ply.yacc as yacc

from . import multy_drag, waveform


class _WaveLexer:
    """Waveform Lexer.
    """

    def __init__(self):
        """Create a PLY lexer."""
        self.lexer = lex.lex(module=self, debug=False)

    def input(self, data):
        self.lexer.input(data)

    def token(self):
        """Return the next token."""
        ret = self.lexer.token()
        return ret

    literals = r'=()[]<>,.+-/*^'
    functions = [
        'D', 'chirp', 'const', 'cos', 'cosh', 'coshPulse', 'cosPulse', 'cut',
        'drag', 'drag_sin', 'drag_sinx', 'exp', 'gaussian', 'general_cosine',
        'hanning', 'interp', 'mixing', 'one', 'poly', 'samplingPoints', 'sign',
        'sin', 'sinc', 'sinh', 'square', 'step', 't', 'zero'
    ]
    tokens = [
        'REAL', 'IMAG', 'INT', 'STRING', 'ID', 'LSHIFT', 'RSHIFT', 'POW',
        'CONST', 'FUNCTION'
    ]

    def t_ID(self, t):
        r'[a-zA-Z_][a-zA-Z_0-9]*'
        if t.value in ['pi', 'e', 'inf']:
            t.type = 'CONST'
            return t
        if t.value in self.functions:
            t.type = 'FUNCTION'
            return t
        else:
            return t

    def t_IMAG(self, t):
        r'((([0-9]+|([0-9]+)?\.[0-9]+|[0-9]+\.)[eE][+-]?[0-9]+)|(([0-9]+)?\.[0-9]+|[0-9]+\.)|[1-9][0-9]*|0)j'
        return t

    def t_REAL(self, t):
        r'(([0-9]+|([0-9]+)?\.[0-9]+|[0-9]+\.)[eE][+-]?[0-9]+)|(([0-9]+)?\.[0-9]+|[0-9]+\.)'
        return t

    def t_INT(self, t):
        r'[1-9][0-9]*|0'
        return t

    def t_STRING(self, t):
        r'(".*")|(\'.*\')'
        return t

    def t_LSHIFT(self, t):
        '<<'
        return t

    def t_RSHIFT(self, t):
        '>>'
        return t

    def t_POW(self, t):
        r'\*\*'
        return t

    def t_eof(self, _):
        return None

    t_ignore = ' \t\r\n'

    def t_error(self, t):
        raise SyntaxError("Unable to match any token rule, got -->%s<-- " %
                          t.value)


class _WaveParser:

    def __init__(self):
        self.lexer = _WaveLexer()
        self.tokens = self.lexer.tokens
        self.parse_dir = tempfile.mkdtemp(prefix='waveforms')
        self.precedence = (('left', 'RSHIFT', 'LSHIFT'), ('left', '+', '-'),
                           ('left', '*', '/'), ('left', 'POW',
                                                '^'), ('right', 'UMINUS'))
        self.parser = yacc.yacc(module=self,
                                debug=False,
                                outputdir=self.parse_dir)
        self.waveform = None

    def parse(self, data):
        #self.waveform = None
        self.parser.parse(data, lexer=self.lexer, debug=False)
        if self.waveform is None:
            raise SyntaxError("Uncaught exception in parser; " +
                              "see previous messages for details.")
        if isinstance(self.waveform, (float, int)):
            self.waveform = waveform.const(self.waveform)
        return self.waveform.simplify()

    def getFunction(self, name):
        for mod in [waveform, multy_drag]:
            try:
                return getattr(waveform, name)
            except AttributeError:
                pass
        raise SyntaxError(f"Unknown function '{name}'")

    # ---- Begin the PLY parser ----
    start = 'main'

    def p_main(self, p):
        """
        main : expression
        """
        self.waveform = p[1]

    def p_const(self, p):
        """
        expression : CONST
        """
        p[0] = {'pi': waveform.pi, 'e': waveform.e, 'inf': waveform.inf}[p[1]]

    def p_real_imag_int_string(self, p):
        """
        expression : REAL
                   | IMAG
                   | INT
                   | STRING
        """
        p[0] = literal_eval(p[1])

    def p_tuple_list(self, p):
        """
        expression : tuple
                   | list
        """
        p[0] = p[1]

    def p_expr_uminus(self, p):
        """
        expression : '-' expression %prec UMINUS
        """
        p[0] = -p[2]

    def p_function_call(self, p):
        """
        expression : FUNCTION '(' ')'
        """
        p[0] = self.getFunction(p[1])()

    def p_function_call_2(self, p):
        """
        expression :  FUNCTION '(' args ')'
        """
        p[0] = self.getFunction(p[1])(*p[3])

    def p_function_call_3(self, p):
        """
        expression :  FUNCTION '(' kwds ')'
        """
        p[0] = self.getFunction(p[1])(**p[3])

    def p_function_call_4(self, p):
        """
        expression :  FUNCTION '(' args ',' kwds ')'
        """
        p[0] = self.getFunction(p[1])(*p[3], **p[5])

    def p_bracket(self, p):
        """
        expression :  '(' expression ')'
        """
        p[0] = p[2]

    def p_binary_operators(self, p):
        """
        expression : expression '+' expression
                   | expression '-' expression
                   | expression '*' expression
                   | expression '/' expression
                   | expression LSHIFT expression
                   | expression RSHIFT expression
                   | expression '^' expression
                   | expression POW expression
        """
        if p[2] == '+':
            p[0] = p[1] + p[3]
        elif p[2] == '-':
            p[0] = p[1] - p[3]
        elif p[2] == '*':
            p[0] = p[1] * p[3]
        elif p[2] == '/':
            p[0] = p[1] / p[3]
        elif p[2] == '>>':
            p[0] = p[1] >> p[3]
        elif p[2] == '<<':
            p[0] = p[1] << p[3]
        elif p[2] == '^':
            p[0] = p[1]**p[3]
        else:
            p[0] = p[1]**p[3]

    def p_expr_list_2(self, p):
        """
        expr_list : expression ',' expression
        """
        p[0] = [p[1], p[3]]

    def p_expr_list_3(self, p):
        """
        expr_list : expr_list ',' expression
        """
        p[0] = [*p[1], p[3]]

    def p_tuple(self, p):
        """
        tuple : '(' expression ',' ')'
              | '(' expr_list ')'
        """
        if len(p) == 5:
            p[0] = (p[2], )
        else:
            p[0] = tuple(p[2])

    def p_list_1(self, p):
        """
        list : '[' expression ']'
        """
        p[0] = [p[2]]

    def p_list_2(self, p):
        """
        list : '[' expr_list ']'
        """
        p[0] = p[2]

    def p_args(self, p):
        """
        args : expression
             | args ',' expression
        """
        if len(p) == 2:
            p[0] = (p[1], )
        else:
            p[0] = p[1] + (p[3], )

    def p_kwds(self, p):
        """
        kwds : ID '=' expression
             | kwds ',' ID '=' expression
        """
        if len(p) == 4:
            p[0] = {p[1]: p[3]}
        else:
            kwds = {}
            kwds.update(p[1])
            kwds[p[3]] = p[5]
            p[0] = kwds
            # p[0] = p[1] | {p[3]: p[5]}   # only works on Python>=3.9

    def p_error(self, p):
        raise SyntaxError("Syntax error in input!")


_wave_parser = _WaveParser()


@lru_cache(maxsize=1024)
def wave_eval(expr: str) -> waveform.Waveform:
    try:
        return _wave_parser.parse(expr)
    except:
        raise SyntaxError(f"Illegal expression '{expr}'")
