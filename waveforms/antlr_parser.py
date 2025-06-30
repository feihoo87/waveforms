"""
ANTLR 4 based waveform expression parser.
"""
import subprocess
import tempfile
import os
from pathlib import Path
from ast import literal_eval
from functools import lru_cache

from antlr4 import *
from antlr4.error.ErrorListener import ErrorListener

from . import multy_drag, waveform


class WaveformParseError(Exception):
    """Custom exception for waveform parsing errors."""
    pass


class WaveformErrorListener(ErrorListener):
    """Custom error listener for ANTLR parser."""
    
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        raise WaveformParseError(f"Syntax error at line {line}, column {column}: {msg}")


class WaveformVisitor:
    """Visitor class to evaluate waveform expressions."""
    
    def __init__(self):
        self.functions = [
            'D', 'chirp', 'const', 'cos', 'cosh', 'coshPulse', 'cosPulse', 'cut',
            'drag', 'drag_sin', 'drag_sinx', 'exp', 'gaussian', 'general_cosine',
            'hanning', 'interp', 'mixing', 'one', 'poly', 'samplingPoints', 'sign',
            'sin', 'sinc', 'sinh', 'square', 'step', 't', 'zero'
        ]
        self.constants = {'pi': waveform.pi, 'e': waveform.e, 'inf': waveform.inf}
        
    def get_function(self, name):
        """Get function from waveform or multy_drag modules."""
        for mod in [waveform, multy_drag]:
            try:
                return getattr(mod, name)
            except AttributeError:
                continue
        raise WaveformParseError(f"Unknown function '{name}'")
    
    def visit(self, ctx):
        """Visit a parse tree node."""
        method_name = f'visit_{type(ctx).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(ctx)
    
    def generic_visit(self, ctx):
        """Default visitor method."""
        if hasattr(ctx, 'children') and ctx.children:
            if len(ctx.children) == 1:
                return self.visit(ctx.children[0])
            else:
                return [self.visit(child) for child in ctx.children]
        return ctx.getText()
    
    def visit_ExprContext(self, ctx):
        """Visit expression context."""
        return self.visit(ctx.children[0])
    
    def visit_PowerExpressionContext(self, ctx):
        """Handle power expressions (** or ^)."""
        left = self.visit(ctx.expression(0))
        right = self.visit(ctx.expression(1))
        return left ** right
    
    def visit_MultiplyDivideExpressionContext(self, ctx):
        """Handle multiplication and division."""
        left = self.visit(ctx.expression(0))
        right = self.visit(ctx.expression(1))
        op = ctx.op.text
        if op == '*':
            return left * right
        else:  # op == '/'
            return left / right
    
    def visit_AddSubtractExpressionContext(self, ctx):
        """Handle addition and subtraction."""
        left = self.visit(ctx.expression(0))
        right = self.visit(ctx.expression(1))
        op = ctx.op.text
        if op == '+':
            return left + right
        else:  # op == '-'
            return left - right
    
    def visit_ShiftExpressionContext(self, ctx):
        """Handle shift expressions (<< and >>)."""
        left = self.visit(ctx.expression(0))
        right = self.visit(ctx.expression(1))
        op = ctx.op.text
        if op == '<<':
            return left << right
        else:  # op == '>>'
            return left >> right
    
    def visit_ParenthesesExpressionContext(self, ctx):
        """Handle parentheses expressions."""
        return self.visit(ctx.expression())
    
    def visit_UnaryMinusExpressionContext(self, ctx):
        """Handle unary minus expressions."""
        return -self.visit(ctx.expression())
    
    def visit_FunctionCallExpressionContext(self, ctx):
        """Handle function call expressions."""
        return self.visit(ctx.functionCall())
    
    def visit_ConstantExpressionContext(self, ctx):
        """Handle constant expressions."""
        const_name = ctx.CONSTANT().getText()
        return self.constants[const_name]
    
    def visit_NumberExpressionContext(self, ctx):
        """Handle number expressions."""
        return literal_eval(ctx.NUMBER().getText())
    
    def visit_StringExpressionContext(self, ctx):
        """Handle string expressions."""
        return literal_eval(ctx.STRING().getText())
    
    def visit_ListExpressionContext(self, ctx):
        """Handle list expressions."""
        return self.visit(ctx.list())
    
    def visit_TupleExpressionContext(self, ctx):
        """Handle tuple expressions."""
        return self.visit(ctx.tuple())
    
    def visit_IdentifierExpressionContext(self, ctx):
        """Handle identifier expressions."""
        # This could be a variable reference, but for now we'll raise an error
        # as the original implementation doesn't support variables
        var_name = ctx.ID().getText()
        raise WaveformParseError(f"Unknown identifier '{var_name}'")
    
    def visit_NoArgFunctionContext(self, ctx):
        """Handle function calls with no arguments."""
        func_name = ctx.ID().getText()
        func = self.get_function(func_name)
        return func()
    
    def visit_ArgsFunctionContext(self, ctx):
        """Handle function calls with positional arguments."""
        func_name = ctx.ID().getText()
        func = self.get_function(func_name)
        args = self.visit(ctx.args())
        return func(*args)
    
    def visit_KwargsFunctionContext(self, ctx):
        """Handle function calls with keyword arguments."""
        func_name = ctx.ID().getText()
        func = self.get_function(func_name)
        kwargs = self.visit(ctx.kwargs())
        return func(**kwargs)
    
    def visit_ArgsKwargsFunctionContext(self, ctx):
        """Handle function calls with both positional and keyword arguments."""
        func_name = ctx.ID().getText()
        func = self.get_function(func_name)
        args = self.visit(ctx.args())
        kwargs = self.visit(ctx.kwargs())
        return func(*args, **kwargs)
    
    def visit_ArgsContext(self, ctx):
        """Handle argument lists."""
        return [self.visit(expr) for expr in ctx.expression()]
    
    def visit_KwargsContext(self, ctx):
        """Handle keyword argument lists."""
        kwargs = {}
        for kwarg in ctx.kwarg():
            key, value = self.visit(kwarg)
            kwargs[key] = value
        return kwargs
    
    def visit_KwargContext(self, ctx):
        """Handle individual keyword arguments."""
        key = ctx.ID().getText()
        value = self.visit(ctx.expression())
        return key, value
    
    def visit_ListContext(self, ctx):
        """Handle list literals."""
        if ctx.expression():
            return [self.visit(expr) for expr in ctx.expression()]
        return []
    
    def visit_TupleContext(self, ctx):
        """Handle tuple literals."""
        if len(ctx.expression()) == 1:
            return (self.visit(ctx.expression(0)),)
        return tuple(self.visit(expr) for expr in ctx.expression())


def _generate_antlr_parser():
    """Generate ANTLR parser files if needed."""
    current_dir = Path(__file__).parent
    grammar_file = current_dir / "Waveform.g4"
    lexer_file = current_dir / "WaveformLexer.py"
    parser_file = current_dir / "WaveformParser.py"
    
    # Check if parser files exist and are newer than grammar file
    if (lexer_file.exists() and parser_file.exists() and 
        lexer_file.stat().st_mtime > grammar_file.stat().st_mtime and
        parser_file.stat().st_mtime > grammar_file.stat().st_mtime):
        return
    
    # Generate ANTLR files
    try:
        result = subprocess.run([
            "antlr4", "-Dlanguage=Python3", "-o", str(current_dir), str(grammar_file)
        ], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # Fall back to java command if antlr4 command is not available
        try:
            antlr_jar = os.environ.get('ANTLR_JAR', 'antlr-4.11.1-complete.jar')
            result = subprocess.run([
                "java", "-jar", antlr_jar, "-Dlanguage=Python3", 
                "-o", str(current_dir), str(grammar_file)
            ], capture_output=True, text=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise WaveformParseError(
                "Failed to generate ANTLR parser. Please install ANTLR4 or set ANTLR_JAR environment variable."
            )


def parse_waveform_expression(expr: str) -> waveform.Waveform:
    """Parse a waveform expression using ANTLR4."""
    # For now, raise an error to indicate ANTLR parser is not yet ready
    # This will cause the fallback to PLY parser in wave_eval
    raise WaveformParseError("ANTLR parser not yet fully implemented - falling back to PLY parser") 