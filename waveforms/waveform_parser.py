import os
import subprocess
from ast import literal_eval
from functools import lru_cache
from pathlib import Path

from antlr4 import *
from antlr4.error.ErrorListener import ErrorListener

from . import multy_drag, waveform


class WaveformParseError(Exception):
    """Custom exception for waveform parsing errors."""
    pass


class WaveformErrorListener(ErrorListener):
    """Custom error listener for ANTLR parser."""

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        raise WaveformParseError(
            f"Syntax error at line {line}, column {column}: {msg}")


class WaveformVisitor:
    """Visitor class to evaluate waveform expressions."""

    def __init__(self):
        self.functions = [
            'D', 'chirp', 'const', 'cos', 'cosh', 'coshPulse', 'cosPulse',
            'cut', 'drag', 'drag_sin', 'drag_sinx', 'exp', 'gaussian',
            'general_cosine', 'hanning', 'interp', 'mixing', 'one', 'poly',
            'samplingPoints', 'sign', 'sin', 'sinc', 'sinh', 'square', 'step',
            't', 'zero'
        ]
        self.constants = {
            'pi': waveform.pi,
            'e': waveform.e,
            'inf': waveform.inf
        }

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
        method_name = f'visit{type(ctx).__name__}'
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

    def visitExprContext(self, ctx):
        """Visit expression context."""
        if ctx.assignment():
            return self.visit(ctx.assignment())
        else:
            return self.visit(ctx.expression())

    def visitAssignmentContext(self, ctx):
        """Visit assignment context - not supported in waveform expressions."""
        raise WaveformParseError("Assignment expressions are not supported")

    def visitPowerExpressionContext(self, ctx):
        """Handle power expressions (** or ^)."""
        left = self.visit(ctx.expression(0))
        right = self.visit(ctx.expression(1))
        return left**right

    def visitMultiplyDivideExpressionContext(self, ctx):
        """Handle multiplication and division."""
        left = self.visit(ctx.expression(0))
        right = self.visit(ctx.expression(1))
        op = ctx.getChild(1).getText()  # Get operator text
        if op == '*':
            return left * right
        else:  # op == '/'
            return left / right

    def visitAddSubtractExpressionContext(self, ctx):
        """Handle addition and subtraction."""
        left = self.visit(ctx.expression(0))
        right = self.visit(ctx.expression(1))
        op = ctx.getChild(1).getText()  # Get operator text
        if op == '+':
            return left + right
        else:  # op == '-'
            return left - right

    def visitShiftExpressionContext(self, ctx):
        """Handle shift expressions (<< and >>)."""
        left = self.visit(ctx.expression(0))
        right = self.visit(ctx.expression(1))
        op = ctx.getChild(1).getText()  # Get operator text
        if op == '<<':
            return left << right
        else:  # op == '>>'
            return left >> right

    def visitParenthesesExpressionContext(self, ctx):
        """Handle parentheses expressions."""
        return self.visit(ctx.expression())

    def visitUnaryMinusExpressionContext(self, ctx):
        """Handle unary minus expressions."""
        return -self.visit(ctx.expression())

    def visitFunctionCallExpressionContext(self, ctx):
        """Handle function call expressions."""
        return self.visit(ctx.functionCall())

    def visitConstantExpressionContext(self, ctx):
        """Handle constant expressions."""
        const_name = ctx.CONSTANT().getText()
        return self.constants[const_name]

    def visitNumberExpressionContext(self, ctx):
        """Handle number expressions."""
        return literal_eval(ctx.NUMBER().getText())

    def visitStringExpressionContext(self, ctx):
        """Handle string expressions."""
        return literal_eval(ctx.STRING().getText())

    def visitListExpressionContext(self, ctx):
        """Handle list expressions."""
        return self.visit(ctx.list_())

    def visitTupleExpressionContext(self, ctx):
        """Handle tuple expressions."""
        return self.visit(ctx.tuple_())

    def visitIdentifierExpressionContext(self, ctx):
        """Handle identifier expressions."""
        # This could be a variable reference, but for now we'll raise an error
        # as the original implementation doesn't support variables
        var_name = ctx.ID().getText()
        raise WaveformParseError(f"Unknown identifier '{var_name}'")

    def visitNoArgFunctionContext(self, ctx):
        """Handle function calls with no arguments."""
        func_name = ctx.ID().getText()
        func = self.get_function(func_name)
        return func()

    def visitArgsFunctionContext(self, ctx):
        """Handle function calls with positional arguments."""
        func_name = ctx.ID().getText()
        func = self.get_function(func_name)
        args = self.visit(ctx.args())
        return func(*args)

    def visitKwargsFunctionContext(self, ctx):
        """Handle function calls with keyword arguments."""
        func_name = ctx.ID().getText()
        func = self.get_function(func_name)
        kwargs = self.visit(ctx.kwargs())
        return func(**kwargs)

    def visitArgsKwargsFunctionContext(self, ctx):
        """Handle function calls with both positional and keyword arguments."""
        func_name = ctx.ID().getText()
        func = self.get_function(func_name)
        args = self.visit(ctx.args())
        kwargs = self.visit(ctx.kwargs())
        return func(*args, **kwargs)

    def visitArgsContext(self, ctx):
        """Handle argument lists."""
        return [self.visit(expr) for expr in ctx.expression()]

    def visitKwargsContext(self, ctx):
        """Handle keyword argument lists."""
        kwargs = {}
        for kwarg in ctx.kwarg():
            key, value = self.visit(kwarg)
            kwargs[key] = value
        return kwargs

    def visitKwargContext(self, ctx):
        """Handle individual keyword arguments."""
        key = ctx.ID().getText()
        value = self.visit(ctx.expression())
        return key, value

    def visitListContext(self, ctx):
        """Handle list literals."""
        if ctx.expression():
            return [self.visit(expr) for expr in ctx.expression()]
        return []

    def visitTupleContext(self, ctx):
        """Handle tuple literals."""
        expressions = ctx.expression()
        if len(expressions) == 1:
            return (self.visit(expressions[0]), )
        return tuple(self.visit(expr) for expr in expressions)


def _generate_antlr_parser():
    """Generate ANTLR parser files if needed."""
    current_dir = Path(__file__).parent
    grammar_file = current_dir / "Waveform.g4"
    lexer_file = current_dir / "WaveformLexer.py"
    parser_file = current_dir / "WaveformParser.py"

    # Check if parser files exist and are newer than grammar file
    if (lexer_file.exists() and parser_file.exists()
            and lexer_file.stat().st_mtime > grammar_file.stat().st_mtime
            and parser_file.stat().st_mtime > grammar_file.stat().st_mtime):
        return

    # Generate ANTLR files
    try:
        result = subprocess.run([
            "antlr4", "-Dlanguage=Python3",
            str(grammar_file)
        ],
                                cwd=str(current_dir),
                                capture_output=True,
                                text=True,
                                check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # Fall back to java command if antlr4 command is not available
        try:
            antlr_jar = os.environ.get('ANTLR_JAR',
                                       'antlr-4.11.1-complete.jar')
            result = subprocess.run([
                "java", "-jar", antlr_jar, "-Dlanguage=Python3",
                str(grammar_file)
            ],
                                    cwd=str(current_dir),
                                    capture_output=True,
                                    text=True,
                                    check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise WaveformParseError(
                "Failed to generate ANTLR parser. Please install ANTLR4 or set ANTLR_JAR environment variable."
            )


def parse_waveform_expression(expr: str) -> waveform.Waveform:
    """Parse a waveform expression using ANTLR4."""
    try:
        # Generate parser files if they don't exist
        # _generate_antlr_parser()
        
        # Import generated ANTLR classes
        from .WaveformLexer import WaveformLexer
        from .WaveformParser import WaveformParser

        # Create lexer and parser
        input_stream = InputStream(expr)
        lexer = WaveformLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = WaveformParser(stream)

        # Add error listener
        error_listener = WaveformErrorListener()
        parser.removeErrorListeners()
        parser.addErrorListener(error_listener)

        # Parse expression
        tree = parser.expr()

        # Visit tree and evaluate
        visitor = WaveformVisitor()
        result = visitor.visit(tree)

        # Convert numeric results to waveforms
        if isinstance(result, (int, float, complex)):
            result = waveform.const(result)

        return result.simplify()

    except Exception as e:
        if isinstance(e, WaveformParseError):
            raise
        raise WaveformParseError(
            f"Failed to parse expression '{expr}': {str(e)}")


@lru_cache(maxsize=1024)
def wave_eval(expr: str) -> "waveform.Waveform":
    """
    Parse and evaluate a waveform expression using ANTLR 4.
    
    Args:
        expr: The expression string to parse
        
    Returns:
        A Waveform object representing the parsed expression
        
    Raises:
        SyntaxError: If the expression cannot be parsed
    """
    try:
        return parse_waveform_expression(expr)
    except WaveformParseError as e:
        raise SyntaxError(f"Failed to parse expression '{expr}': {str(e)}")
    except Exception as e:
        raise SyntaxError(f"Failed to parse expression '{expr}': {str(e)}")
