from functools import lru_cache
from typing import TYPE_CHECKING
from .antlr_parser import parse_waveform_expression, WaveformParseError

if TYPE_CHECKING:
    from . import waveform


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
