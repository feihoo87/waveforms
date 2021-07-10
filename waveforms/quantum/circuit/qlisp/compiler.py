from typing import Optional

from .assembler import assembly
from .config import Config
from .library import Library
from .macro import extend_macro, reduceVirtualZ
from .qasm import qasm_eval
from .stdlib import std


def compile(prog, cfg: Optional[Config] = None, lib: Library = std, **options):
    """
    options: 
        qasm_only = True: only compile qasm to qlisp
        no_virtual_z = True: keep P gates as original form.
        no_assembly = True: return simplified qlisp.
    """
    if isinstance(prog, str):
        prog = qasm_eval(prog, lib)
    if 'qasm_only' in options:
        return list(prog)
    prog = extend_macro(prog, lib)
    if 'no_virtual_z' in options:
        return list(prog)
    prog = reduceVirtualZ(prog, lib)
    if 'no_assembly' in options:
        return list(prog)
    code = assembly(prog, cfg, lib)
    return code
