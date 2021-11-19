from typing import Literal, Optional, Sequence, Union

from .arch import get_arch
from .assembly_left import assembly_align_left
from .assembly_right import assembly_align_right
from .config import Config
from .library import Library, libraries
from .macro import extend_macro, reduceVirtualZ
from .qasm import qasm_eval
from .qlisp import Context, create_context, getConfig
from .stdlib import std


def assembly(qlisp,
             cfg: Optional[Config] = None,
             lib: Library = std,
             ctx: Optional[Context] = None,
             align: Literal['left', 'right'] = 'left'):

    if cfg is None:
        cfg = getConfig()

    if ctx is None:
        ctx = create_context(cfg=cfg)

    if align == 'left':
        return assembly_align_left(qlisp, ctx, lib)
    elif align == 'right':
        return assembly_align_right(qlisp, ctx, lib)
    else:
        raise ValueError(f'align={align} is not supported.')


def compile(prog,
            cfg: Optional[Config] = None,
            lib: Union[Library, Sequence[Library]] = std,
            **options):
    """
    options: 
        qasm_only = True: only compile qasm to qlisp
        no_virtual_z = True: keep P gates as original form.
        no_assembly = True: return simplified qlisp.
    """
    if isinstance(lib, Library):
        lib = lib
    else:
        lib = libraries(*lib)

    if isinstance(prog, str):
        prog = qasm_eval(prog, lib)
    if 'qasm_only' in options:
        return list(prog)
    prog = extend_macro(prog, lib)
    if 'no_virtual_z' in options:
        return list(prog)
    prog = reduceVirtualZ(prog, lib)
    if 'no_link' in options:
        return list(prog)
    if 'align_right' in options or 'ar' in options:
        align = 'right'
    else:
        align = 'left'
    code = assembly(prog, cfg, lib, align=align)
    if 'arch' in options:
        code.arch = options['arch']
    if code.arch == 'general' or 'no_assembly' in options:
        return code
    return get_arch(code.arch).assembly_code(code)
