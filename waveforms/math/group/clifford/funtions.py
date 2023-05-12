import operator
from functools import reduce


def cliffordOrder(n: int) -> int:
    """
    Order of complex Clifford group of degree 2^n arising in quantum coding theory.
    
    Sloane, N. J. A. (ed.). "Sequence A003956 (Order of Clifford group)".
    The On-Line Encyclopedia of Integer Sequences. OEIS Foundation.
    https://oeis.org/A003956
    """
    return reduce(operator.mul, (((1 << (2 * j)) - 1) << 2 * j + 1
                                 for j in range(1, n + 1)), 1)