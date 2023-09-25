def encode_paulis(a: str):
    a = a.strip()
    if a.startswith('-i'):
        sign = 3
        a = a[2:]
    elif a.startswith('i'):
        sign = 1
        a = a[1:]
    elif a.startswith('+i'):
        sign = 1
        a = a[2:]
    elif a.startswith('-'):
        sign = 2
        a = a[1:]
    elif a.startswith('+'):
        sign = 0
        a = a[1:]
    else:
        sign = 0

    code = {'I': 0, 'X': 1, 'Z': 2, 'Y': 3}

    ret = 0
    for op in a[::-1]:
        ret <<= 2
        ret |= code[op]
    return (ret << 2) | sign


def decode_paulis(a: int):
    code = ['I', 'X', 'Z', 'Y']
    sign = ['  ', ' i', ' -', '-i'][a & 3]
    a >>= 2
    s = []
    while a:
        s.append(code[a & 3])
        a >>= 2
    if not s:
        s.append('I')
    return sign + ''.join(s)


def mul_paulis(a, b):
    if isinstance(a, str):
        a = encode_paulis(a)
    if isinstance(b, str):
        b = encode_paulis(b)
    return decode_paulis(imul_paulis(a, b))


def imul_paulis(a, b):
    tab = [0, 0, 0, 0, 0, 0, 3, 1, 0, 1, 0, 3, 0, 3, 1, 0]
    sign = (a + b) & 3
    code = ((a ^ b) | 3) ^ 3
    a = a >> 2
    b = b >> 2
    while a or b:
        sign += tab[((a & 3) << 2) | (b & 3)]
        a = a >> 2
        b = b >> 2
    return code | sign & 3
