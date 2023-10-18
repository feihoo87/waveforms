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


def decode_paulis(a: int, N=None):
    code = ['I', 'X', 'Z', 'Y']
    sign = ['  ', ' i', ' -', '-i'][a & 3]
    a >>= 2
    s = []
    while a:
        s.append(code[a & 3])
        a >>= 2
    if not s:
        s.append('I')
    if N is not None and N > len(s):
        s.extend(['I'] * (N - len(s)))
    return sign + ''.join(s)


def mul_paulis(a: int | str, b: int | str):
    if isinstance(a, str):
        a = encode_paulis(a)
    if isinstance(b, str):
        b = encode_paulis(b)
    return decode_paulis(imul_paulis(a, b))


def imul_paulis(a: int, b: int):
    sign = (a + b) & 3

    a = a >> 2
    b = b >> 2
    c = a ^ b

    d = max(a.bit_length(), b.bit_length())
    d += d % 2
    m = ((1 << d) - 1) // 3

    l = (a | (a >> 1)) & (b | (b >> 1)) & (c | c >> 1) & m
    h = (((a >> 1) & b) ^ (c & (c >> 1))) & l

    sign = (h.bit_count() * 2 + l.bit_count() + sign) & 3

    return (c << 2) | sign
