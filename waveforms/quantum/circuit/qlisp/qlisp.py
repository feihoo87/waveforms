def gateName(st):
    if isinstance(st[0], str):
        return st[0]
    else:
        return st[0][0]


class QLispError(Exception):
    pass


class QLisp():
    def __init__(self, qlisp):
        self.prog = qlisp
        self.stack = list()

    def send(self, st):
        if st is not None:
            if isinstance(st, tuple):
                self.stack.append(st)
            elif isinstance(st, list):
                self.stack.extend(reversed(st))

    def __iter__(self):
        prog = iter(self.prog)
        while True:
            try:
                yield self.stack.pop()
                continue
            except IndexError:
                pass
            try:
                yield next(prog)
            except StopIteration:
                break
        yield from self.stack
