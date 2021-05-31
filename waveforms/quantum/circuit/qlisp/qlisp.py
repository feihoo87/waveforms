from collections import deque


def gateName(st):
    if isinstance(st[0], str):
        return st[0]
    else:
        return st[0][0]


class QLisp():
    def __init__(self, qlisp):
        self.prog = qlisp
        self.stack = deque()

    def send(self, st):
        if st is not None:
            self.stack.append(st)

    def __iter__(self):
        prog = iter(self.prog)
        while True:
            try:
                yield self.stack.popleft()
                continue
            except IndexError:
                pass
            try:
                yield next(prog)
            except StopIteration:
                break
        yield from self.stack
