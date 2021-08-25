from .base import Terminal as BaseTerminal
from .base import Scheduler, User, Task


class Terminal(BaseTerminal):
    def __init__(self, kernel: Scheduler, user: User):
        self.__kernel = kernel
        self.__user = user

    @property
    def db(self) -> Scheduler:
        return self.__kernel.db()

    @property
    def user(self) -> User:
        return self.__user

    def logout(self):
        return super().logout()

    def submit(self, task: Task):
        return super().submit(task)

    def cancel(self, task: Task):
        return super().cancel(task)

    def create_task(self, cls, args=(), kwds={}) -> Task:
        return super().create_task(cls, args=args, kwds=kwds)
