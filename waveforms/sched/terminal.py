from .base import Terminal as BaseTerminal
from .base import Scheduler, User
from .task import Task, create_task


class Terminal(BaseTerminal):
    def __init__(self, scheduler: Scheduler, user: User):
        self.__scheduler = scheduler
        self.__user = user

    @property
    def db(self) -> Scheduler:
        return self.__scheduler.db()

    @property
    def cfg(self):
        return self.__scheduler.cfg

    @property
    def user(self) -> User:
        return self.__user

    def logout(self):
        pass

    def submit(self, task: Task):
        task.runtime.user = self.__user
        return self.__scheduler.submit(task)

    def cancel(self, task: Task):
        pass

    def create_task(self, app, args=(), kwds={}):
        """
        create a task from a string or a class

        Args:
            app: a string or a class
            args: arguments for the class
            kwds: keyword arguments for the class
        
        Returns:
            a task
        """
        task = create_task((app, args, kwds))
        task._set_kernel(self.__scheduler, -1)
        task.runtime.user = self.__user
        return task
