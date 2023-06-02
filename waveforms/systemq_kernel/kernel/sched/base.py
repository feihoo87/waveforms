from __future__ import annotations

import asyncio
import copy
import functools
import inspect
import itertools
import logging
import threading
import time
from abc import ABC, ABCMeta, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from os import system
from typing import (Any, Generator, Iterable, Literal, NamedTuple, Optional,
                    Sequence, Type, Union)

from sqlalchemy.orm.session import Session
from waveforms.scan_iter import Storage

from qlisp import COMMAND, Architecture, Program
from storage.models import Record, Report, User

from .progress import Progress


class ThreadWithKill(threading.Thread):

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self._kill_event = threading.Event()

    def kill(self):
        self._kill_event.set()


class Executor(ABC):
    """
    Base class for executors.
    """

    @property
    def log(self):
        return logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}")

    @abstractmethod
    def feed(self, priority: int, task_id: int, step_id: int,
             cmds: list[COMMAND]):
        pass

    @abstractmethod
    def fetch(self, task_id: int, skip: int = 0) -> list:
        pass

    @abstractmethod
    def free(self, task_id: int):
        pass

    @abstractmethod
    def boot(self):
        pass

    @abstractmethod
    def shutdown(self):
        pass

    @abstractmethod
    def reset(self):
        pass


TASKSTATUS = Literal['not submited', 'pending', 'compiling', 'submiting',
                     'running', 'finished', 'cancelled', 'failed']


@dataclass
class TaskRuntime():
    priority: int = 0  # Priority of the task
    daemon: bool = False  # Is the task a daemon
    at: float = -1  # Time at which the task is scheduled
    period: float = -1  # Period of the task

    status: TASKSTATUS = 'not submited'
    id: int = -1
    created_time: float = field(default_factory=time.time)
    started_time: float = field(default_factory=time.time)
    finished_time: float = field(default=-1)
    kernel: object = None
    db: Session = None
    user: User = None

    prog: Program = field(default_factory=Program)
    arch: Architecture = None

    #################################################
    compiled_step: int = 0
    finished_step: int = 0
    sub_index: int = 0
    cmds: list = field(default_factory=list)
    skip_compile: bool = False
    storage: Storage = field(default_factory=Storage)
    record: Optional[Record] = None
    system_info: dict = field(default_factory=dict)
    keep_last_status: bool = False
    dry_run: bool = False

    progress: Progress = field(default_factory=Progress)

    threads: dict = field(default_factory=dict)
    _status_lock: threading.Lock = field(default_factory=threading.Lock)
    _kill_event: threading.Event = None

    used_elements: set = field(default_factory=set)


class AnalyzeResult(NamedTuple):
    """
    Result of the analysis.
    """
    score: int = 0
    # how good is the result
    # 100 is perfect
    # 0 implied full calibration is required
    # and negative is bad data

    parameters: dict = {}
    # new values of the parameters from the analysis
    # only required for 100 score

    tags: set[str] = set()

    status: str = 'not analyzed'
    message: str = ''


@functools.total_ordering
class Task(ABC):

    def __init__(self):
        self.__runtime = TaskRuntime()

    @abstractmethod
    def scan_range(self):
        pass

    @abstractmethod
    def main(self):
        pass

    @abstractmethod
    def scan(self):
        pass

    @abstractmethod
    def analyze(self, result) -> AnalyzeResult:
        pass

    @property
    def priority(self):
        return self.__runtime.priority

    @property
    def id(self):
        return self.__runtime.id

    @property
    def name(self):
        return f"{self.__class__.__module__}.{self.__class__.__name__}"

    @property
    def log(self):
        return logging.getLogger(f"{self.name}")

    @property
    def kernel(self):
        from . import sched
        return sched

    @property
    def meta_info(self):
        return self.__runtime.prog.meta_info

    @property
    def runtime(self):
        return self.__runtime

    @property
    def db(self):
        return self.__runtime.db

    @property
    def cfg(self):
        return self.kernel.cfg

    @property
    def status(self):
        return self.__runtime.status

    @property
    @abstractmethod
    def tags(self):
        pass

    async def done(self):
        pass

    def result(self):
        pass

    def cancel(self):
        pass

    def __deepcopy__(self, memo):
        memo[id(self.__runtime)] = TaskRuntime(
            arch=self.__runtime.arch,
            prog=self.__runtime.prog,
            used_elements=self.__runtime.used_elements)
        ret = copy.copy(self)
        for attr, value in self.__dict__.items():
            setattr(ret, attr, copy.deepcopy(value, memo))
        return ret

    def __lt__(self, other: Task):
        return ((self.runtime.at, self.priority, self.runtime.created_time) <
                (self.runtime.at, other.priority, other.runtime.created_time))


class Terminal(ABC):

    @property
    def log(self):
        return logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}")

    @abstractmethod
    def db(self) -> Session:
        pass

    @property
    @abstractmethod
    def user(self) -> User:
        pass

    @abstractmethod
    def logout(self):
        pass

    @abstractmethod
    def submit(self, task: Task):
        pass

    @abstractmethod
    def cancel(self, task: Task):
        pass

    @abstractmethod
    def create_task(self, cls, args=(), kwds={}) -> Task:
        pass
