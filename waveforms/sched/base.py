from __future__ import annotations

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
from typing import (Any, Generator, Iterable, Literal, NamedTuple, Optional,
                    Sequence, Type, Union)

from sqlalchemy.orm.session import Session
from waveforms.quantum.circuit.qlisp import (COMMAND, FREE, PUSH, READ, SYNC,
                                             TRIG, WRITE, Architecture,
                                             DataMap, QLispCode)
from waveforms.storage.models import Record, Report, User


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
    def get_config(self) -> dict:
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


@dataclass
class ProgramFrame():
    """
    A frame of a program.
    """
    step_id: int = 0
    cmds: list[COMMAND] = field(default_factory=list)
    data_map: DataMap = field(default_factory=dict)
    code: Optional[QLispCode] = None

    task_id: int = None
    priority: int = None


@dataclass
class Program:
    """
    A program is a list of commands.
    """
    with_feedback: bool = False
    arch: str = 'baqis'

    index: list = field(default_factory=list)
    side_effects: dict = field(default_factory=dict)

    steps: list[ProgramFrame] = field(default_factory=list)
    shots: int = 1024
    signal: str = 'state'

    snapshot: dict = field(default_factory=dict)
    task_arguments: tuple[tuple, dict] = (tuple(), dict())
    meta: dict = field(default_factory=dict)


@dataclass
class TaskRuntime():
    priority: int = 0  # Priority of the task
    daemon: bool = False  # Is the task a daemon
    at: float = -1  # Time at which the task is scheduled
    period: float = -1  # Period of the task

    status: str = 'not submited'
    id: int = -1
    created_time: float = field(default_factory=time.time)
    started_time: float = field(default_factory=time.time)
    finished_time: float = field(default=-1)
    kernel: Scheduler = None
    db: Session = None
    user: User = None

    prog: Program = field(default_factory=Program)
    arch: Architecture = None

    #################################################
    step: int = 0
    sub_index: int = 0
    cmds: list = field(default_factory=list)
    result: dict = field(default_factory=lambda: {
        'index': {},
        'data': [],
        'states': [],
        'counts': [],
        'diags': []
    })
    record: Optional[Record] = None

    threads: dict = field(default_factory=dict)
    _status_lock: threading.Lock = field(default_factory=threading.Lock)

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
        return self.__runtime.kernel

    @property
    def meta(self):
        return self.__runtime.prog.meta

    def _set_kernel(self, kernel, id):
        self.__runtime.id = id
        self.__runtime.kernel = kernel

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

    def __deepcopy__(self):
        memo = {id(self.__runtime): TaskRuntime(prog=self.__runtime.prog)}
        ret = copy.copy(self)
        for attr, value in self.__dict__.items():
            setattr(ret, attr, copy.deepcopy(value, memo))
        return ret

    def __lt__(self, other: Task):
        return ((self.runtime.at, self.priority, self.runtime.created_time) <
                (self.runtime.at, other.priority, other.runtime.created_time))


class Scheduler(ABC):
    @property
    def log(self):
        return logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}")

    @abstractmethod
    def db(self) -> Session:
        pass

    @abstractmethod
    def login(self, username: str, password: str) -> Terminal:
        pass

    @abstractmethod
    def submit(self, task: Task):
        pass

    @abstractmethod
    def maintain(self, task: Task):
        pass

    @abstractmethod
    def create_task(self, cls, args=(), kwds={}) -> Task:
        pass

    @abstractmethod
    def cancel(self, task: Task):
        pass

    @abstractmethod
    def get_task_by_id(self, task_id: int) -> Task:
        pass

    def __deepcopy__(self, memo):
        # DO NOT COPY THE KERNEL
        return self


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
