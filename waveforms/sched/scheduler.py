import asyncio
import functools
import itertools
import logging
import os
import threading
import time
import uuid
import warnings
import weakref
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Any, Union

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from waveforms.storage.models import create_tables
from waveforms.waveform import Waveform

from .task import COMMAND, READ, WRITE, Task, create_task

log = logging.getLogger(__name__)


class Executor(ABC):
    @property
    def log(self):
        return logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}")

    @abstractmethod
    def feed(self, task_id: int, task_step: int, cmds: list[COMMAND],
             extra: dict):
        """
        """
        pass

    @abstractmethod
    def free(self, task_id: int) -> None:
        pass

    @abstractmethod
    def submit(self, task_id: int, data_template: dict) -> None:
        pass

    @abstractmethod
    def fetch(self, task_id: int, skip: int = 0) -> list:
        pass

    @abstractmethod
    def save(self, path: str, task_id: int, data: dict) -> str:
        pass


class _ThreadWithKill(threading.Thread):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self._kill_event = threading.Event()

    def kill(self):
        self._kill_event.set()


def join_task(task: Task, executor: Executor):
    try:
        while True:
            if threading.current_thread()._kill_event.is_set():
                break
            time.sleep(1)
            if task._runtime.status not in ['submiting', 'pending'] and len(
                    task.result()
                ['data']) == task._runtime.step or task._runtime.status in [
                    'canceled', 'finished'
                ]:
                executor.save(task.data_path, task.id)
                task._runtime.finished_time = time.time()
                if task._runtime.record is not None:
                    task._runtime.record.save()
                    task.db.commit()
                break
    except:
        executor.free(task.id)
    finally:
        clean_side_effects(task, executor)


def clean_side_effects(task: Task, executor: Executor):
    cmds = []
    for k, v in task._runtime.side_effects.items():
        cmds.append(WRITE(k, v))
        executor.update(k, v, cache=False)
    executor.feed(task.id, -1, cmds)
    task._runtime.side_effects.clear()
    task.cfg.clear_buffer()


def exec_circuit(task, circuit, lib, cfg, signal, compile_once):
    """Execute a circuit."""
    from waveforms import compile
    from waveforms.backends.quark.executable import getCommands

    if task._runtime.step == 0 or not compile_once:
        code = compile(circuit, lib=lib, cfg=cfg)
        cmds, dataMap = getCommands(code, signal=signal, shots=task.shots)
        task._runtime.cmds.extend(cmds)
        task._runtime.dataMaps[-1].update(dataMap)
    else:
        for cmd in task._runtime.cmds_list[-1]:
            if (isinstance(cmd, READ) or cmd.key.endswith('.StartCapture')
                    or cmd.key.endswith('.CaptureMode')):
                task._runtime.cmds.append(cmd)
        task._runtime.dataMaps[-1] = task._runtime.dataMaps[0]
    return task._runtime.step


def submit_loop(task_queue: deque, current_stack: list[tuple[Task,
                                                             _ThreadWithKill]],
                running_pool: dict[int, tuple[Task, _ThreadWithKill]],
                executor: Executor):
    while True:
        if len(current_stack) > 0:
            current_task, thread = current_stack.pop()

            if thread.is_alive():
                current_stack.append((current_task, thread))
            else:
                current_task._runtime.status = 'running'

        try:
            task = task_queue.popleft()
        except IndexError:
            time.sleep(1)
            continue

        if (len(current_stack) == 0
                or task.is_children_of(current_stack[-1][0])):
            submit(task, current_stack, running_pool, executor)
        else:
            task_queue.appendleft(task)


def submit(task: Task, current_stack: list[tuple[Task, _ThreadWithKill]],
           running_pool: dict[int, tuple[Task, _ThreadWithKill]],
           executor: Executor):
    executor.free(task.id)
    executor.submit(task.id, {})
    task._runtime.status = 'submiting'
    submit_thread = _ThreadWithKill(target=task.main)
    current_stack.append((task, submit_thread))
    task._runtime.started_time = time.time()
    submit_thread.start()

    fetch_data_thread = _ThreadWithKill(target=join_task,
                                        args=(task, executor))
    running_pool[task.id] = task, fetch_data_thread
    fetch_data_thread.start()


def waiting_loop(running_pool: dict[int, tuple[Task, _ThreadWithKill]],
                 debug_mode: bool = False):
    while True:
        for taskID, (task, thread) in list(running_pool.items()):
            if not thread.is_alive():
                try:
                    if not debug_mode:
                        del running_pool[taskID]
                except:
                    pass
                task._runtime.status = 'finished'
        time.sleep(0.01)


def expand_task(task: Task, executor: Executor):
    task._runtime.step = 0
    for args in task.scan_range():
        try:
            if threading.current_thread()._kill_event.is_set():
                break
        except AttributeError:
            pass
        task._runtime.result['index'].append(args)
        task._runtime.dataMaps.append({})
        task._runtime.cmds = []
        yield args
        task.trig()
        cmds = task._runtime.cmds
        task._runtime.cmds_list.append(task._runtime.cmds)
        for k, v in task.cfg._history.items():
            task._runtime.side_effects.setdefault(k, v)
        executor.feed(task.id,
                      task._runtime.step,
                      cmds,
                      extra={
                          'hello': 'world',
                      })
        for cmd in task._runtime.cmds:
            if isinstance(cmd.value, Waveform):
                task._runtime.side_effects[cmd.key] = 'zero()'
        task._runtime.step += 1


class Scheduler():
    def __init__(self,
                 executor: Executor,
                 url: str = 'sqlite:///:memory:',
                 data_path: Union[str, Path] = Path.home() / 'data',
                 debug_mode: bool = False):
        """
        Parameters
        ----------
        executor : Executor
            The executor to use to submit tasks
        url : str
            The url of the database. These URLs follow RFC-1738, and usually
            can include username, password, hostname, database name as well
            as optional keyword arguments for additional configuration.
            In some cases a file path is accepted, and in others a "data
            source name" replaces the "host" and "database" portions. The
            typical form of a database URL is:
                `dialect+driver://username:password@host:port/database`
        """
        self.counter = itertools.count()
        self.uuid = uuid.uuid1()
        self._task_pool = {}
        self._queue = deque()
        self._waiting_pool = {}
        self._submit_stack = []
        self.executor = executor
        self.db = url
        self.data_path = Path(data_path)
        self.eng = create_engine(url)
        if (self.db == 'sqlite:///:memory:' or self.db.startswith('sqlite:///')
                and not os.path.exists(self.db.removeprefix('sqlite:///'))):
            create_tables(self.eng)

        self._read_data_thread = threading.Thread(target=waiting_loop,
                                                  args=(self._waiting_pool,
                                                        debug_mode),
                                                  daemon=True)
        self._read_data_thread.start()

        self._submit_thread = threading.Thread(
            target=submit_loop,
            args=(self._queue, self._submit_stack, self._waiting_pool,
                  self.executor),
            daemon=True)
        self._submit_thread.start()

    @property
    def cfg(self):
        return self.executor.cfg

    @property
    def executer(self):
        warnings.warn(
            'kernel.executer is deprecated, use kernel.executor instead',
            DeprecationWarning)
        return self.executor

    def session(self):
        return sessionmaker(bind=self.eng)()

    def get_task_by_id(self, task_id):
        try:
            return self._task_pool.get(task_id)()
        except:
            return None

    def cancel(self):
        self.executor.cancel()
        self._queue.clear()
        while self._submit_stack:
            task, thread = self._submit_stack.pop()
            thread.kill()
            task._runtime.status = 'canceled'

        while self._waiting_result:
            task_id, (task, thread) = self._waiting_result.popitem()
            thread.kill()
            task._runtime.status = 'canceled'

    def join(self, task):
        while True:
            if task._runtime.status == 'finished':
                break
            time.sleep(1)

    def set(self, key: str, value: Any, cache: bool = False):
        cmds = []
        if not cache:
            cmds.append(WRITE(key, value))
        if len(cmds) > 0:
            self.executor.feed(0, -1, cmds)
            self.executor.free(0)
        self.cfg.update(key, value, cache=cache)

    def get(self, key: str):
        """
        return the value of the key in the kernel config
        """
        return self.query(key)

    async def join_async(self, task):
        while True:
            if task._runtime.status == 'finished':
                break
            await asyncio.sleep(1)

    def generate_task_id(self):
        i = uuid.uuid3(self.uuid, f"{next(self.counter)}").int
        return i & ((1 << 64) - 1)

    def scan(self, task):
        """Yield from task.scan_range().

        :param task: task to scan
        :return: a generator yielding step arguments.
        """
        yield from expand_task(task, self.executor)

    def _exec(self,
              task,
              circuit,
              lib=None,
              cfg=None,
              signal='state',
              compile_once=False):
        """Execute a circuit."""
        from waveforms import stdlib

        if lib is None:
            lib = stdlib
        if cfg is None:
            cfg = self.cfg

        return exec_circuit(task,
                            circuit,
                            lib=lib,
                            cfg=cfg,
                            signal=signal,
                            compile_once=compile_once)

    def exec(self, circuit, lib=None, cfg=None, signal='state', cmds=[]):
        """Execute a circuit.
        
        Parameters:
            circuit: a QLisp Circuit.
            lib: a Library used by compiler,default is stdlib.
            cfg: configuration of system.
            signal: a str of the name of the signal type to be returned.
            cmds: additional commands.

        Returns:
            A Task.
        """
        from waveforms.sched import App

        class A(App):
            def scan_range(self):
                yield 0

            def main(self):
                for _ in self.scan():
                    self._runtime.cmds.extend(cmds)
                    self.exec(circuit, lib=lib, cfg=cfg)

        t = A()
        t.signal = signal
        self.submit(t)
        return t

    def _measure(self, task, keys, labels=None):
        if labels is None:
            labels = keys
        dataMap = {label: key for key, label in zip(keys, labels)}
        task._runtime.dataMaps[-1].update(dataMap)
        cmds = [(key, READ) for key in keys]
        task._runtime.cmds.extend(cmds)

    def measure(self, keys, labels=None, cmds=[]):
        pass

    def update_parameters(self, parameters: dict[str, Any]):
        """Update parameters.

        Args:
            parameters: a dict of parameters.
        """
        for key, value in parameters.items():
            self.update(key, value)
        self.cfg.clear_buffer()

    def maintain(self, task: Task) -> Task:
        """Maintain a task.
        """
        from ._bigbrother import maintain

        return maintain(self, task)

    def diagnose(self, task: Task) -> bool:
        """
        Diagnose a task.

        Returns: True if node
        or dependent recalibrated.
        """
        from ._bigbrother import diagnose

        return diagnose(self, task)

    def fetch(self, task: Task, skip: int = 0) -> list[dict]:
        """Fetch result of task from the executor, skip the
        first `skip` steps.

        Args:
            task: a task.
            skip: the number of steps to skip.

        Returns:
            A list of dicts.
        """
        return self.executor.fetch(task.id, skip)

    def submit(self, task: Task) -> Task:
        """Submit a task.
        """
        taskID = self.generate_task_id()
        task.id = taskID
        task.kernel = self
        task.db = self.session()
        task._runtime.status = 'pending'
        self._queue.append(task)

        def delete(ref, dct, key):
            dct.pop(key)

        self._task_pool[task.id] = weakref.ref(
            task, functools.partial(delete, dct=self._task_pool, key=task.id))
        return task

    def query(self, key):
        return self.cfg.query(key)

    def update(self, key, value, cache=False):
        self.executor.update(key, value, cache=cache)

    def feedback(self, task, obj):
        task._runtime.feedback_buffer = obj

    def get_feedback(self, task):
        obj = task._runtime.feedback_buffer
        task._runtime.feedback_buffer = None
        return obj

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
        return create_task(app, args, kwds)
