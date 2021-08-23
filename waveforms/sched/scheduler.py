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
from pathlib import Path
from queue import Empty, PriorityQueue
from typing import Any, Optional, Union

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.pool import SingletonThreadPool
from waveforms.quantum.circuit.qlisp.config import Config
from waveforms.quantum.circuit.qlisp.library import Library
from waveforms.storage.models import User, create_tables
from waveforms.waveform import Waveform

from .base import COMMAND, READ, WRITE, Executor, ThreadWithKill
from .scan_iters import scan_iters
from .task import Task, create_task

log = logging.getLogger(__name__)


def _is_finished(task: Task) -> bool:
    """Check if a task is finished."""
    if task.kernel is None:
        return False
    finished_step = len(task._fetch_result()['data'])
    return task.status not in ['submiting', 'pending', 'compiling'
                               ] and finished_step >= task.runtime.step


def fetch_data(task: Task, executor: Executor):
    try:
        while True:
            if threading.current_thread()._kill_event.is_set():
                with task.runtime._status_lock:
                    task.runtime.status = 'cancelled'
                break
            time.sleep(1)

            if _is_finished(task):
                executor.save(task.id, task.data_path)
                task.runtime.finished_time = time.time()
                if task.runtime.record is not None:
                    try:
                        data = task._fetch_result()
                        data['program'] = task.runtime.prog
                        task.runtime.record.data = data
                        task.db.commit()
                    except Exception as e:
                        log.error(f"Failed to save record: {e}")
                else:
                    log.warning(f"No record for task {task.name}({task.id})")
                with task.runtime._status_lock:
                    task.runtime.status = 'finished'
                break
    except:
        with task.runtime._status_lock:
            task.runtime.status = 'failed'
        log.exception(f"{task.name}({task.id}) is failed")
        executor.free(task.id)
    finally:
        log.debug(f'{task.name}({task.id}) is finished')


def clean_side_effects(task: Task, executor: Executor):
    cmds = []
    for k, v in task.runtime.prog.side_effects.items():
        cmds.append(WRITE(k, v))
        executor.update(k, v, cache=False)
    executor.feed(task.id, -1, cmds)
    task.cfg.clear_buffer()


def exec_circuit(task: Task, circuit: Union[str, list], lib: Library,
                 cfg: Config, signal: str, compile_once: bool) -> int:
    """Execute a circuit."""
    from waveforms import compile
    from waveforms.backends.quark.executable import getCommands

    task.runtime.prog.steps[-1][2].extend(task.runtime.cmds)
    if task.runtime.step == 0 or not compile_once:
        code = compile(circuit, lib=lib, cfg=cfg)
        cmds, dataMap = getCommands(code, signal=signal, shots=task.shots)
        task.runtime.cmds.extend(cmds)
        task.runtime.prog.data_maps[-1].update(dataMap)
        task.runtime.prog.steps[-1][0].extend(circuit)
    else:
        for cmd in task.runtime.prog.commands[-1]:
            if (isinstance(cmd, READ) or cmd.address.endswith('.StartCapture')
                    or cmd.address.endswith('.CaptureMode')):
                task.runtime.cmds.append(cmd)
        task.runtime.prog.data_maps[-1] = task.runtime.prog.data_maps[0]
        task.runtime.prog.steps[-1][2].extend(task.runtime.cmds)
        task.runtime.prog.steps[-1][3].update(task.runtime.prog.data_maps[-1])
    return task.runtime.step


def submit_loop(task_queue: PriorityQueue, current_stack: list[Task],
                running_pool: dict[int, Task], executor: Executor):
    while True:
        if len(current_stack) > 0:
            current_task = current_stack.pop()

            if current_task.runtime.threads['submit'].is_alive():
                current_stack.append(current_task)
            else:
                with current_task.runtime._status_lock:
                    current_task.runtime.status = 'running'

        try:
            task = task_queue.get_nowait()
        except Empty:
            time.sleep(1)
            continue
        if task.runtime.at > 0 and task.runtime.at > time.time():
            task_queue.put(task)
            time.sleep(1)
        elif (len(current_stack) == 0
              or task.is_children_of(current_stack[-1])):
            submit(task, current_stack, running_pool, executor)
        else:
            task_queue.put_nowait(task)


def submit_thread(task: Task, executor: Executor):
    """Submit a task."""
    i = 0
    while True:
        if any(t._kill_event.is_set() for t in task.runtime.threads.values()):
            break
        if (i >= task.runtime.step
                and not task.runtime.threads['compile'].is_alive()):
            break
        if i == len(task.runtime.prog.commands):
            time.sleep(1)
            continue
        executor.feed(task.id, i, task.runtime.prog.commands[i])
        i += 1
    clean_side_effects(task, executor)


def submit(task: Task, current_stack: list[Task],
           running_pool: dict[int, Task], executor: Executor):
    executor.free(task.id)
    with task.runtime._status_lock:
        task.runtime.status = 'submiting'
    if task.runtime.prog.with_feedback:
        task.runtime.threads['compile'].start()

    current_stack.append(task)
    task.runtime.started_time = time.time()
    task.runtime.threads['submit'].start()

    running_pool[task.id] = task
    task.runtime.threads['fetch_data'].start()


def waiting_loop(running_pool: dict[int, Task], debug_mode: bool = False):
    while True:
        for taskID, task in list(running_pool.items()):
            if not task.runtime.threads['fetch_data'].is_alive():
                try:
                    if not debug_mode:
                        del running_pool[taskID]
                except:
                    pass
        time.sleep(0.1)


def expand_task(task: Task, executor: Executor):
    task.runtime.step = 0
    task.runtime.prog.index = []
    task.runtime.prog.commands = []
    task.runtime.prog.data_maps = []
    task.runtime.prog.side_effects = {}
    task.runtime.prog.steps = []
    task.runtime.prog.shots = task.shots
    task.runtime.prog.signal = task.signal

    iters = task.scan_range()
    if isinstance(iters, tuple) and len(iters) == 2:
        iters, filter_func = iters
    elif isinstance(iters, dict):
        iters, filter_func = iters, None
    else:
        raise ValueError(f"Invalid scan range: {iters}")
    for step in scan_iters(iters, filter_func):
        try:
            if threading.current_thread()._kill_event.is_set():
                break
        except AttributeError:
            pass

        task.runtime.prog.index.append(step)
        task.runtime.prog.data_maps.append({})
        task.runtime.prog.steps.append(([], {}, [], {}))

        for k, v in step.kwds.items():
            if k in task.runtime.result['index']:
                task.runtime.result['index'][k].append(v)
            else:
                task.runtime.result['index'][k] = [v]

        task.runtime.cmds = []
        yield step
        task.trig()
        cmds = task.runtime.cmds
        task.runtime.prog.commands.append(task.runtime.cmds)

        for k, v in task.cfg._history.items():
            task.runtime.prog.side_effects.setdefault(k, v)

        for cmd in cmds:
            if isinstance(cmd.value, Waveform):
                task.runtime.prog.side_effects[cmd.address] = 'zero()'
        task.runtime.step += 1


class Scheduler():
    def __init__(self,
                 executor: Executor,
                 url: Optional[str] = None,
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
        self.__uuid = uuid.uuid1()
        self._task_pool = {}
        self._queue = PriorityQueue()
        self._waiting_pool = {}
        self._submit_stack = []
        self.mutex = set()
        self.executor = executor
        if url is None:
            url = 'sqlite:///{}'.format(data_path / 'waveforms.db')
        self.db = url
        self.data_path = Path(data_path)
        self.eng = create_engine(url,
                                 echo=debug_mode,
                                 poolclass=SingletonThreadPool,
                                 connect_args={'check_same_thread': False})
        if (self.db == 'sqlite:///:memory:' or self.db.startswith('sqlite:///')
                and not os.path.exists(self.db.removeprefix('sqlite:///'))):
            create_tables(self.eng)

        self.system_user = self.login('BIG BROTHER', self.__uuid)

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

    def login(self, username: str, password: str) -> User:
        db = self.session()
        if username == 'BIG BROTHER' and password == self.__uuid:
            try:
                user = db.query(User).filter(User.name == username).one()
            except NoResultFound:
                user = User(name=username)
                db.add(user)
                db.commit()
        else:
            try:
                user = db.query(User).filter(User.name == username).one()
            except NoResultFound:
                raise ValueError('User not found')
            if not user.verify(password):
                raise ValueError('Wrong password')
        return user

    @property
    def cfg(self):
        return self.executor.cfg

    def session(self):
        return sessionmaker(bind=self.eng)()

    def get_task_by_id(self, task_id):
        try:
            return self._task_pool.get(task_id)()
        except:
            return None

    def list_tasks(self):
        return {id: ref() for id, ref in self._task_pool.items()}

    def cancel(self):
        self.executor.cancel()
        self._queue.clear()
        while self._submit_stack:
            task, thread = self._submit_stack.pop()
            thread.kill()
            with task.runtime._status_lock:
                task.runtime.status = 'cancelled'

        while self._waiting_result:
            task_id, (task, thread) = self._waiting_result.popitem()
            thread.kill()
            with task.runtime._status_lock:
                task.runtime.status = 'cancelled'

    def join(self, task):
        while True:
            if task.status in ['finished', 'cancelled', 'failed']:
                break
            time.sleep(0.01)

    async def join_async(self, task):
        while True:
            if task.status in ['finished', 'cancelled', 'failed']:
                break
            await asyncio.sleep(0.01)

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

    def generate_task_id(self):
        i = uuid.uuid3(self.__uuid, f"{next(self.counter)}").int
        return i & ((1 << 64) - 1)

    def scan(self, task):
        """Yield from task.scan_range().

        :param task: task to scan
        :return: a generator yielding step arguments.
        """
        yield from expand_task(task, self.executor)
        with task.runtime._status_lock:
            if task.status == 'compiling':
                task.runtime.status = 'pending'

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
                    self.runtime.cmds.extend(cmds)
                    self.exec(circuit, lib=lib, cfg=cfg)

        t = A()
        t.signal = signal
        self.submit(t)
        return t

    def _measure(self, task, keys, labels=None):
        if labels is None:
            labels = keys
        dataMap = {'data': {label: key for key, label in zip(keys, labels)}}
        task.runtime.prog.data_maps[-1].update(dataMap)
        task.runtime.cmds.extend([READ(key) for key in keys])

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
        with task.runtime._status_lock:
            if task.status != 'not submited':
                raise RuntimeError(
                    f'Task({task.id}, status={task.status}) has been submited!'
                )
        taskID = self.generate_task_id()
        task._set_kernel(self, taskID)
        task.runtime.threads.update({
            'submit':
            ThreadWithKill(target=submit_thread, args=(task, self.executor)),
            'fetch_data':
            ThreadWithKill(target=fetch_data, args=(task, self.executor))
        })
        with task.runtime._status_lock:
            if not task.runtime.prog.with_feedback:
                task.runtime.threads['compile'].start()
                task.runtime.status = 'compiling'
            else:
                task.runtime.status = 'pending'
        self._queue.put_nowait(task)

        def delete(ref, dct, key):
            dct.pop(key)

        self._task_pool[task.id] = weakref.ref(
            task, functools.partial(delete, dct=self._task_pool, key=task.id))
        return task

    def query(self, key):
        return self.cfg.query(key)

    def update(self, key, value, cache=False):
        self.executor.update(key, value, cache=cache)

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
        task._set_kernel(self, -1)
        return task
