import asyncio
import copy
import functools
import itertools
import logging
import os
import pickle
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
from waveforms.storage.models import User, create_tables, set_data_path

from .base import WRITE, Executor
from .base import Scheduler as BaseScheduler
from .base import ThreadWithKill
from .ipy_events import set_sessionmaker, setup_ipy_events
from .task import Task, create_task
from .terminal import Terminal

log = logging.getLogger(__name__)


def clean_side_effects(task: Task, executor: Executor):
    cmds = []
    for k, v in task.runtime.prog.side_effects.items():
        cmds.append(WRITE(k, v))
    return executor.feed(task.id, -2, cmds, next_feed_time=30)


def submit_loop(task_queue: PriorityQueue, current_stack: list[Task]):
    while True:
        if len(current_stack) > 0:
            current_task = current_stack.pop()
            if current_task.runtime.threads['run'].is_alive():
                current_stack.append(current_task)
        try:
            task = task_queue.get_nowait()
        except Empty:
            time.sleep(1)
            continue
        if task.status == 'cancelled':
            pass
        elif task.runtime.at > 0 and task.runtime.at > time.time():
            task_queue.put_nowait(task)
            time.sleep(1)
        elif (len(current_stack) == 0
              or task.is_children_of(current_stack[-1])):
            submit(task, current_stack)
        else:
            task_queue.put_nowait(task)
            time.sleep(0.1)


def _feed_step(task, feed_step, executor):
    data_map = copy.copy(task.runtime.prog.steps[feed_step].data_map)
    if data_map['signal'] in ['count', 'diag']:
        data_map['signal'] = 'state'

    extra = {'dataMap': pickle.dumps(data_map)}

    while True:
        succeed = executor.feed(task.id,
                                feed_step,
                                task.runtime.prog.steps[feed_step].cmds,
                                extra,
                                next_feed_time=30)
        if succeed == 0 and feed_step == 0:
            with task.runtime._status_lock:
                task.runtime.status = 'pending'
            time.sleep(1)
        elif succeed == 1:
            with task.runtime._status_lock:
                task.runtime.status = 'submiting'
            break
        else:
            raise RuntimeError(
                f"Failed to feed {task.name}({task.id}), Executor busy.")


def _save_data(task):
    if task.runtime.record is not None:
        try:
            data = task.result()
            data['program'] = task.runtime.prog
            task.runtime.record.data = data
            task.db.commit()
        except Exception as e:
            log.error(f"Failed to save record: {e}")
    else:
        log.warning(f"No record for task {task.name}({task.id})")


def task_run_thread(task: Task, executor: Executor):
    LIMIT = 10
    feed_step = 0
    feed_more = True
    feed_finished = False
    side_effect_cleared = False

    executor.create_task(task.id, task.meta_info)

    try:
        while True:
            if threading.current_thread()._kill_event.is_set():
                with task.runtime._status_lock:
                    task.runtime.status = 'cancelled'
                break
            if (not feed_finished and feed_step >= task.runtime.step
                    and not task.runtime.threads['compile'].is_alive()):
                clean_side_effects(task, executor)
                feed_more = False
                feed_finished = True
                side_effect_cleared = True
                with task.runtime._status_lock:
                    if task.status in ['submiting', 'pending', 'compiling']:
                        task.runtime.status = 'running'
            if (feed_finished or feed_step + 1 >= len(task.runtime.prog.steps)
                    and task.runtime.threads['compile'].is_alive()) or (
                        feed_step - len(task.runtime.result['data']) > LIMIT):
                feed_more = False
            else:
                feed_more = True

            if feed_more:
                _feed_step(task, feed_step, executor)
                feed_step += 1

            finished_step = len(task._fetch_result()['data'])
            if (feed_finished and finished_step >= task.runtime.step):
                executor.save(task.id, task.data_path)
                task.runtime.finished_time = time.time()
                with task.runtime._status_lock:
                    task.runtime.status = 'finished'
                break
            if not feed_more:
                time.sleep(1)
    except:
        with task.runtime._status_lock:
            task.runtime.status = 'failed'
        log.exception(f"{task.name}({task.id}) is failed")
        executor.free(task.id)
    finally:
        if not side_effect_cleared:
            clean_side_effects(task, executor)
        _save_data(task)
        log.debug(f'{task.name}({task.id}) is finished')


def submit(task: Task, current_stack: list[Task]):
    with task.runtime._status_lock:
        task.runtime.status = 'submiting'
    if task.runtime.prog.with_feedback:
        task.runtime.threads['compile'].start()

    current_stack.append(task)
    task.runtime.started_time = time.time()
    task.runtime.threads['run'].start()


class Scheduler(BaseScheduler):
    def __init__(self):
        self.counter = itertools.count()
        self.__uuid = uuid.uuid1()
        self._task_pool = {}
        self._queue = PriorityQueue()
        self._submit_stack = []
        self.mutex = set()
        self.executor = None

    def bootstrap(self,
                  executor: Executor,
                  url: Optional[str] = None,
                  data_path: Union[str, Path] = Path.home() / 'data',
                  debug_mode: bool = False):
        """
        Bootstrap the scheduler.

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
        data_path : str
            The path to the data directory.
        debug_mode : bool
            Whether to enable debug mode.
        """
        if self.executor is not None:
            return
        self.executor = executor
        if url is None:
            url = 'sqlite:///{}'.format(data_path / 'waveforms.db')
        self.db_url = url
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        set_data_path(self.data_path)
        if url.startswith('sqlite'):
            self.eng = create_engine(url,
                                     echo=debug_mode,
                                     poolclass=SingletonThreadPool,
                                     connect_args={'check_same_thread': False})
        else:
            self.eng = create_engine(url, echo=debug_mode)
        create_tables(self.eng)

        self.system_user = self.verify_user('BIG BROTHER', self.__uuid)

        set_sessionmaker(sessionmaker(bind=self.eng))
        setup_ipy_events()

        self._main_loop_thread = threading.Thread(target=submit_loop,
                                                  args=(self._queue,
                                                        self._submit_stack),
                                                  daemon=True)
        self._main_loop_thread.start()

    def verify_user(self, username: str, password: str) -> User:
        with self.session() as db:
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

    def login(self, username: str, password: str) -> Terminal:
        user = self.verify_user(username, password)
        return Terminal(self, user)

    def add_user(self, username: str, password: str):
        db = self.session()
        try:
            user = db.query(User).filter(User.name == username).one()
        except NoResultFound:
            user = User(name=username)
            db.add(user)
            db.commit()
        else:
            raise ValueError('User already exists')
        user.setPassword(password)
        db.commit()

    @property
    def cfg(self):
        return self.executor.cfg

    def session(self):
        return sessionmaker(bind=self.eng)()

    def db(self):
        return sessionmaker(bind=self.eng)()

    def get_task_by_id(self, task_id):
        try:
            return self._task_pool.get(task_id)()
        except:
            return None

    def list_tasks(self):
        return {id: ref() for id, ref in self._task_pool.items()}

    def cancel(self):
        while not self._queue.empty():
            task = self._queue.get_nowait()
            task.cancel()
        while self._submit_stack:
            task = self._submit_stack.pop()
            task.cancel()
        self.executor.cancel()
        while self._waiting_result:
            task_id, task = self._waiting_result.popitem()
            task.cancel()

    def join(self, task, timeout=None):
        warnings.warn(
            'Scheduler.join(task) is deprecated, use task.join() instead',
            DeprecationWarning, 2)
        task.join(timeout)

    def set(self, key: str, value: Any, cache: bool = False):
        cmds = []
        if not cache:
            cmds.append(WRITE(key, value))
        if len(cmds) > 0:
            succeed = self.executor.feed(0, -1, cmds, next_feed_time=0)
            if succeed == 0:
                raise RuntimeError(
                    f'Failed to set {key} to {value}, executor busy.')
            if succeed == 1:
                self.executor.free(0)
        self.cfg.update(key, value, cache=cache)

    def get(self, key: str):
        """
        return the value of the key in the kernel config
        """
        return self.cfg.query(key)

    def generate_task_id(self):
        i = uuid.uuid3(self.__uuid, f"{next(self.counter)}").int
        return i & ((1 << 64) - 1)

    def exec(self,
             circuit,
             signal='state',
             shots=1024,
             arch='baqis',
             lib=None,
             cfg=None,
             cmds=[],
             no_record=False):
        """Execute a circuit.
        
        Parameters:
            circuit: a QLisp Circuit.
            signal: a str of the name of the signal type to be returned.
            shots: the number of shots to be executed.
            lib: a Library used by compiler,default is stdlib.
            cfg: configuration of system.
            cmds: additional commands.

        Returns:
            A Task.
        """
        from waveforms.sched.task import RunCircuits

        t = self.create_task(RunCircuits,
                             kwds=dict(circuits=[circuit],
                                       shots=shots,
                                       signal=signal,
                                       arch=arch,
                                       lib=lib,
                                       cfg=cfg,
                                       cmds=cmds))
        t.no_record = no_record
        self.submit(t)
        return t

    def measure(self, keys, labels=None, cmds=[]):
        pass

    def update_parameters(self, parameters: dict[str, Any]):
        """Update parameters.

        Args:
            parameters: a dict of parameters.
        """
        for key, value in parameters.items():
            self.set(key, value)
        self.cfg.clear_buffer()

    def maintain(self, task: Task) -> Task:
        """Maintain a task.
        """
        from ._bigbrother import maintain

        return maintain(self, task)

    def submit(self, task: Task, dry_run: bool = False, config=None) -> Task:
        """Submit a task.
        """
        with task.runtime._status_lock:
            if task.status != 'not submited':
                raise RuntimeError(
                    f'Task({task.id}, status={task.status}) has been submited!'
                )
        if task.kernel is None:
            taskID = self.generate_task_id()
            task._set_kernel(self, taskID)
        task.runtime.threads.update({
            'compile':
            ThreadWithKill(target=task.main, name=f"Compile-{task.id}"),
            'run':
            ThreadWithKill(target=task_run_thread,
                           name=f"Run-{task.id}",
                           args=(task, self.executor))
        })
        if config is not None:
            task.runtime.prog.snapshot = config
        else:
            task.runtime.prog.snapshot = self.cfg.export()
        with task.runtime._status_lock:
            if not task.runtime.prog.with_feedback:
                task.runtime.threads['compile'].start()
                task.runtime.status = 'compiling'
            else:
                task.runtime.status = 'pending'

        if dry_run:
            return task

        self._queue.put_nowait(task)

        def delete(ref, dct, key):
            dct.pop(key)

        self._task_pool[task.id] = weakref.ref(
            task, functools.partial(delete, dct=self._task_pool, key=task.id))
        return task

    def query(self, key):
        warnings.warn('query is deprecated, use get instead',
                      DeprecationWarning, 2)
        return self.get(key)

    def update(self, key, value, cache=False):
        warnings.warn('update is deprecated, use set instead',
                      DeprecationWarning, 2)
        self.set(key, value)

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
        taskID = self.generate_task_id()
        task._set_kernel(self, taskID)
        task.runtime.user = self.system_user
        return task


scheduler = Scheduler()


def bootstrap(executor: Executor,
              url: Optional[str] = None,
              data_path: Union[str, Path] = Path.home() / 'data',
              debug_mode: bool = False):
    """
    Bootstrap the scheduler.

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
    data_path : str
        The path to the data directory.
    debug_mode : bool
        Whether to enable debug mode.
    """
    scheduler.bootstrap(executor, url, data_path, debug_mode)
    return scheduler


def login(username, password):
    """
    Login to the scheduler.

    Parameters
    ----------
    username : str
        The username to login with.
    password : str
        The password to login with.
    """
    return scheduler.login(username, password)
