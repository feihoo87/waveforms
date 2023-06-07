import asyncio
import copy
import functools
import hashlib
import itertools
import logging
import os
import pickle
import platform
import sys
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
from waveforms.dicttree import NOTSET

from qlisp import WRITE
from storage.fs import set_data_path
from storage.git import get_heads_of_repositories, track_repository
from storage.ipy_events import set_sessionmaker, setup_ipy_events
from storage.models import User, create_tables

from .base import Executor
from .task import Task
from .task import create_task as _create_task

log = logging.getLogger(__name__)

__counter = itertools.count()
__uuid = uuid.uuid1()
__task_pool = weakref.WeakValueDictionary()
__queue = PriorityQueue()
__submit_stack = []
__submit_stack_lock = threading.Lock()
__mutex = set()
__executor = None
__db_url = None
__data_path = None
__debug_mode = None
__eng = None
__system_user = None
__repositories = {}
__system_info = {
    'OS': platform.uname()._asdict(),
    'Python': sys.version,
    'packages': [],
    'repositories': {}
}
__system_hooks = {
    'before_task_start': [],
    'after_task_finished': [],
    'before_task_step': [],
}


def register_hook(hook_name, func):
    if func not in __system_hooks[hook_name]:
        __system_hooks[hook_name].append(func)
    return func


def unregister_hook(hook_name, func):
    if func in __system_hooks[hook_name]:
        __system_hooks[hook_name].remove(func)
    return func


def unregister_all_hooks(hook_name):
    if hook_name is None:
        for k in __system_hooks.keys():
            unregister_all_hooks(k)
    elif hook_name in __system_hooks:
        __system_hooks[hook_name].clear()


def before_task_start(func):
    return register_hook('before_task_start', func)


def after_task_finished(func):
    return register_hook('after_task_finished', func)


def before_task_step(func):
    return register_hook('before_task_step', func)


def _stack():
    return __submit_stack.copy()


def _clear_stack():
    __submit_stack.clear()


def get_system_info():
    from pip._internal.operations.freeze import freeze

    info = __system_info
    info['packages'] = list(freeze())
    info['repositories'] = get_heads_of_repositories(__repositories)
    return info


def get_executor():
    return __executor


def get_config():
    return __executor.cfg


def clean_side_effects(task: Task,
                       executor: Executor,
                       keep_last_status: bool = False):
    if keep_last_status:
        return
    cmds = []
    for k, v in task.runtime.prog.side_effects.items():
        if isinstance(v, tuple) and len(v) == 2 and v[0] is NOTSET:
            pass
        else:
            cmds.append(WRITE(k, v))
    return executor.feed(task.id,
                         -2,
                         cmds,
                         priority=task.task_priority,
                         name=task.name,
                         next_feed_time=0)


def submit_loop(task_queue: PriorityQueue, current_stack: list[Task],
                stack_lock: threading.Lock):
    while True:
        with stack_lock:
            if len(current_stack) > 0:
                current_task = current_stack.pop()
                if current_task.status in ['cancelled', 'failed', 'finished']:
                    for fut, evt in current_task.runtime.threads.values():
                        evt.set()
                elif current_task.runtime.threads['run'][0].running():
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
        else:
            with stack_lock:
                if (len(current_stack) == 0
                        or task.is_children_of(current_stack[-1])):
                    _submit(task, current_stack)
                    push_back = False
                else:
                    push_back = True
            if push_back:
                task_queue.put_nowait(task)
                time.sleep(0.1)


def _submit(task: Task, current_stack: list[Task]):
    with task.runtime._status_lock:
        task.runtime.status = 'submiting'
    if task.runtime.prog.with_feedback:
        kill_evt = threading.Event()
        task.runtime.threads['compile'] = (__executor.thread_pool.submit(
            task_compile_thread, task, kill_evt), kill_evt)

    current_stack.append(task)
    task.runtime.started_time = time.time()
    if not task.runtime.prog.with_feedback:
        kill_evt = threading.Event()
        task.runtime.threads['run'] = (__executor.thread_pool.submit(
            task_run_thread, task, kill_evt, __executor), kill_evt)


__main_loop_thread = threading.Thread(target=submit_loop,
                                      args=(__queue, __submit_stack,
                                            __submit_stack_lock),
                                      daemon=True)


def _feed_step(task, feed_step, executor):
    data_map = copy.copy(task.runtime.prog.steps[feed_step].data_map)

    extra = {'dataMap': data_map}

    while True:
        succeed = executor.feed(task.id,
                                feed_step,
                                task.runtime.prog.steps[feed_step].cmds,
                                extra,
                                priority=task.task_priority,
                                name=task.name,
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


def _fetch_data(task: Task, executor: Executor):
    skip = task.runtime.finished_step
    additional = executor.fetch(task.id, skip)
    if isinstance(additional, str):
        additional = []
    for step, (result,
               prog_frame) in enumerate(zip(additional,
                                            task.runtime.prog.steps[skip:]),
                                        start=skip):
        prog_frame.fut.set_result(result)
        prog_frame.step.feed(result, store=True)
        task.runtime.finished_step += 1
        task.runtime.progress.goto(task.runtime.finished_step)
    return task.runtime.finished_step


def _save_tmp_data(task, skip):
    return
    dir_path = Path.home() / 'systemq' / 'current'
    dir_path.mkdir(parents=True, exist_ok=True)
    temp_path = dir_path / f"{task.id}.{skip}"
    index_path = dir_path / f"{task.id}"
    if index_path.exists():
        with index_path.open('rb') as f:
            index = pickle.load(f)
    else:
        index = []
    data = task.result(False, skip)
    if skip == 0 and task.debug_mode:
        data['program'] = task.runtime.prog
    with open(temp_path, 'wb') as f:
        pickle.dump(data, f)
    index.append(temp_path)
    with open(index_path, 'wb') as f:
        pickle.dump(index, f)
    pass


def _clear_tmp_data(task):
    dir_path = Path.home() / 'systemq' / 'current'
    index_path = dir_path / f"{task.id}"
    if index_path.exists():
        with index_path.open('rb') as f:
            index = pickle.load(f)
        for path in index:
            path.unlink()
        index_path.unlink()


def _save_data(task,
               temp=False,
               last_time=None,
               flush_time=None,
               saved_step=0):
    t = time.time()
    if temp and (t - last_time) >= flush_time:
        _save_tmp_data(task, saved_step)
        saved_step = task.runtime.finished_step
        return t, saved_step
    elif not temp and task.runtime.record is not None:
        try:
            data = task.result(task.reshape_record)
            if task.debug_mode:
                data['program'] = task.runtime.prog
            task.runtime.record.data = data
            task.db.commit()
            _clear_tmp_data(task)
        except Exception as e:
            log.error(f"Failed to save record: {e}")
    elif not temp:
        log.warning(f"No record for task {task.name}({task.id})")
    return last_time, saved_step


def _exec_hooks(task, step, executor, hooks):
    for hook in hooks:
        hook(task, step, executor)
    time.sleep(0.01)


def task_compile_thread(task: Task, kill_event: threading.Event):
    task.runtime._kill_event = kill_event
    task.main()


def task_run_thread(task: Task, kill_event: threading.Event,
                    executor: Executor):
    feed_step = 0
    feed_more = True
    feed_finished = False
    side_effect_cleared = False
    last_save_time = time.time()
    saved_step = 0

    if task.task_priority == 15:
        user = 'quafu'
    elif task.task_priority == 25:
        user = 'huawei'
    else:
        user = 'baqis'

    if hasattr(task, 'circuits'):
        circuit = task.circuits
    else:
        circuit = []
    executor.create_task(
        task.id, {
            'priority': task.task_priority,
            'user': user,
            'name': task.name,
            'circuit': circuit
        })
    if task._init_hooks:
        _exec_hooks(task, feed_step, executor, task._init_hooks)

    try:
        while True:
            if kill_event.is_set():
                with task.runtime._status_lock:
                    task.runtime.status = 'cancelled'
                task.runtime.progress.finish(False)
                break
            if (not feed_finished and feed_step >= task.runtime.compiled_step
                    and not task.runtime.threads['compile'][0].running()):
                clean_side_effects(task, executor,
                                   task.runtime.keep_last_status)
                feed_more = False
                feed_finished = True
                side_effect_cleared = True
                with task.runtime._status_lock:
                    if task.status in ['submiting', 'pending', 'compiling']:
                        task.runtime.status = 'running'
            if (feed_finished
                    or (feed_step >= len(task.runtime.prog.steps)
                        or len(task.runtime.prog.steps[feed_step].cmds) == 0)
                    and task.runtime.threads['compile'][0].running()):
                feed_more = False
            elif task._hooks and feed_step > saved_step:
                feed_more = False
            else:
                feed_more = True

            if feed_more:
                if task._hooks:
                    _exec_hooks(task, feed_step, executor, task._hooks)
                _feed_step(task, feed_step, executor)
                feed_step += 1

            if _fetch_data(task, executor) > saved_step:
                last_save_time, saved_step = _save_data(
                    task, True, last_save_time, 10, saved_step)
            if (feed_finished and
                    task.runtime.finished_step >= task.runtime.compiled_step):
                # executor.save(task.id, task.data_path)
                executor.save(
                    task.id,
                    f'{task.data_path}/{task.record.id}',  # save checkpoint once the Task is finished
                    task.result()['index'])
                task.runtime.finished_time = time.time()
                with task.runtime._status_lock:
                    task.runtime.status = 'finished'
                task.runtime.progress.finish(True)
                break
            if not feed_more:
                time.sleep(1)
    except:
        with task.runtime._status_lock:
            task.runtime.status = 'failed'
        task.runtime.progress.finish(False)
        log.exception(f"{task.name}({task.id}) is failed")
        executor.free(task.id)
    finally:
        if task._final_hooks:
            _exec_hooks(task, feed_step, executor, task._final_hooks)
        if not side_effect_cleared:
            clean_side_effects(task, executor, task.runtime.keep_last_status)
        _save_data(task)
        log.debug(f'{task.name}({task.id}) is finished')


def bootstrap(executor: Executor,
              url: Optional[str] = None,
              data_path: Union[str, Path] = Path.home() / 'data',
              repositories: Optional[dict[str, Union[str, Path]]] = None,
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
    repositories : dict[str, str]
        The repositories to use.
    debug_mode : bool
        Whether to enable debug mode.
    """
    global __executor, __db_url, __data_path, __debug_mode, __eng, __system_user

    if __executor is not None:
        return
    __executor = executor
    if url is None:
        url = 'sqlite:///{}'.format(data_path / 'waveforms.db')
    __db_url = url
    __data_path = Path(data_path)
    __data_path.mkdir(parents=True, exist_ok=True)
    set_data_path(__data_path)
    if url.startswith('sqlite'):
        __eng = create_engine(url,
                              echo=debug_mode,
                              poolclass=SingletonThreadPool,
                              connect_args={'check_same_thread': False})
    else:
        __eng = create_engine(url, echo=debug_mode)
    create_tables(__eng)

    if repositories is not None:
        for name, url in repositories.items():
            __repositories[name] = track_repository(
                name, url, __data_path / 'repositories')

    __system_user = verify_user('BIG BROTHER', __uuid)

    set_sessionmaker(sessionmaker(bind=__eng))
    setup_ipy_events()
    __main_loop_thread.start()


def verify_user(username: str, password: str) -> User:
    with session() as db:
        if username == 'BIG BROTHER' and password == __uuid:
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


def session():
    return sessionmaker(bind=__eng)()


def get_task_by_id(task_id):
    try:
        return __task_pool[task_id]
    except:
        return None


def list_tasks():
    return list(__task_pool.keys())


def set(key: str, value: Any, cache: bool = False):
    if key in __executor._state_caches:
        __executor._state_caches.pop(key)
    cmds = []
    if not cache:
        cmds.append(WRITE(key, value))
    if len(cmds) > 0:
        succeed = __executor.feed(0, -1, cmds, next_feed_time=0)
        if succeed == 0:
            raise RuntimeError(
                f'Failed to set {key} to {value}, executor busy.')
        if succeed == 1:
            __executor.free(0)
    get_config().update(key, value, cache=cache)


def get(key: str, default: Any = NOTSET):
    """
    return the value of the key in the kernel config
    """
    ret = get_config().query(key)
    if isinstance(ret, tuple) and ret[0] is NOTSET:
        if default is NOTSET:
            raise KeyError(f'Key {key} not found')
        return default
    return ret


def generate_task_id():
    i = uuid.uuid3(__uuid, f"{next(__counter)}").int
    return i & ((1 << 64) - 1)


def update_parameters(parameters: dict[str, Any]):
    """Update parameters.

    Args:
        parameters: a dict of parameters.
    """
    for key, value in parameters.items():
        set(key, value)
    get_config().clear_buffer()


def create_task(app, args=(), kwds={}):
    """
    create a task from a string or a class

    Args:
        app: a string or a class
        args: arguments for the class
        kwds: keyword arguments for the class

    Returns:
        a task
    """
    task = _create_task((app, args, kwds))
    task.runtime.id = generate_task_id()
    task.runtime.user = __system_user
    task.runtime.system_info = get_system_info()
    for fun in __system_hooks['before_task_start']:
        if fun not in task._init_hooks:
            task._init_hooks.append(fun)
    for fun in __system_hooks['before_task_step']:
        if fun not in task._hooks:
            task._hooks.append(fun)
    for fun in __system_hooks['after_task_finished']:
        if fun not in task._final_hooks:
            task._final_hooks.append(fun)
    return task


def submit(task: Task, dry_run: bool = False, config=None) -> Task:
    """Submit a task.
    """
    get_executor()._state_caches.clear()
    with task.runtime._status_lock:
        if task.status != 'not submited':
            raise RuntimeError(
                f'Task({task.id}, status={task.status}) has been submited!')
    if task.runtime.id is None:
        task.runtime.id = generate_task_id()
    if config is not None:
        task.runtime.prog.snapshot = config
    else:
        task.runtime.prog.snapshot = get_config().export()
    with task.runtime._status_lock:
        if not task.runtime.prog.with_feedback:
            kill_evt = threading.Event()
            task.runtime.threads['compile'] = (__executor.thread_pool.submit(
                task_compile_thread, task, kill_evt), kill_evt)
            task.runtime.status = 'compiling'
        else:
            task.runtime.status = 'pending'

    if dry_run:
        task.runtime.dry_run = True
        return task

    __task_pool[task.id] = task
    __queue.put_nowait(task)

    return task


def maintain(task: Task) -> Task:
    """Maintain a task.
    """
    from ._bigbrother import maintain

    return maintain(task)


def exec(circuit,
         signal='state',
         shots=1024,
         arch='baqis',
         lib=None,
         cfg=None,
         cmds=[],
         no_record=False,
         parent=None,
         dry_run: bool = False,
         **kw):
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
    if isinstance(circuit, list) and isinstance(circuit[0], tuple):
        circuits = [circuit]
    elif isinstance(circuit, list) and isinstance(
            circuit[0], list) and isinstance(circuit[0][0], tuple):
        circuits = circuit
    else:
        raise TypeError('circuit must be a list of tuples')

    t = create_task('RunCircuits',
                    kwds=dict(circuits=circuits,
                              shots=shots,
                              signal=signal,
                              arch=arch,
                              lib=lib,
                              cfg=cfg,
                              cmds=cmds) | kw)
    if parent is not None:
        t.parent = parent.id
        t.runtime.prog.snapshot = parent.cfg.export()
    t.no_record = no_record
    if kw.get('tid', 0):
        t.runtime.id = kw['tid']
    submit(t, dry_run=dry_run)
    return t


def measure(keys, labels=None, cmds=[]):
    pass


def cancel():
    while not __queue.empty():
        __queue.get_nowait().cancel()
    with __submit_stack_lock:
        while __submit_stack:
            __submit_stack.pop().cancel()
    __task_pool.clear()
    __executor.cancel()
