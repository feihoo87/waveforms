import asyncio
import functools
import itertools
import logging
import pickle
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import deque


class _COMMANDREAD():
    pass


READ = _COMMANDREAD()


class Executor(ABC):
    @abstractmethod
    def feed(self, task_id, task_step, keys, values):
        pass

    @abstractmethod
    def free(self, task_id):
        pass

    @abstractmethod
    def submit(self, task_id, data_template):
        pass

    @abstractmethod
    def result(self, task_id, skip=0):
        pass

    @abstractmethod
    def save(self, path, task_id, data):
        pass


class _ThreadWithKill(threading.Thread):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self._kill_event = threading.Event()

    def kill(self):
        self._kill_event.set()


class Scheduler():
    def __init__(self, excuter):
        from waveforms import getConfig

        self.counter = itertools.count()
        self.uuid = uuid.uuid1()
        self._task_pool = {}
        self._queue = deque()
        self._waiting_result = {}
        self._submit_stack = []
        self.excuter = excuter
        self.cfg = getConfig()

        self._read_data_thread = threading.Thread(target=self._read_data_loop,
                                                  daemon=True)
        self._read_data_thread.start()

        self._submit_thread = threading.Thread(target=self._submit_loop,
                                               daemon=True)
        self._submit_thread.start()

    def _get_next_task(self):
        try:
            return self._queue.popleft()
        except IndexError:
            return None

    def _submit_loop(self):
        while True:
            if len(self._submit_stack) > 0:
                current_task, thread = self._submit_stack.pop()

                if thread.is_alive():
                    self._submit_stack.append((current_task, thread))
                else:
                    t = _ThreadWithKill(target=self._join,
                                        args=(current_task, ))
                    t.start()
                    current_task._runtime.status = 'running'
                    self._waiting_result[current_task.id] = current_task, t

            task = self._get_next_task()
            if task is None:
                time.sleep(1)
                continue

            if (len(self._submit_stack) == 0
                    or task.is_children_of(self._submit_stack[-1][0])):
                self._submit(task)
            else:
                self._queue.append(task)

    def _submit(self, task):
        self.excuter.free(task.id)
        logging.info(f'free({task.id})')
        self.excuter.submit(task.id, {})
        logging.info(f'submit({task.id}, dict())')
        task._runtime.status = 'submiting'
        thread = _ThreadWithKill(target=task.main)
        self._submit_stack.append((task, thread))
        thread.start()

    def _read_data_loop(self):
        while True:
            for taskID, (task, thread) in list(self._waiting_result.items()):
                if not thread.is_alive():
                    try:
                        del self._waiting_result[taskID]
                    except:
                        pass
                    task._runtime.status = 'finished'
            time.sleep(0.01)

    def _join(self, task):
        while True:
            time.sleep(1)
            try:
                if len(task.result()['data']) == task._runtime.step:
                    result = {
                        key: {
                            'data': value
                        }
                        for key, value in task.result().items()
                        if key not in ['counts', 'diags']
                    }
                    self.excuter.save(task.data_path(), task.id,
                                      pickle.dumps(result))
                    self.excuter.free(task.id)
                    break
            except:
                raise

    def get_task_by_id(self, task_id):
        try:
            return self._task_pool.get(task_id)()
        except:
            return None

    def cancel(self):
        self.excuter.cancel()
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

    async def join_async(self, task):
        while True:
            if task._runtime.status == 'finished':
                break
            await asyncio.sleep(1)

    def generate_task_id(self):
        i = uuid.uuid3(self.uuid, f"{next(self.counter)}").int
        return i & ((1 << 64) - 1)

    def scan(self, task):
        task._runtime.step = 0
        for args in task.scan_range():
            try:
                if threading.current_thread()._kill_event.is_set():
                    break
            except AttributeError:
                pass
            task._runtime.result['index'].append(args)
            task._runtime.cmds.clear()
            yield args
            task._runtime.step += 1

    def _exec(self, task, circuit, lib=None, cfg=None):
        from waveforms import compile, stdlib
        from waveforms.backends.quark.executable import getCommands

        if lib is None:
            lib = stdlib
        if cfg is None:
            try:
                self.cfg.clear_buffer()
            except AttributeError:
                pass
            cfg = self.cfg

        code = compile(circuit, lib=lib, cfg=cfg)
        cmds, dataMap = getCommands(code)
        task._runtime.dataMaps.append(dataMap)
        cmds.extend(task._runtime.cmds)

        self.excuter.feed(task.id, task._runtime.step, *zip(*cmds))
        logging.info(
            f'feed({task.id}, {task._runtime.step}, <{len(cmds)} commands ...>)'
        )

    def exec(self, circuit, lib=None, cfg=None):
        pass

    def _measure(self, task, keys, labels=None):
        cmds = [(key, READ) for key in keys]
        cmds.extend(task._runtime.cmds)

        self.excuter.feed(task.id, task._runtime.step, *zip(*cmds))
        logging.info(
            f'feed({task.id}, {task._runtime.step}, <{len(cmds)} commands ...>)'
        )

    def measure(self, keys, labels=None):
        pass

    def result(self, task, skip=0):
        logging.info(f'result({task.id})')
        return self.excuter.result(task.id, skip)

    def submit(self, task):
        taskID = self.generate_task_id()
        task.id = taskID
        task.kernel = self
        task._runtime.status = 'pending'
        self._queue.append(task)

        def delete(ref, dct, key):
            dct.pop(key)

        self._task_pool[task.id] = weakref.ref(
            task, functools.partial(delete, dct=self._task_pool, key=task.id))

    def get_config(self):
        return self.cfg

    def query(self, key):
        return self.excuter.query(key)

    def update(self, key, value):
        self.excuter.update(key, value)
        self.cfg.clear_buffer()

    def feedback(self, task, obj):
        task._runtime.feedback_buffer = obj

    def get_feedback(self, task):
        obj = task._runtime.feedback_buffer
        task._runtime.feedback_buffer = None
        return obj

    def create_task(self, app, args=(), kwds={}):
        if isinstance(app, str):
            app = self._getAppClass(app)
        task = app(*args, **kwds)
        return task

    def _getAppClass(self, name):
        pass
