import asyncio
import functools
import importlib
import itertools
import logging
import os
import sys
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import deque
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from waveforms.storage.models import create_tables
from waveforms.waveform import Waveform

from .task import CalibrationResult, Task

_COMMANDS = uuid.UUID('urn:uuid:308c54dc-c13d-4e52-9caa-3a05a7bc7fa8')

READ = uuid.uuid3(_COMMANDS, "READ")
TRIG = uuid.uuid3(_COMMANDS, "TRIG")


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
    def fetch(self, task_id, skip=0):
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
    def __init__(self, excuter: Executor, url: str = 'sqlite:///:memory:'):
        """
        Parameters
        ----------
        excuter : Executor
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
        from waveforms import getConfig

        self.counter = itertools.count()
        self.uuid = uuid.uuid1()
        self._task_pool = {}
        self._queue = deque()
        self._waiting_result = {}
        self._submit_stack = []
        self.excuter = excuter
        self.db = url
        self.eng = create_engine(url)
        if (self.db == 'sqlite:///:memory:' or self.db.startswith('sqlite:///')
                and not os.path.exists(self.db.removeprefix('sqlite:///'))):
            create_tables(self.eng)
        self.cfg = getConfig()

        self._read_data_thread = threading.Thread(target=self._read_data_loop,
                                                  daemon=True)
        self._read_data_thread.start()

        self._submit_thread = threading.Thread(target=self._submit_loop,
                                               daemon=True)
        self._submit_thread.start()

    def session(self):
        return sessionmaker(bind=self.eng)()

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
        task._runtime.started_time = time.time()
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
        import numpy as np

        try:
            while True:
                if threading.current_thread()._kill_event.is_set():
                    break
                time.sleep(1)
                if len(task.result()['data']) == task._runtime.step:
                    result = {
                        key: {
                            'data': np.asarray(value).tolist(),
                        }
                        for key, value in task.result().items()
                        if key not in ['counts', 'diags']
                    }
                    self.excuter.save(task.data_path(), task.id, {})
                    task._runtime.finished_time = time.time()
                    break
        finally:
            self.excuter.free(task.id)
            self.clean_side_effects(task)

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
        """Yield from task.scan_range().

        :param task: task to scan
        :return: a generator yielding step arguments.
        """
        task._runtime.step = 0
        for args in task.scan_range():
            try:
                if threading.current_thread()._kill_event.is_set():
                    break
            except AttributeError:
                pass
            task._runtime.result['index'].append(args)
            task._runtime.dataMaps.append({})
            task._runtime.cmds.clear()
            yield args
            task.trig()
            cmds = task._runtime.cmds
            self.excuter.feed(task.id, task._runtime.step, *zip(*cmds))
            task._runtime.side_effects.update(self.cfg._history)
            for k, v in task._runtime.cmds:
                if isinstance(v, Waveform):
                    task._runtime.side_effects[k] = 'zero()'
            logging.info(
                f'feed({task.id}, {task._runtime.step}, <{len(cmds)} commands ...>)'
            )
            task._runtime.step += 1

    def clean_side_effects(self, task):
        from .task import _is_feedable
        keys = []
        values = []
        for k, v in task._runtime.side_effects.items():
            if _is_feedable(k):
                keys.append(k)
                values.append(v)
            self.update(k, v)
        self.excuter.feed(task.id, -1, keys, values)
        task._runtime.side_effects.clear()
        self.cfg.clear_buffer()

    def _exec(self, task, circuit, lib=None, cfg=None, signal='state'):
        """Execute a circuit."""
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
        cmds, dataMap = getCommands(code, signal=signal)
        task._runtime.dataMaps[-1].update(dataMap)
        task._runtime.cmds.extend(cmds)

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
                for f in self.scan():
                    for cmd in cmds:
                        self.set(*cmd)
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
        for key, value in parameters.item():
            self.update(key, value)
        self.cfg.clear_buffer()

    def calibrate(self, task: Task) -> CalibrationResult:
        """Calibrate a task.

        Args:
            task: a task to be calibrated.
        Returns:
            A CalibrationResult.
        """
        raise NotImplementedError()
        task.calibration_level = 0
        self.submit(task)
        self.join(task)
        return task.analyze(task.result())

    def check_data(self, task: Task) -> CalibrationResult:
        raise NotImplementedError()
        task.calibration_level = 100
        self.submit(task)
        self.join(task)
        return task.analyze(task.result())

    def maintain(self, task: Task):
        """Maintain a task.
        """
        # recursive maintain
        for n in task.depends():
            self.maintain(self.create_task(*n))

        # check state
        success = task.chech_state()
        if success:
            return

        # check data
        result = self.chech_data(task)
        if result.in_spec:
            return
        elif result.bad_data:
            for n in task.depends():
                self._diagnose(self.create_task(*n))

        # calibrate
        result = self.calibrate(task)
        self.update_parameters(result.parameters)
        return

    def _diagnose(self, task: Task) -> bool:
        """
        Diagnose a task.

        Returns: True if node
        or dependent recalibrated.
        """
        # check data
        result = task.check_data()

        # in spec case
        if result.in_spec:
            return False

        # bad data case
        if result.bad_data:
            recalibrated = [
                self._diagnose(self.create_task(*n)) for n in task.depends()
            ]
        if not any(recalibrated):
            return False

        # calibrate
        result = self.calibrate(task)
        self.update_parameters(result)
        return True

    def fetch(self, task, skip=0):
        logging.info(f'fetch({task.id}, {skip})')
        return self.excuter.fetch(task.id, skip)

    def submit(self, task):
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

    def get_config(self):
        return self.cfg

    def query(self, key):
        return self.cfg.query(key)

    def update(self, key, value):
        self.excuter.update(key, value)

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
        *module, name = name.split('.')
        if len(module) == 0:
            module = sys.modules['__main__']
        else:
            module = '.'.join(module)
            module = importlib.import_module(module)
        return getattr(module, name)
