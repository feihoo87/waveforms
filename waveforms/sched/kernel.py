import asyncio
import itertools
import logging
import multiprocessing
import multiprocessing.dummy
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from waveforms import compile as Qcompile
from waveforms import getConfig, stdlib
from waveforms.backends.quark import set_up_backend
from waveforms.backends.quark.executable import assymblyData, getCommands

TASKLIMIT = 1


@dataclass
class _Task():
    cmds: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)


class Executor():
    def __init__(self):
        from waveforms import setConfig

        setConfig('/Users/feihoo87/Nutstore Files/baqis/代码程序/XHK/config.json')

        self._pool = defaultdict(_Task)
        self._queue = deque()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def feed(self, task_id, task_step, keys, values):
        self._queue.append([task_id, task_step, keys, values])
        self._pool[task_id].cmds[task_step] = [keys, values]

    def _run(self):
        while True:
            try:
                task_id, task_step, keys, values = self._queue.popleft()
                time.sleep(0.5)
                self._feed_random_result(task_id, task_step, keys, values)
            except:
                time.sleep(0.2)

    def _feed_random_result(self, task_id, task_step, keys, values):
        import numpy as np

        ret = {}
        for key, value in zip(keys, values):
            if value is None:
                ret[key] = [np.random.randn(1024, 3), np.random.randn(1024, 3)]
        self._pool[task_id].results[task_step] = ret

    def free(self, task_id):
        try:
            self._pool.pop(task_id)
        except KeyError:
            pass

    def submit(self, task_id, data_template):
        pass

    def result(self, task_id, skip=0):
        keys = sorted(self._pool[task_id].results.keys())
        return [self._pool[task_id].results[key] for key in keys[skip:]]


class QuarkExcutor(Executor):
    def __init__(self, host):
        self.host = host
        self._conn_pool = {}
        self.connect()
        self._gct = threading.Thread(target=self._gc, daemon=True)
        self._gct.start()

    def _gc(self):
        while True:
            time.sleep(1)
            alived = [t.ident for t in threading.enumerate()]
            for tid in list(self._conn_pool.keys()):
                if tid not in alived:
                    del self._conn_pool[tid]

    @property
    def conn(self):
        from quark import connect
        tid = threading.get_ident()
        if tid not in self._conn_pool:
            self._conn_pool[tid] = connect('QuarkServer', host=self.host)
        return self._conn_pool[tid]

    def connect(self):
        set_up_backend(self.host)

    def feed(self, task_id, task_step, keys, values):
        #print(task_id, task_step, keys)
        self.conn.feed(task_id, task_step, keys, values)

    def free(self, task_id):
        self.conn.free(task_id)

    def submit(self, task_id, data_template):
        self.conn.submit(task_id, data_template)

    def result(self, task_id, skip=0):
        ret = self.conn.result(task_id, skip)
        if ret is None:
            return []
        return ret

    def save(self, path, task_id, data):
        print('save', path, task_id)
        import pickle
        self.conn.save(path, task_id, pickle.dumps(data))


class Kernel():
    def __init__(self, excuter):
        self.counter = itertools.count()
        self.uuid = uuid.uuid1()
        self._queue = deque()
        self._waiting_result = {}
        self._submit_stack = []
        self.excuter = excuter
        self.cfg = getConfig()

        self._waiting_thread = threading.Thread(target=self._waiting,
                                                daemon=True)
        self._waiting_thread.start()

        self._submit_thread = threading.Thread(target=self._submit,
                                               daemon=True)
        self._submit_thread.start()

    def _get_next_task(self):
        try:
            return self._queue.popleft()
        except IndexError:
            return None

    def _submit(self):
        def is_children_of(parent, child):
            return (child.parent is not None and child.parent == parent.id)

        while True:
            if len(self._submit_stack) > 0:
                current_task, thread = self._submit_stack.pop()

                if thread.is_alive():
                    self._submit_stack.append((current_task, thread))
                else:
                    t = threading.Thread(target=self._join,
                                         args=(current_task, ))
                    t.start()
                    current_task._runtime.status = 'running'
                    self._waiting_result[current_task.id] = current_task, t

            task = self._get_next_task()
            if task is None:
                time.sleep(1)
                continue

            if (len(self._submit_stack) == 0
                    or is_children_of(self._submit_stack[-1][0], task)):
                self._run_submit(task)
            else:
                self._queue.append(task)

    def _run_submit(self, task):
        self.excuter.free(task.id)
        logging.info(f'free({task.id})')
        self.excuter.submit(task.id, {})
        logging.info(f'submit({task.id}, dict())')
        task._runtime.status = 'submiting'
        thread = threading.Thread(target=task.main)
        self._submit_stack.append((task, thread))
        thread.start()

    def _waiting(self):
        while True:
            for taskID, (task, thread) in list(self._waiting_result.items()):
                if not thread.is_alive():
                    try:
                        del self._waiting_result[taskID]
                    except:
                        pass
                    task._runtime.status = 'finished'
            time.sleep(0.01)

    def cancel(self):
        self._queue.clear()
        for taskID, (task, proc) in list(self._submitting.items()):
            if not proc.is_alive():
                try:
                    proc.terminate()
                    del self._submitting[taskID]
                except:
                    pass

                task._runtime.status = 'canceled'
                t = threading.Thread(target=self._join, args=(task, ))
                t.start()
                self._waiting_result[taskID] = task, t
            for taskID, (task, thread) in list(self._waiting_result.items()):
                if not thread.is_alive():
                    try:
                        del self._waiting_result[taskID]
                    except:
                        pass
                    task._runtime.status = 'finished'

    def _join(self, task):
        while True:
            time.sleep(1)
            try:
                if len(task.result()['data']) == task._runtime.step:
                    self.excuter.save(task.data_path(), task.id, task.result())
                    self.excuter.free(task.id)
                    break
            except:
                raise

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
            task._runtime.result['index'].append(args)
            task._runtime.cmds.clear()
            yield args
            task._runtime.step += 1

    def exec(self, task, circuit, lib=None, cfg=None):
        if lib is None:
            lib = stdlib
        if cfg is None:
            try:
                self.cfg.clear_buffer()
            except AttributeError:
                pass
            cfg = self.cfg

        code = Qcompile(circuit, lib=lib, cfg=cfg)
        cmds, dataMap = getCommands(code)
        task._runtime.dataMaps.append(dataMap)
        cmds.extend(task._runtime.cmds)

        a, b = list(zip(*cmds))
        self.excuter.feed(task.id, task._runtime.step, list(a), list(b))
        logging.info(
            f'feed({task.id}, {task._runtime.step}, {list(a)}, {list(b)})')

    def run_sub_task(self, task, sub_task):
        sub_task.parent = task.id
        self.submit(sub_task)
        self.join(sub_task)
        ret = sub_task.result()
        task._runtime.data.append(ret['data'])
        return ret

    def result(self, task, skip=0):
        logging.info(f'result({task.id})')
        return self.excuter.result(task.id, skip)

    def submit(self, task):
        taskID = self.generate_task_id()
        task.id = taskID
        task.kernel = self
        task._runtime.status = 'pending'
        self._queue.append(task)

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


@dataclass
class AppRuntime():
    step: int = 0
    status: str = 'not submited'
    dataMaps: list = field(default_factory=list)
    data: list = field(default_factory=list)
    cmds: list = field(default_factory=list)
    feedback_buffer: Any = None
    result: dict = field(default_factory=lambda: {
        'index': [],
        'states': [],
        'counts': [],
        'diags': []
    })


class App():
    def __init__(self):
        self.parent = None
        self.id = None
        self.kernel = None
        self.signal = 'count'
        self._runtime = AppRuntime()

    def __del__(self):
        try:
            self.kernel.excuter.free(self.id)
        except:
            pass

    def set(self, key, value):
        self._runtime.cmds.append((key, value))

    def get(self, key):
        self.kernel.query(key)

    def data_path(self):
        name = self.__class__.__name__
        return f"Test:/Q1/{name}"

    def scan_range(self):
        pass

    def main(self):
        pass

    def result(self):
        try:
            i = len(self._runtime.data)
            a = self.kernel.result(self, i)
            for raw_data, dataMap in zip(a, self._runtime.dataMaps[i:]):
                result = assymblyData(raw_data, dataMap, self.signal)
                self._runtime.data.append(result['data'])
                self._runtime.result['states'].append(result.get(
                    'state', None))
                self._runtime.result['counts'].append(result.get(
                    'count', None))
                self._runtime.result['diags'].append(result.get('diag', None))
            return {
                'index': self._runtime.result['index'],
                'data': self._runtime.data,
                'states': self._runtime.result['states'],
                'counts': self._runtime.result['counts'],
                'diags': self._runtime.result['diags']
            }
        except:
            return {
                'index': [],
                'data': [],
                'states': [],
                'counts': [],
                'diags': []
            }
