import logging
import queue
import re
import threading
import warnings
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, NamedTuple

from waveforms.waveform_parser import wave_eval

from qlisp import COMMAND, NOTSET, READ, SYNC, TRIG, WRITE, get_arch
from storage.ipy_events import get_current_cell_id

from ..config import QuarkConfig, QuarkLocalConfig
from .base import Executor

log = logging.getLogger(__name__)


def _is_feedable(cmd):
    if isinstance(cmd, WRITE):
        if cmd.address.startswith('gate.'):
            return False
        if re.match(r'[QCM]\d+\..+', cmd.address) and not re.match(
                r'[QCM]\d+\.(setting|waveform)\..+', cmd.address):
            return False
    return True


class QuarkClient():

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def __del__(self):
        self.stop()

    def connect(self):
        from srpc import connect
        self._conn = connect('QuarkServer', host=self.host, port=self.port)

    def rpc_call(self, api, *args, **kwds):
        return getattr(self._conn, api)(*args, **kwds)

    def call(self, api, *args, **kwds):
        if not self.is_alive():
            raise RuntimeError(
                f'rpc_call({api}, ...) failure. QuarkClient is not alive')
        fut = Future()
        self.queue.put_nowait((api, args, kwds, fut))
        return fut

    def stop(self):
        self._stop.set()

    def is_alive(self):
        return self._thread.is_alive()

    def run(self):
        self.connect()
        while True:
            if self._stop.is_set():
                break
            try:
                api, args, kwds, fut = self.queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                ret = self.rpc_call(api, *args, **kwds)
                fut.set_result(ret)
            except Exception as e:
                fut.set_result(e)


class QuarkExecutor(Executor):

    def __init__(self, host='127.0.0.1', port=2088, **kwds):
        self.host = host
        self.port = port
        self._state_caches = {}
        self._workers = defaultdict(lambda: QuarkClient(self.host, self.port))
        self._config_worker = self._workers['__config__']
        self.thread_pool = ThreadPoolExecutor()

    @property
    def conn(self):
        """rpc connection to quarkstudio
        """
        warnings.warn('conn is deprecated. use call_api instead',
                      DeprecationWarning,
                      stacklevel=2)
        return self._config_worker._conn

    @property
    def cfg(self):
        return QuarkConfig(server=self._config_worker._conn)

    def call_api(self, api, *args, **kwds):
        worker = self._workers[api]
        self.log.debug(f'{api}(*{args}, **{kwds})')
        return worker.call(api, *args, **kwds)

    def boot(self):
        pass

    def shutdown(self):
        pass

    def reset(self):
        pass

    def start(self, *args):
        self.call_api('start', *args)

    def create_task(self, task_id: int, meta: dict = {}):
        if not isinstance(meta, dict):
            meta = {}
        # try:
        #     meta['input_cell_id'] = get_current_cell_id()
        # except:
        #     meta['input_cell_id'] = None
        #     warnings.warn('create_task: failed to get current cell id',
        #                   RuntimeWarning,
        #                   stacklevel=2)
        self.call_api('reset', task_id, meta).result()

    def feed(self,
             task_id: int,
             task_step: int,
             cmds: list[COMMAND],
             extra: dict = {},
             priority: int = 0,
             name: str = '',
             next_feed_time: float = 5) -> None:
        """feed api of quark

        Args:
            task_id (int): uuid of task
            task_step (int): step of task (start from 0)
            cmds (list): commands to be executed
            extra (dict): extra data
        """
        commands = []

        writes = {}
        updates = {}
        others = []

        active_suffixs = ['.CaptureMode', '.StartCapture']
        active_prefixs = []

        for cmd in cmds:
            if _is_feedable(cmd):
                if isinstance(cmd, WRITE):
                    try:
                        if self._state_caches[cmd.address] == cmd.value\
                            and not any(cmd.address.endswith(suffix) for suffix in active_suffixs)\
                            and not any(cmd.address.startswith(prefix) for prefix in active_prefixs):
                            continue
                    except:
                        pass
                    self._state_caches[cmd.address] = cmd.value
                    if not (isinstance(cmd.value, tuple)
                            and cmd.value[0] is NOTSET):
                        writes[cmd.address] = (type(cmd).__name__, cmd.value)
                elif isinstance(cmd, SYNC):
                    commands.extend(list(writes.items()))
                    writes = {}
                    commands.extend(others)
                    others = []
                    # commands.append(
                    #     (cmd.address, (type(cmd).__name__, cmd.value)))
                else:
                    others.append(
                        (cmd.address, (type(cmd).__name__, cmd.value)))
            else:
                updates[cmd.address] = ('UPDATE', cmd.value)
        commands.extend(list(writes.items()))
        commands.extend(others)

        if len(commands) == 0:
            return -1

        # cmds = {'INIT': [], 'WRITE': [], 'TRIG': [], 'READ': []}
        cmds = {'WRITE': [], 'TRIG': [], 'READ': []}

        # for address, (cmd, value) in updates.items():
        #     cmds['INIT'].append((cmd, address, value, ''))

        for address, (cmd, value) in commands:
            cmds[cmd].append((cmd, address, value, ''))
        """
        cmds: dict
            in the form of {
                'INIT': [('UPDATE', address, value, ''), ...],
                'step1': [('WRITE', address, value, ''), ...],
                'step2': [('WAIT', '', delay, '')],
                'step3': [('TRIG', address, 0, ''), ...],
                ...
                'READ': [('READ', address, 0, ''), ...]
            }
        """

        if priority == 15:
            user = 'quafu'
        elif priority == 25:
            user = 'huawei'
        else:
            user = 'baqis'
        fut = self.call_api('feed',
                            task_id,
                            task_step,
                            cmds,
                            priority=priority,
                            extra=extra,
                            user=user,
                            name=name,
                            wait=0.01)
        succeed = fut.result()

        if isinstance(succeed, str) and succeed.startswith('Busy'):
            succeed = 0
        else:
            succeed = 1
        return int(succeed)

    def free(self, task_id: int) -> None:
        """release resources of task

        Args:
            task_id (int): uuid of task
        """
        self.call_api('free', task_id)

    def free_all(self) -> None:
        """release all resources
        """
        self.free(-1000)

    def submit(self, task_info: dict) -> None:
        self.call_api('submit', task_info)

    def fetch(self, task_id: int, skip: int = 0, extract='READ') -> list:
        """get results of task

        Args:
            task_id (int): uuid of task
            skip (int, optional): skip. Defaults to 0.

        Returns:
            list: list of results.
        """
        fut = self.call_api('fetch', task_id, skip)
        ret = fut.result()
        if isinstance(ret, str) and ret.startswith('No data found'):
            return []
        # result = [d[extract] for d in ret[:-1]]
        # if extract in ret[-1]:
        #     result.append(ret[-1][extract])
        return ret

    def save(self, task_id: int, path: str, index: dict) -> None:
        """save data to file
        """
        path = str(path).replace('.', '_')
        fut = self.call_api('save', task_id, path, index=index)
        return fut.result()

    def cancel(self, task_id: int) -> None:
        """cancel all tasks
        """
        self.call_api('cancel', task_id)


class FakeExecutor(Executor):

    def __init__(self, config={}, **kwds):
        from queue import Queue

        self._cfg = QuarkLocalConfig(config)
        self._queue = Queue()
        self._buff = {}

        self._state_caches = {}

        self._cancel_evt = threading.Event()
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        self.thread_pool = ThreadPoolExecutor()

    def run(self):
        from collections import defaultdict

        import numpy as np

        sampleRate = 1e9

        while True:
            if self._cancel_evt.is_set():
                while not self._queue.empty():
                    self._queue.get()
                self._cancel_evt.clear()
            task_id, task_step, cmds, extra, next_feed_time = self._queue.get()
            if task_step < 0:
                if task_id in self._buff:
                    self._buff[task_id]['state'] = 'finished'
                continue
            read_config = defaultdict(lambda: {
                'shots': 1024,
                'fnum': 1,
                'numberOfPoints': 1024
            })
            dataframe = {}

            for cmd in cmds:
                ch = '.'.join(cmd.address.split('.')[:-1])
                if isinstance(cmd, WRITE):
                    if cmd.address.endswith('.Coefficient'):
                        start, stop = cmd.value['start'], cmd.value['stop']
                        fnum = len(cmd.value['wList'])
                        read_config[ch]['start'] = start
                        read_config[ch]['stop'] = stop
                        read_config[ch]['fnum'] = fnum
                        numberOfPoints = int((stop - start) * sampleRate)
                        if numberOfPoints % 1024 != 0:
                            numberOfPoints = numberOfPoints + 1024 - numberOfPoints % 1024
                        read_config[ch]['numberOfPoints'] = numberOfPoints
                    elif cmd.address.endswith('.Shot'):
                        shots = cmd.value
                        read_config[ch]['shots'] = shots
                elif isinstance(cmd, READ):
                    if cmd.address.endswith('.IQ'):
                        shape = (read_config[ch]['shots'],
                                 read_config[ch]['fnum'])
                        dataframe['READ.' +
                                  cmd.address] = (np.random.randn(*shape),
                                                  np.random.randn(*shape))
                    elif cmd.address.endswith('.TraceIQ'):
                        shape = (read_config[ch]['shots'],
                                 read_config[ch]['numberOfPoints'])
                        dataframe['READ.' +
                                  cmd.address] = (np.random.randn(*shape),
                                                  np.random.randn(*shape))
                    else:
                        dataframe['READ.' +
                                  cmd.address] = np.random.randn(1000)
            data_map = extra['dataMap']
            dataframe = get_arch(data_map['arch']).assembly_data(
                dataframe, data_map)

            import time
            time.sleep(0.001)

            if task_id in self._buff:
                self._buff[task_id]['data'].append(dataframe)

    @property
    def cfg(self):
        return self._cfg

    def boot(self):
        pass

    def shutdown(self):
        pass

    def reset(self):
        pass

    def create_task(self, task_id: int, meta: dict = {}):
        self._buff[task_id] = {'state': 'init', 'data': []}

    def start(self, *args):
        pass

    def feed(self,
             task_id: int,
             task_step: int,
             cmds: list[COMMAND],
             extra: dict = {},
             priority: int = 0,
             name: str = '',
             next_feed_time=0):
        self._queue.put((task_id, task_step, cmds, extra, next_feed_time))
        return True

    def free(self, task_id: int) -> None:
        try:
            self._buff.pop(task_id)
        except:
            pass

    def free_all(self) -> None:
        self._buff.clear()

    def fetch(self, task_id: int, skip: int = 0) -> list:
        """get results of task

        Args:
            task_id (int): uuid of task
            skip (int, optional): skip. Defaults to 0.

        Returns:
            list: list of results.
        """
        if task_id not in self._buff:
            return []
        ret = self._buff[task_id]['data'][skip:]
        if len(ret) == 0 and self._buff[task_id]['state'] == 'finished':
            self._buff.pop(task_id)
        return ret

    def save(self, task_id: int, path: str, index: dict) -> None:
        pass

    def cancel(self) -> None:
        self._cancel_evt.set()
