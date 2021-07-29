import logging
import random
import re
import threading
import time
from typing import Any, NamedTuple

import numpy as np
from waveforms.baseconfig import _flattenDictIter
from waveforms.math import getFTMatrix
from waveforms.math.fit import classifyData, count_to_diag, countState
from waveforms.sched.scheduler import Executor
from waveforms.sched.task import COMMAND, READ, WRITE

from quark import connect

from .quarkconfig import QuarkConfig
from .quarkcontext import QuarkContext

log = logging.getLogger(__name__)


def set_up_backend(host='127.0.0.1'):
    from waveforms.quantum.circuit.qlisp.config import set_config_factory
    from waveforms.quantum.circuit.qlisp.qlisp import set_context_factory

    set_config_factory(lambda: QuarkConfig(host=host))
    set_context_factory(QuarkContext)


def getCommands(code, signal='state', shots=1024):
    cmds = []
    for key, wav in code.waveforms.items():
        cmds.append(WRITE(key, wav))

    ADInfo = {}
    dataMap = {}
    readChannels = set()
    for cbit in sorted(code.measures.keys()):
        task = code.measures[cbit][-1]
        readChannels.add(task.hardware['channel']['IQ'])
        ad = task.hardware['channel']['IQ']
        if ad not in ADInfo:
            ADInfo[ad] = {
                'fList': [],
                'sampleRate': task.hardware['params']['sampleRate']['IQ'],
                'tasks': [],
                'w': []
            }
        Delta = task.params['frequency'] - task.hardware['params'][
            'LOFrequency']

        if task.params['w'] is not None:
            w = task.params['w']
        else:
            w = getFTMatrix([Delta],
                            4096,
                            weight=task.params['weight'],
                            sampleRate=ADInfo[ad]['sampleRate'])[:, 0]

        ADInfo[ad]['w'].append(w)
        ADInfo[ad]['fList'].append(Delta)
        ADInfo[ad]['tasks'].append(task)
        dataMap[cbit] = (ad, len(ADInfo[ad]['fList']) - 1, Delta, task.params)

    for channel, info in ADInfo.items():
        coefficient = np.asarray(info['w'])
        cmds.append(WRITE(channel + '.coefficient', coefficient))
        cmds.append(WRITE(channel + '.pointNum', coefficient.shape[-1]))
        cmds.append(WRITE(channel + '.shots', shots))

    for channel in readChannels:
        if signal == 'trace':
            cmds.append(READ(channel + '.TraceIQ'))
            cmds.append(WRITE(channel + '.CaptureMode', 'raw'))
        else:
            cmds.append(READ(channel + '.IQ'))
            cmds.append(WRITE(channel + '.CaptureMode', 'alg'))

    for channel in readChannels:
        cmds.append(
            WRITE(channel + '.StartCapture', random.randint(0, 2**16 - 1)))

    return cmds, dataMap


def assymblyData(raw_data, dataMap, signal='state', classify=classifyData):
    ret = []
    gate_list = []
    result = {k: v[0] + 1j * v[1] for k, v in _flattenDictIter(raw_data)}

    min_shots = np.inf
    for cbit in sorted(dataMap):
        ch, i, Delta, params = dataMap[cbit]
        gate_list.append({'params': params})
        try:
            key = f'{ch}.IQ'
            ret.append(result[key][:, i])
        except KeyError:
            key = f'{ch}.TraceIQ'
            ret.append(result[key])
        min_shots = min(min_shots, ret[-1].shape[0])

    ret = [r[:min_shots] for r in ret]

    result = {}
    result['data'] = np.asarray(ret).T

    if signal in ['state', 'count', 'diag']:
        result['state'] = classify(result['data'], gate_list, avg=False)
    if signal in ['count', 'diag']:
        result['count'] = countState(result['state'])
    if signal == 'diag':
        result['diag'] = count_to_diag(result['count'])

    return result


def _is_feedable(key):
    return re.match(r'[QCM]\d+\.(setting|waveform)\..+', key)


class _connection_pool(NamedTuple):
    actived: dict
    disactived: list
    lock: threading.Lock
    max_unused: int = 10


class QuarkExcutor(Executor):
    def __init__(self, host='127.0.0.1'):

        self.host = host
        set_up_backend(self.host)

        self._conn_pool = _connection_pool({}, [], threading.Lock())
        self._gc_thread = threading.Thread(target=self._gc, daemon=True)
        self._gc_thread.start()

    def _gc(self):
        """clear unused connections.
        """
        while True:
            time.sleep(60)
            with self._conn_pool.lock:
                alived = [t.ident for t in threading.enumerate()]
                for tid in list(self._conn_pool.actived.keys()):
                    if tid not in alived:
                        conn = self._conn_pool.actived.pop(tid)
                        if len(self._conn_pool.disactived
                               ) < self._conn_pool.max_unused:
                            self._conn_pool.disactived.append(conn)

    @property
    def conn(self):
        """rpc connection to quarkstudio
        """
        with self._conn_pool.lock:
            tid = threading.get_ident()
            if tid not in self._conn_pool.actived:
                try:
                    self._conn_pool.actived[
                        tid] = self._conn_pool.disactived.pop()
                except IndexError:
                    self._conn_pool.actived[tid] = connect('QuarkServer',
                                                           host=self.host)
            return self._conn_pool.actived[tid]

    def start(self, *args):
        self.conn.start(*args)
        self.log.debug(f'start({args})')

    def feed(self,
             task_id: int,
             task_step: int,
             cmds: list[COMMAND],
             extra: dict = {}):
        """feed api of quark

        Args:
            task_id (int): uuid of task
            task_step (int): step of task (start from 0)
            cmds (list): commands to be executed
            extra (dict): extra data
        """

        keys = []
        values = []

        for cmd in cmds:
            if _is_feedable(cmd.key):
                keys.append(cmd.key)
                values.append((type(cmd).__name__, cmd.value))

        self.conn.feed(task_id, task_step, keys, values, extra=extra)
        self.log.debug(f'feed({task_id}, {task_step}, {keys}, {values})')

    def free(self, task_id: int) -> None:
        """release resources of task

        Args:
            task_id (int): uuid of task
        """
        self.conn.free(task_id)
        self.log.debug(f'free({task_id})')

    def free_all(self) -> None:
        """release all resources
        """
        self.free(-1000)
        self.log.debug('free(-1000)')

    def submit(self, task_id: int, data_template: dict) -> None:
        self.conn.submit(task_id, data_template)
        self.log.debug(f'submit({task_id}, {data_template})')

    def fetch(self, task_id: int, skip: int = 0) -> list:
        """get results of task

        Args:
            task_id (int): uuid of task
            skip (int, optional): skip. Defaults to 0.

        Returns:
            list: list of results.
        """
        ret = self.conn.fetch(task_id, skip)
        self.log.debug(f'fetch({task_id}, {skip})')
        if ret is None:
            return []
        return ret

    def update(self, key: str, value: Any) -> None:
        """update key to value

        Args:
            key (str): key to update
            value (Any): value to update
        """
        self.conn.update(key, value)
        self.log.debug(f'update({key}, {value})')

    def save(self, path: str, task_id: int, data: dict) -> None:
        """save data to file
        """
        ret = self.conn.save(path, task_id, data)
        self.log.debug(f'save({path}, {task_id}, {data})')
        return ret

    def query(self, key: str) -> Any:
        """query key
        """
        ret = self.conn.query(key)
        self.log.debug(f'query({key})')
        return ret

    def cancel(self) -> None:
        """cancel all tasks
        """
        self.conn.cancel()
        self.log.debug('cancel()')
