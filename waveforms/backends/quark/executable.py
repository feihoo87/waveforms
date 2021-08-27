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
from waveforms.math.signal import shift
from waveforms.namespace import _NOTSET
from waveforms.sched.base import COMMAND, READ, SYNC, TRIG, WRITE, Executor
from waveforms.waveform_parser import wave_eval

from quark import connect

from .quarkconfig import QuarkConfig
from .quarkcontext import QuarkContext

log = logging.getLogger(__name__)


def set_up_backend(host='127.0.0.1'):
    from waveforms.quantum.circuit.qlisp.config import set_config_factory
    from waveforms.quantum.circuit.qlisp.qlisp import set_context_factory

    set_config_factory(lambda: QuarkConfig(host=host))
    set_context_factory(QuarkContext)


def _getADInfo(measures):
    ADInfo = {}
    for cbit in sorted(measures.keys()):
        task = measures[cbit][-1]
        ad = task.hardware['channel']['IQ']
        if ad not in ADInfo:
            ADInfo[ad] = {
                'start': np.inf,
                'stop': 0,
                'triggerDelay': task.hardware['params'].get('triggerDelay', 0),
                'fList': [],
                'sampleRate': task.hardware['params']['sampleRate']['IQ'],
                'tasks': [],
                'w': []
            }
        ADInfo[ad]['start'] = min(ADInfo[ad]['start'], task.time)
        #ADInfo[ad]['start'] = np.floor_divide(ADInfo[ad]['start'], 8e-9) * 8e-9
        ADInfo[ad]['stop'] = max(ADInfo[ad]['stop'],
                                 task.time + task.params['duration'])
        ADInfo[ad]['tasks'].append(task)
    return ADInfo


def _get_w_and_data_maps(ADInfo):
    dataMap = {'cbits': {}}

    for channel, info in ADInfo.items():
        numberOfPoints = int(
            (info['stop'] - info['start']) * info['sampleRate'])
        if numberOfPoints % 1024 != 0:
            numberOfPoints = numberOfPoints + 1024 - numberOfPoints % 1024
        t = np.arange(numberOfPoints) / info['sampleRate'] + info['start']

        for task in info['tasks']:
            Delta = task.params['frequency'] - task.hardware['params'][
                'LOFrequency']
            info['fList'].append(Delta)
            dataMap['cbits'][task.cbit] = (channel, len(info['fList']) - 1,
                                           Delta, task.params, task.time,
                                           info['start'], info['stop'])
            if task.params['w'] is not None:
                w = np.zeros(numberOfPoints, dtype=complex)
                w[:len(task.params['w'])] = task.params['w']
                w = shift(w, task.time - info['start'])
            else:
                fun = wave_eval(task.params['weight']) >> task.time
                weight = fun(t)
                w = getFTMatrix([Delta],
                                numberOfPoints,
                                weight=weight,
                                sampleRate=info['sampleRate'])[:, 0]
            info['w'].append(w)
    return ADInfo, dataMap


def getCommands(code, signal='state', shots=1024):
    cmds = []

    for key, wav in code.waveforms.items():
        wav = wav << code.end
        wav = wav >> 99e-6
        cmds.append(WRITE(key, wav))

    ADInfo = _getADInfo(code.measures)
    ADInfo, dataMap = _get_w_and_data_maps(ADInfo)

    for channel, info in ADInfo.items():
        coefficient = np.asarray(info['w'])
        delay = 0*info['start'] + info['triggerDelay']
        cmds.append(WRITE(channel + '.coefficient', coefficient))
        cmds.append(WRITE(channel + '.pointNum', coefficient.shape[-1]))
        cmds.append(WRITE(channel + '.triggerDelay', delay))
        cmds.append(WRITE(channel + '.shots', shots))

    for channel in ADInfo:
        if signal == 'trace':
            cmds.append(READ(channel + '.TraceIQ'))
            cmds.append(WRITE(channel + '.CaptureMode', 'raw'))
        else:
            cmds.append(READ(channel + '.IQ'))
            cmds.append(WRITE(channel + '.CaptureMode', 'alg'))

    for channel in ADInfo:
        cmds.append(
            WRITE(channel + '.StartCapture', random.randint(0, 2**16 - 1)))

    return cmds, dataMap


def _sort_cbits(raw_data, dataMap):
    ret = []
    gate_list = []
    min_shots = np.inf
    for cbit in sorted(dataMap):
        ch, i, Delta, params, time, start, stop = dataMap[cbit]
        gate_list.append({'params': params})
        try:
            key = f'{ch}.IQ'
            ret.append(raw_data[key][:, i])
        except KeyError:
            key = f'{ch}.TraceIQ'
            ret.append(raw_data[key])
        min_shots = min(min_shots, ret[-1].shape[0])

    ret = [r[:min_shots] for r in ret]

    return np.asfortranarray(ret).T, gate_list


def _sort_data(raw_data, dataMap):
    ret = {}
    for label, channel in dataMap.items():
        if label in raw_data:
            ret[label] = raw_data[channel]
    return ret


def _process_classify(data, gate_params_list, signal, classify):
    result = {}
    if signal in ['state', 'count', 'diag']:
        result['state'] = classify(data, gate_params_list, avg=False)
    if signal in ['count', 'diag']:
        result['count'] = countState(result['state'])
    if signal == 'diag':
        result['diag'] = count_to_diag(result['count'])
    return result


def _get_classify_func(fun_name):
    dispatcher = {}
    try:
        return dispatcher[fun_name]
    except:
        return classifyData


def assymblyData(raw_data, dataMap, signal='state', classify=classifyData):
    if not dataMap:
        return raw_data

    result = {}
    raw_data = {k: v[0] + 1j * v[1] for k, v in _flattenDictIter(raw_data)}
    if 'cbits' in dataMap:
        data, gate_params_list = _sort_cbits(raw_data, dataMap['cbits'])
        classify = _get_classify_func(gate_params_list.get('classify', None))
        result.update(
            _process_classify(data, gate_params_list, signal, classify))
        result['data'] = data
    if 'data' in dataMap:
        result.update(_sort_data(raw_data, dataMap['data']))
    return result


def _is_feedable(cmd):
    if isinstance(cmd, WRITE):
        if re.match(r'[QCM]\d+\..+', cmd.address) and not re.match(
                r'[QCM]\d+\.(setting|waveform)\..+', cmd.address):
            return False
        if cmd.address.startswith('gate.'):
            return False
    return True


class _connection_pool(NamedTuple):
    actived: dict
    disactived: list
    lock: threading.Lock
    max_unused: int = 10


class QuarkExecutor(Executor):
    def __init__(self, host='127.0.0.1'):
        from waveforms import getConfig

        self.host = host
        set_up_backend(self.host)
        self._cfg = getConfig()

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
                        # if len(self._conn_pool.disactived
                        #        ) < self._conn_pool.max_unused:
                        #     self._conn_pool.disactived.append(conn)

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
                                                           host=self.host,
                                                           verbose=False)
            return self._conn_pool.actived[tid]

    def get_config(self):
        return self._cfg

    @property
    def cfg(self):
        return self._cfg

    def boot(self):
        pass

    def shutdown(self):
        pass

    def reset(self):
        pass

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
        commands = []

        writes = {}
        others = []
        for cmd in cmds:
            if _is_feedable(cmd):
                if isinstance(cmd, WRITE):
                    if not (isinstance(cmd.value, tuple)
                            and isinstance(cmd.value[0], _NOTSET)):
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
        commands.extend(list(writes.items()))
        commands.extend(others)

        if len(commands) == 0:
            return False

        priority = 0
        self.conn.feed(priority, task_id, task_step, commands, extra=extra)
        self.log.debug(
            f'feed({priority}, {task_id}, {task_step}, {commands}, extra={extra})'
        )
        return True

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

    def submit(self, task_info: dict) -> None:
        self.conn.submit(task_info)
        self.log.debug(f'submit({task_info})')

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

    def update(self, key: str, value: Any, cache: bool = False) -> None:
        """update key to value

        Args:
            key (str): key to update
            value (Any): value to update
        """
        self.cfg.update(key, value, cache=cache)
        self.log.debug(f'update({key}, {value})')

    def update_all(self, data: list[tuple[str, Any]]) -> None:
        """update all keys to values
        """
        self.cfg.update_all(data)
        self.log.debug(f'update_all({data})')

    def save(self, task_id: int, path: str) -> None:
        """save data to file
        """
        ret = self.conn.save(task_id, path)
        self.log.debug(f'save({task_id}, {path})')
        return ret

    def query(self, key: str) -> Any:
        """query key
        """
        ret = self.cfg.query(key)
        self.log.debug(f'query({key})')
        return ret

    def cancel(self) -> None:
        """cancel all tasks
        """
        self.conn.cancel()
        self.log.debug('cancel()')


class FakeExecutor(Executor):
    def __init__(self, **kwds):
        pass

    def get_config(self):
        return {}

    @property
    def cfg(self):
        return {}

    def boot(self):
        pass

    def shutdown(self):
        pass

    def reset(self):
        pass

    def start(self, *args):
        pass

    def feed(self,
             task_id: int,
             task_step: int,
             cmds: list[COMMAND],
             extra: dict = {}):
        return True

    def free(self, task_id: int) -> None:
        pass

    def free_all(self) -> None:
        pass

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

    def update(self, key: str, value: Any, cache: bool = False) -> None:
        pass

    def save(self, task_id: int, path: str) -> None:
        pass

    def query(self, key: str) -> Any:
        pass

    def cancel(self) -> None:
        pass