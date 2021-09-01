import logging
import re
import threading
import time
import warnings
from typing import Any, NamedTuple

from waveforms.namespace import _NOTSET
from waveforms.quantum.circuit.qlisp.arch import get_arch
from waveforms.sched.base import COMMAND, READ, SYNC, TRIG, WRITE, Executor
from waveforms.waveform_parser import wave_eval

from quark import connect

from .quarkconfig import QuarkConfig

log = logging.getLogger(__name__)


def set_up_backend(host='127.0.0.1'):
    from waveforms.quantum.circuit.qlisp.config import set_config_factory

    set_config_factory(lambda: QuarkConfig(host=host))


def assymblyData(raw_data, dataMap, *args, **kwargs):
    warnings.warn("assymblyData is deprecated, use getCommands instead",
                  DeprecationWarning, 2)
    if args:
        warnings.warn(f'Unused arguments: {args}', DeprecationWarning, 2)
    if kwargs:
        warnings.warn(f'Unused arguments: {kwargs}', DeprecationWarning, 2)
    return get_arch().assembly_data(raw_data, dataMap)


def getCommands(code, *args, **kwargs):
    warnings.warn("getCommands is deprecated, use get_arch instead",
                  DeprecationWarning, 2)
    if args:
        warnings.warn(f'Unused arguments: {args}', DeprecationWarning, 2)
    if kwargs:
        warnings.warn(f'Unused arguments: {kwargs}', DeprecationWarning, 2)
    return get_arch().assembly_code(code)


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
        # ret = ret['READ']
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
        warnings.warn('update() is deprecated, use cfg.update() instead',
                      DeprecationWarning, 2)
        self.set(key, value, cache)
        self.cfg.update(key, value, cache=cache)
        self.log.debug(f'update({key}, {value})')

    def update_all(self, data: list[tuple[str, Any]]) -> None:
        """update all keys to values
        """
        warnings.warn(
            'update_all() is deprecated, use cfg.update_all() instead',
            DeprecationWarning, 2)
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
        warnings.warn('query() is deprecated, use cfg.query() instead',
                      DeprecationWarning, 2)
        ret = self.cfg.query(key)
        self.log.debug(f'query({key})')
        return ret

    def cancel(self) -> None:
        """cancel all tasks
        """
        self.conn.cancel()
        self.log.debug('cancel()')


class FakeExecutor(Executor):
    def __init__(self, config={}, **kwds):
        from waveforms.backends.quark.quarkconfig import QuarkLocalConfig
        from waveforms.quantum.circuit.qlisp.config import set_config_factory
        from waveforms.quantum.circuit.qlisp.qlisp import set_context_factory

        self._cfg = QuarkLocalConfig(config)
        set_config_factory(lambda: self._cfg)

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
        return [d['READ'] for d in ret.values()]

    def save(self, task_id: int, path: str) -> None:
        pass

    def cancel(self) -> None:
        pass
