import copy
import random
from dataclasses import dataclass, field

import numpy as np
from waveforms.dicttree import flattenDictIter
from waveforms.math.fit import (classify_data, count_state, count_to_diag,
                                install_classify_method)
from waveforms.waveform import square

from qlisp import (COMMAND, READ, SYNC, TRIG, WRITE, Architecture, CommandList,
                   DataMap, MeasurementTask, QLispCode, RawData, Result,
                   Signal)


def default_classify(data, params):
    """
    默认的分类方法
    """
    thr = params.get('threshold', 0)
    phi = params.get('phi', 0)
    return 1 + ((data * np.exp(-1j * phi)).real > thr)


install_classify_method("state", default_classify)


@dataclass
class ADTask():
    start: float = np.inf
    stop: float = 0
    trigger: str = ''
    triggerDelay: float = 0
    sampleRate: float = 1e9
    triggerClockCycle: float = 8e-9
    triggerLevel: float = 0
    triggerSlope: str = 'rising'
    triggerSource: str = 'external'
    triggerDelayAddress: str = ''
    triggerDration: float = 1e-6
    fList: list = field(default_factory=list)
    tasks: list = field(default_factory=list)
    wList: list = field(default_factory=list)
    wList_info: list = field(default_factory=list)
    coef_info: dict = field(default_factory=lambda: {
        'start': 0,
        'stop': 1024,
        'wList': []
    })


def _getADInfo(measures: dict[int, MeasurementTask]) -> dict[str, ADTask]:
    AD_tasks = {}
    for cbit in sorted(measures.keys()):
        task = measures[cbit]
        ad = task.hardware.IQ.name
        if ad not in AD_tasks:
            mapping = dict(task.hardware.IQ.commandAddresses)
            triggerDelayAddress = mapping.get('triggerDelayAddress', '')
            AD_tasks[ad] = ADTask(
                trigger=task.hardware.IQ.trigger,
                triggerDelay=task.hardware.IQ.triggerDelay,
                sampleRate=task.hardware.IQ.sampleRate,
                triggerClockCycle=task.hardware.IQ.triggerClockCycle,
                triggerDelayAddress=triggerDelayAddress)
        ad_task = AD_tasks[ad]
        ad_task.start = min(ad_task.start, task.time)
        #ad_task.start = np.floor_divide(ad_task.start,
        #                                task.hardware.IQ.triggerClockCycle
        #                                ) * task.hardware.IQ.triggerClockCycle
        ad_task.start = (round(ad_task.start * 1e15) //
                         round(task.hardware.IQ.triggerClockCycle *
                               1e15)) * task.hardware.IQ.triggerClockCycle
        ad_task.stop = max(ad_task.stop, task.time + task.params['duration'])
        ad_task.tasks.append(task)
    return AD_tasks


def _get_w_and_data_maps(AD_tasks: dict[str, ADTask]):
    dataMap = {'cbits': {}}

    for channel, ad_task in AD_tasks.items():
        ad_task.coef_info['start'] = ad_task.start
        ad_task.coef_info['stop'] = ad_task.stop

        for task in ad_task.tasks:
            Delta = task.params['frequency'] - task.hardware.lo_freq
            ad_task.fList.append(Delta)
            params = copy.copy(task.params)
            params['w'] = None
            dataMap['cbits'][task.cbit] = ('READ.' + channel,
                                           len(ad_task.fList) - 1, Delta,
                                           params, task.time, ad_task.start,
                                           ad_task.stop)

            ad_task.coef_info['wList'].append({
                'Delta':
                Delta,
                'phase':
                0,
                'weight':
                task.params.get('weight', 'one()'),
                'window':
                task.params.get('window', (0, 1024)),
                'w':
                task.params.get('w', None),
                't0':
                task.time,
                'phi':
                task.params.get('phi'),
                'threshold':
                task.params.get('threshold'),
            })
    return AD_tasks, dataMap


def assembly_code(code: QLispCode,
                  context=None) -> tuple[CommandList, DataMap]:
    cmds = []

    for key, wav in code.waveforms.items():
        cmds.append(WRITE(key, wav))

    ADInfo = _getADInfo(code.measures)
    ADInfo, dataMap = _get_w_and_data_maps(ADInfo)
    dataMap['signal'] = code.signal.value
    dataMap['arch'] = 'baqis'

    for channel, ad_task in ADInfo.items():
        delay = ad_task.start + ad_task.triggerDelay
        if ad_task.trigger:
            cmds.append(
                WRITE(
                    ad_task.trigger,
                    square(ad_task.triggerDration) >>
                    ad_task.triggerDration / 2))
        cmds.append(WRITE(channel + '.Shot', code.shots))
        cmds.append(WRITE(channel + '.Coefficient', ad_task.coef_info))
        # cmds.append(WRITE(channel + '.Classify', code.signal.value))
        if ad_task.triggerDelayAddress == "":
            cmds.append(WRITE(channel + '.TriggerDelay', delay))
        else:
            cmds.append(
                WRITE(ad_task.triggerDelayAddress + '.TriggerDelay', delay))

    mode_pointer = capture_pointer = len(cmds)

    for channel in ADInfo:
        if code.signal & Signal.trace:
            cmds.append(READ(channel + '.TraceIQ'))
            cmds.insert(mode_pointer, WRITE(channel + '.CaptureMode', 'raw'))
        else:
            cmds.append(READ(channel + '.IQ'))
            cmds.insert(mode_pointer, WRITE(channel + '.CaptureMode', 'alg'))
        mode_pointer += 1
        capture_pointer += 1
        cmds.insert(
            capture_pointer,
            WRITE(channel + '.StartCapture', random.randint(0, 2**16 - 1)))
        capture_pointer += 1

    if (code.signal & Signal.remote_count) == Signal.remote_count:
        cmds.append(READ(channel + '.Counts'))

    return cmds, dataMap


def _sort_cbits(raw_data, dataMap):
    ret = []
    gate_list = []
    # min_shots = np.inf
    for cbit in sorted(dataMap):
        ch, i, Delta, params, time, start, stop = dataMap[cbit]
        gate_list.append({'params': params})
        try:
            key = f'{ch}.IQ'
            if isinstance(raw_data[key], np.ndarray):
                ret.append(raw_data[key][..., i])
            else:
                raise ValueError('error on ad', str(raw_data[key]))
        except KeyError:
            key = f'{ch}.TraceIQ'
            ret.append(raw_data[key])
        # min_shots = min(min_shots, ret[-1].shape[0])

    # ret = [r[:min_shots] for r in ret]

    return np.asfortranarray(ret).T, gate_list


def _sort_data(raw_data, dataMap):
    ret = {}
    for label, channel in dataMap.items():
        if label in raw_data:
            ret[label] = raw_data['READ.' + channel]
    return ret


def _process_classify(data, gate_params_list, signal, classify):
    result = {}

    if signal & Signal._remote:
        if (signal & Signal.remote_trace_avg) == Signal.remote_trace_avg:
            result['remote_trace_avg'] = data
        elif (signal & Signal.remote_iq_avg) == Signal.remote_iq_avg:
            result['remote_iq_avg'] = data
        elif (signal & Signal.remote_count) == Signal.remote_count:
            pass
        elif (signal & Signal.remote_population) == Signal.remote_population:
            result['remote_population'] = data
        else:
            result['remote_state'] = data
        return result

    if signal & Signal.trace:
        if signal & Signal._avg_trace:
            result['trace_avg'] = data.mean(axis=-2)
        else:
            result['trace'] = data
        return result

    if signal & Signal.iq:
        if signal & Signal._avg_iq:
            result['iq_avg'] = data.mean(axis=-2)
        else:
            result['iq'] = data

    if signal & Signal.state:
        state = classify(data, gate_params_list, avg=False)
    if signal & Signal._count:
        count = count_state(state)

    if (signal & Signal.diag) == Signal.diag:
        result['diag'] = count_to_diag(count)
    elif (signal & Signal.count) == Signal.count:
        result['count'] = count
    elif (signal & Signal.population) == Signal.population:
        populations = np.array(
            [np.count_nonzero(state == 2**i, axis=-2) for i in range(8)])
        populations = populations / np.sum(populations, axis=0)
        result['population'] = []
        for gate_params, p in zip(gate_params_list, populations[1]):
            Pg, Pe = gate_params['params'].get('PgPe', [0, 1])
            p1 = (p - Pg) / (Pe - Pg)
            result['population'].append(p1)
        result['population'] = np.asarray(result['population'])
        for i in range(4):
            result[f'P{i}'] = populations[i]
        for i, gate_params in enumerate(gate_params_list):
            M = gate_params['params'].get('M', np.eye(4))
            q = np.linalg.inv(np.asarray(M)) @ populations[:M.shape[0], i]
            for j, v in enumerate(q):
                if f'Q{j}' in result:
                    result[f'Q{j}'].append(v)
                else:
                    result[f'Q{j}'] = [v]
    elif signal & Signal.state:
        result['state'] = state

    return result


def _get_classify_func(fun_name):
    dispatcher = {}
    try:
        return dispatcher[fun_name]
    except:
        return classify_data


def _sort_remote_count_data(raw_data, dataMap):

    def _resort(key, _data_mapping):
        return tuple([key[i] for i in _data_mapping])

    raw_count, sort_keys = None, []
    for cbit in sorted(dataMap):
        ch, i, Delta, params, time, start, stop = dataMap[cbit]
        sort_keys.append((ch, i, cbit))
        if ch + '.Counts' in raw_data.keys():
            raw_count = dict(raw_data[ch + '.Counts'])
    sort_rank = {k[-1]: i for i, k in enumerate(sorted(sort_keys))}
    sort_list = [sort_rank[cbit] for cbit in sorted(sort_rank)]

    ret = {}
    for item in raw_count.keys():
        ret[_resort(item, sort_list)] = raw_count[item]
    return ret


def assembly_data(raw_data: RawData, dataMap: DataMap) -> Result:
    if not dataMap:
        return raw_data

    result = {}

    def decode(value):
        if (isinstance(value, tuple) and len(value) == 2
                and isinstance(value[0], np.ndarray)
                and isinstance(value[1], np.ndarray)
                and value[0].shape == value[1].shape):
            return value[0] + 1j * value[1]
        else:
            return value

    raw_data = {k: decode(v) for k, v in flattenDictIter(raw_data)}
    if 'cbits' in dataMap:
        data, gate_params_list = _sort_cbits(raw_data, dataMap['cbits'])
        classify = _get_classify_func(gate_params_list[0].get(
            'classify', None))
        result.update(
            _process_classify(data, gate_params_list,
                              Signal(dataMap['signal']), classify))
        if (dataMap['signal']
                & Signal.remote_count.value) == Signal.remote_count.value:
            result.update({
                'remote_count':
                _sort_remote_count_data(raw_data, dataMap['cbits'])
            })
    if 'data' in dataMap:
        result.update(_sort_data(raw_data, dataMap['data']))
    return result


baqisArchitecture = Architecture('baqis', "", assembly_code, assembly_data)
