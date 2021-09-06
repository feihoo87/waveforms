import copy
import random
from dataclasses import dataclass, field

import numpy as np
from waveforms.baseconfig import _flattenDictIter
from waveforms.math import getFTMatrix
from waveforms.math.fit import classifyData, count_to_diag, countState
from waveforms.math.signal import shift
from waveforms.waveform import square
from waveforms.waveform_parser import wave_eval

from .base import (COMMAND, READ, SYNC, TRIG, WRITE, Architecture, CommandList,
                   DataMap, MeasurementTask, QLispCode, RawData, Result)


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
    triggerDration: float = 1e-6
    fList: list = field(default_factory=list)
    tasks: list = field(default_factory=list)
    wList: list = field(default_factory=list)


def _getADInfo(
        measures: dict[int, list[MeasurementTask]]) -> dict[str, ADTask]:
    AD_tasks = {}
    for cbit in sorted(measures.keys()):
        task = measures[cbit][-1]
        ad = task.hardware.IQ.name
        if ad not in AD_tasks:
            AD_tasks[ad] = ADTask(
                trigger=task.hardware.IQ.trigger,
                triggerDelay=task.hardware.IQ.triggerDelay,
                sampleRate=task.hardware.IQ.sampleRate,
                triggerClockCycle=task.hardware.IQ.triggerClockCycle)
        ad_task = AD_tasks[ad]
        ad_task.start = min(ad_task.start, task.time)
        ad_task.start = np.floor_divide(ad_task.start,
                                        task.hardware.IQ.triggerClockCycle
                                        ) * task.hardware.IQ.triggerClockCycle
        ad_task.stop = max(ad_task.stop, task.time + task.params['duration'])
        ad_task.tasks.append(task)
    return AD_tasks


def _get_w_and_data_maps(AD_tasks: dict[str, ADTask]):
    dataMap = {'cbits': {}}

    for channel, ad_task in AD_tasks.items():
        numberOfPoints = int(
            (ad_task.stop - ad_task.start) * ad_task.sampleRate)
        if numberOfPoints % 1024 != 0:
            numberOfPoints = numberOfPoints + 1024 - numberOfPoints % 1024
        t = np.arange(numberOfPoints) / ad_task.sampleRate + ad_task.start

        for task in ad_task.tasks:
            Delta = task.params['frequency'] - task.hardware.lo_freq
            ad_task.fList.append(Delta)
            params = copy.copy(task.params)
            params['w'] = None
            dataMap['cbits'][task.cbit] = (channel, len(ad_task.fList) - 1,
                                           Delta, params, task.time,
                                           ad_task.start, ad_task.stop)
            if task.params['w'] is not None:
                w = np.zeros(numberOfPoints, dtype=complex)
                w[:len(task.params['w'])] = task.params['w']
                w = shift(w, task.time - ad_task.start)
            else:
                fun = wave_eval(task.params['weight']) >> task.time
                weight = fun(t)
                phase = 2 * np.pi * Delta * ad_task.start
                w = getFTMatrix([Delta],
                                numberOfPoints,
                                phaseList=[phase],
                                weight=weight,
                                sampleRate=ad_task.sampleRate)[:, 0]
            ad_task.wList.append(w)
    return AD_tasks, dataMap


def assembly_code(code: QLispCode) -> tuple[CommandList, DataMap]:
    cmds = []

    for key, wav in code.waveforms.items():
        cmds.append(WRITE(key, wav))

    ADInfo = _getADInfo(code.measures)
    ADInfo, dataMap = _get_w_and_data_maps(ADInfo)
    dataMap['signal'] = code.signal
    dataMap['arch'] = 'baqis'

    for channel, ad_task in ADInfo.items():
        coefficient = np.asarray(ad_task.wList)
        delay = ad_task.start + ad_task.triggerDelay
        if ad_task.trigger:
            cmds.append(
                WRITE(
                    ad_task.trigger,
                    square(ad_task.triggerDration) >>
                    ad_task.triggerDration / 2))
        cmds.append(WRITE(channel + '.coefficient', coefficient))
        cmds.append(WRITE(channel + '.pointNum', coefficient.shape[-1]))
        cmds.append(WRITE(channel + '.triggerDelay', delay))
        cmds.append(WRITE(channel + '.shots', code.shots))

    for channel in ADInfo:
        if code.signal == 'trace':
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


def assembly_data(raw_data: RawData, dataMap: DataMap) -> Result:
    if not dataMap:
        return raw_data

    result = {}
    raw_data = {k: v[0] + 1j * v[1] for k, v in _flattenDictIter(raw_data)}
    if 'cbits' in dataMap:
        data, gate_params_list = _sort_cbits(raw_data, dataMap['cbits'])
        classify = _get_classify_func(gate_params_list[0].get(
            'classify', None))
        result.update(
            _process_classify(data, gate_params_list, dataMap['signal'],
                              classify))
        result['data'] = data
    if 'data' in dataMap:
        result.update(_sort_data(raw_data, dataMap['data']))
    return result


baqisArchitecture = Architecture('baqis', "", assembly_code, assembly_data)
