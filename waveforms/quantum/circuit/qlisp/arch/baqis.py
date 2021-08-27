import random

import numpy as np
from waveforms.baseconfig import _flattenDictIter
from waveforms.math import getFTMatrix
from waveforms.math.fit import classifyData, count_to_diag, countState
from waveforms.math.signal import shift
from waveforms.waveform_parser import wave_eval

from .base import COMMAND, READ, SYNC, TRIG, WRITE, Architecture


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


def getCommands(code):
    cmds = []

    for key, wav in code.waveforms.items():
        wav = wav << code.end
        wav = wav >> 99e-6
        cmds.append(WRITE(key, wav))

    ADInfo = _getADInfo(code.measures)
    ADInfo, dataMap = _get_w_and_data_maps(ADInfo)

    for channel, info in ADInfo.items():
        coefficient = np.asarray(info['w'])
        delay = 0 * info['start'] + info['triggerDelay']
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


baqisArchitecture = Architecture('baqis', "", getCommands, assymblyData)
