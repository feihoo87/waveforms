import numpy as np
from waveforms.math import getFTMatrix
from waveforms.math.fit import classifyData, count_to_diag, countState


def getCommands(code, signal='state'):
    GETVALUECMD = 'EMPTY'

    cmds = list(code.waveforms.items())

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
        cmds.append((channel + '.coefficient', coefficient))
        cmds.append((channel + '.pointNum', coefficient.shape[-1]))

    for channel in readChannels:
        if signal == 'trace':
            cmds.append((channel + '.TraceIQ', GETVALUECMD))
        else:
            cmds.append((channel + '.IQ', GETVALUECMD))

    return cmds, dataMap


def assymblyData(raw_data, dataMap, signal='state', classify=classifyData):
    def flattenDictIter(d, prefix=[]):
        for k in d:
            if isinstance(d[k], dict):
                yield from flattenDictIter(d[k], prefix=[*prefix, k])
            else:
                yield '.'.join(prefix + [k]), d[k]

    ret = []
    gate_list = []
    result = {k: v[0] + 1j * v[1] for k, v in flattenDictIter(raw_data)}

    for cbit in sorted(dataMap):
        ch, i, Delta, params = dataMap[cbit]
        gate_list.append({'params': params})
        try:
            key = f'{ch}.IQ'
            ret.append(result[key][:, i])
        except KeyError:
            key = f'{ch}.TraceIQ'
            ret.append(result[key][:, i])

    result = {}
    result['data'] = np.asarray(ret).T

    if signal in ['state', 'count', 'diag']:
        result['state'] = classify(result['data'], gate_list, avg=False)
    if signal in ['count', 'diag']:
        result['count'] = countState(result['state'])
    if signal == 'diag':
        result['diag'] = count_to_diag(result['count'])

    return result
