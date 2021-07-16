import numpy as np
from waveforms.math import getFTMatrix


def getCommands(code, signal='state'):
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
        dataMap[cbit] = (ad, len(ADInfo[ad]['fList']) - 1, Delta, task)

    for i, (channel, info) in enumerate(ADInfo.items()):
        coefficient = np.asarray(info['w'])
        cmds.append((channel + '.coefficient', coefficient))
        cmds.append((channel + '.pointNum', coefficient.shape[-1]))

    for channel in readChannels:
        if signal == 'trace':
            cmds.append((channel + '.TraceIQ', None))
        else:
            cmds.append((channel + '.IQ', None))

    return cmds, dataMap
