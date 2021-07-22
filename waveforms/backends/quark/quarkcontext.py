from typing import Union

from waveforms.quantum.circuit.qlisp.qlisp import Context


def _getSharedCoupler(qubitsDict: dict) -> set:
    s = set(qubitsDict[0]['couplers'])
    for qubit in qubitsDict[1:]:
        s = s & set(qubit['couplers'])
    return s


def _makeAWGChannelInfo(section: str, cfgDict: dict,
                        name: str) -> Union[str, dict]:
    ret = {}
    if name == 'RF':
        # if cfgDict['channel']['DDS'] is not None:
        #     return f"{section}.waveform.DDS"
        if cfgDict['channel']['I'] is not None:
            ret['I'] = f"{section}.waveform.RF.I"
        if cfgDict['channel']['Q'] is not None:
            ret['Q'] = f"{section}.waveform.RF.Q"
        ret['LO'] = cfgDict['setting']['LO']
        return ret
    elif name == 'AD.trigger':
        return f"{section}.waveform.TRIG"
    else:
        return f"{section}.waveform.{name}"


class QuarkContext(Context):
    def _getAWGChannel(self, name, *qubits) -> Union[str, dict]:

        qubitsDict = [self.cfg.getQubit(q) for q in qubits]

        if name.startswith('readoutLine.'):
            #name = name.removeprefix('readoutLine.')
            name = name[len('readoutLine.'):]
            section = qubitsDict[0]['probe']
            cfgDict = self.cfg.getReadout(section)
        elif name.startswith('coupler.'):
            #name = name.removeprefix('coupler.')
            name = name[len('coupler.'):]
            section = _getSharedCoupler(qubitsDict).pop()
            cfgDict = self.cfg.getCoupler(section)
        else:
            section = qubits[0]
            cfgDict = qubitsDict[0]

        chInfo = _makeAWGChannelInfo(section, cfgDict, name)

        return chInfo

    def _getReadoutADLO(self, qubit):
        rl = self.cfg.getReadout(self.cfg.getQubit(qubit)['probe'])
        lo = rl['setting']['LO']
        return lo

    def _getADChannel(self, qubit) -> Union[str, dict]:
        rl = self.cfg.getQubit(qubit)['probe']
        rlDict = self.cfg.getReadout(rl)
        chInfo = {
            'IQ': rlDict['channel']['ADC'],
            'LO': rlDict['setting']['LO'],
            'trigger': f'{rl}.waveform.TRIG',
            'sampleRate': rlDict['adcsr']
        }
        return chInfo

    def _getLOFrequencyOfChannel(self, chInfo) -> float:
        return chInfo['LO']

    def _getADChannelDetails(self, chInfo) -> dict:
        hardware = {'channel': {}, 'params': {}}
        if isinstance(chInfo, dict):
            if 'LO' in chInfo:
                hardware['params']['LOFrequency'] = chInfo['LO']

            hardware['params']['sampleRate'] = {}
            for ch in ['I', 'Q', 'IQ', 'Ref']:
                if ch in chInfo:
                    hardware['channel'][ch] = chInfo[ch]
                    sampleRate = chInfo['sampleRate']
                    hardware['params']['sampleRate'][ch] = sampleRate
        elif isinstance(chInfo, str):
            hardware['channel'] = chInfo

        return hardware

    def _getGateConfig(self, name, *qubits) -> dict:
        try:
            gate = self.cfg.getGate(name, *qubits)
            if not isinstance(gate, dict):
                return {'type': 'default', 'params': {}}
            params = gate['params']
            type = gate.get('type', 'default')
        except:
            type = 'default'
            params = {}
        return {'type': type, 'params': params}

    def _getAllQubitLabels(self) -> list[str]:
        return self.cfg.query('AllQubit')
