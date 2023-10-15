import warnings

from waveforms.qlisp.simulator import (apply_circuit, apply_gate, applySeq,
                                       circuit2mat, circuit_network, gate2mat,
                                       place_at, reduceSubspace,
                                       regesterGateMatrix, seq2mat, splite_at)
from waveforms.qlisp.simulator.mat import U, fSim, make_immutable, rfUnitary

warnings.warn(
    'waveforms.quantum.circuit.simulator is deprecated, please use waveforms.qlisp.simulator instead',
    DeprecationWarning,
    stacklevel=2,
)
