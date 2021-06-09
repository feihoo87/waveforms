import pytest
from waveforms.quantum import compile, libraries, stdlib
from waveforms.quantum.circuit.qlisp.library import Library
from waveforms.quantum.circuit.qlisp.qlisp import QLispCode

from config import config

qasm = """
OPENQASM 2.0;

include "qelib1.inc";
opaque iSWAP c,t;

gate createBellPair a, b
{
   h a;
   cx a,b;
}

gate bellMeasure a, b
{
   cx a, b;
   h a;
}

qreg q[2];
creg c[2];

createBellPair q[0], q[1];
iSWAP q[0],q[1];
bellMeasure q[0],q[1];
barrier q;
measure q -> c;
"""

qlisp = [
    ('createBellPair', (0, 1)),
    ('iSWAP', (0, 1)),
    ('bellMeasure', (0, 1)),
    ('Barrier', (0, 1)),
    (('Measure', 0), 0),
    (('Measure', 1), 1),
]


@pytest.fixture
def cfg():
    from waveforms.quantum.circuit.qlisp.config import Config
    try:
        yield Config.fromdict(config)
    finally:
        pass


@pytest.fixture
def lib():
    from numpy import pi

    lib = libraries(stdlib)

    @lib.opaque('CZ')
    def CZ(ctx, qubits):
        control, target = qubits
        from waveforms.waveform import cos, square

        gate = ctx.cfg.getGate('CZ', control, target)
        duration = gate.params.duration
        amp = gate.params.amp
        t = max(ctx.time[control], ctx.time[target])

        if amp > 0 and duration > 0:
            pulse = (cos(pi / duration) * square(duration)) >> duration / 2
            ctx.channel['coupler.Z', control, target] += amp * pulse >> t
        ctx.time[control] = ctx.time[target] = t + duration

    @lib.opaque('iSWAP')
    def iSWAP(ctx, qubits):
        from waveforms.waveform import sin, square

        control, target = qubits

        gate = ctx.cfg.getGate('iSWAP', control, target)
        duration = gate.params.duration
        amp = gate.params.amp
        offset = gate.params.offset
        frequency = gate.params.frequency

        t = max(ctx.time[control], ctx.time[target])
        if duration > 0:
            pulse = square(duration) >> duration / 2
            pulse = pulse * offset + amp * pulse * sin(2 * pi * frequency)
            ctx.channel['coupler.Z', control, target] += pulse >> t
        ctx.time[control] = ctx.time[target] = t + duration

    @lib.gate(2)
    def bellMeasure(qubits):
        a, b = qubits
        yield ('Cnot', (a, b))
        yield ('H', a)

    @lib.gate(2)
    def createBellPair(qubits):
        a, b = qubits
        yield ('H', a)
        yield ('Cnot', (a, b))

    try:
        yield lib
    finally:
        pass


def test_lib(lib):
    assert isinstance(lib, Library)


def test_compile(lib, cfg):
    ret = compile(qasm, cfg=cfg, lib=lib)
    assert isinstance(ret, QLispCode)
    ret2 = compile(qlisp, cfg=cfg, lib=lib)
    assert isinstance(ret2, QLispCode)
    for k, wav in ret.waveforms.items():
        assert k in ret2.waveforms
        assert wav == ret2.waveforms[k]
