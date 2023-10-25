import pytest
from config import config

from waveforms.qlisp import (Library, Parameter, QLispCode, compile, libraries,
                             stdlib)

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

qlisp3 = [
    ('createBellPair', (0, 1)),
    (('iSWAP', ('with', ('param:amp', 0.5))), (0, 1)),
    ('bellMeasure', (0, 1)),
    ('Barrier', (0, 1)),
    (('Measure', 0), 0),
    (('Measure', 1), 1),
]

qlisp4 = [
    ('define', 'Q1', 0),
    ('define', 'Q2', 1),
    ('createBellPair', ('Q1', 'Q2')),
    (('iSWAP', ('with', ('param:amp', lambda amp: amp - 0.3))), ('Q1', 'Q2')),
    ('bellMeasure', (0, 1)),
    ('Barrier', (0, 1)),
    (('Measure', 0), 0),
    (('Measure', 1), 1),
]


@pytest.fixture
def cfg():
    from waveforms.qlisp import Config
    try:
        yield Config.fromdict(config)
    finally:
        pass


@pytest.fixture
def lib():
    from numpy import pi

    lib = libraries(stdlib)

    @lib.opaque('CZ',
                params=[
                    Parameter('duration', float, 50e-9, 's'),
                    Parameter('amp', float, 0.8, 'a.u.')
                ])
    def CZ(ctx, qubits):
        control, target = qubits
        from waveforms.waveform import cos, square

        duration = ctx.params['duration']
        amp = ctx.params['amp']
        t = max(ctx.time[control], ctx.time[target])

        if amp > 0 and duration > 0:
            pulse = (cos(pi / duration) * square(duration)) >> duration / 2
            yield ('!play', amp * pulse >> t), ('coupler.Z', control, target)
        yield ('!add_time', t + duration), control
        yield ('!add_time', t + duration), target

    @lib.opaque('iSWAP',
                params=[('duration', float, 50e-9, 's'),
                        ('amp', float, 0.8, 'a.u.'),
                        ('offset', float, 0, 'a.u.'),
                        ('frequency', float, 10e6, 'Hz')])
    def iSWAP(ctx, qubits):
        from waveforms.waveform import sin, square

        control, target = qubits

        duration = ctx.params['duration']
        amp = ctx.params['amp']
        offset = ctx.params['offset']
        frequency = ctx.params['frequency']

        t = max(ctx.time[control], ctx.time[target])
        if duration > 0:
            pulse = square(duration) >> duration / 2
            pulse = pulse * offset + amp * pulse * sin(2 * pi * frequency)
            yield ('!play', amp * pulse >> t), ('coupler.Z', control, target)
        yield ('!add_time', t + duration), control
        yield ('!add_time', t + duration), target

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
    ret3 = compile(qlisp3, cfg=cfg, lib=lib)
    assert isinstance(ret3, QLispCode)
    ret4 = compile(qlisp4, cfg=cfg, lib=lib)
    assert isinstance(ret4, QLispCode)
    assert 'AWG.Z' in ret.waveforms
    for k, wav in ret.waveforms.items():
        assert k in ret2.waveforms
        assert wav.simplify() == ret2.waveforms[k].simplify()
        if k == 'AWG.Z':
            assert wav.simplify() != ret3.waveforms[k].simplify()
            assert wav.simplify() != ret4.waveforms[k].simplify()
            assert ret3.waveforms[k].simplify() == ret4.waveforms[k].simplify()
        else:
            assert wav.simplify() == ret3.waveforms[k].simplify()
            assert wav.simplify() == ret4.waveforms[k].simplify()
