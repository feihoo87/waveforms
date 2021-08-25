from waveforms.namespace import NamespaceCache, NOTSET, DictDriver
import pytest
import copy


@pytest.fixture
def data():
    from config import config
    return copy.deepcopy(config)


def test_DictDriver(data):
    d = DictDriver(data)

    assert d.query('gates.rfUnitary.Q1.params.frequency') == 4675808085.0
    assert d.query('gates.rfUnitary.Q2.params.frequency') == 4354202438.483826
    assert d.query('gates.CZ') == data['gates']['CZ']
    assert set(d.keys('gates')) == {
        'Measure', 'Reset', 'rfUnitary', 'CZ', 'iSWAP', 'CR'
    }
    assert set(d.keys('*')) == {'__version__', 'station', 'chip', 'gates'}
    assert set(d.keys('gates.C*')) == {'CZ', 'CR'}
    d.delete_many(['gates.rfUnitary.Q1', 'gates.rfUnitary.Q2.params.duration'])
    assert data['gates']['rfUnitary'] == {
        "Q2": {
            "type": "default",
            "params": {
                "shape": "CosPulse",
                "frequency": 4354202438.483826,
                "amp": [[0, 1], [0, 0.658]],
                "phase": [[-1, 1], [-1, 1]],
                "DRAGScaling": 9.866314574173999e-10
            }
        }
    }
    assert d.query('gates.rfUnitary.Q1.params.frequency') == (
        NOTSET, 'gates.rfUnitary.Q1')


def test_DictDriver_update(data):
    d = DictDriver(data)

    d.update_many({
        'gates.rfUnitary.Q1.params.frequency': 4.6e9,
        'gates.rfUnitary.Q2.params.frequency': 4.3e9
    })

    assert d.query('gates.rfUnitary.Q1.params.frequency') == 4.6e9
    assert d.query('gates.rfUnitary.Q2.params.frequency') == 4.3e9

    d.update_many({
        'gates.rfUnitary.Q1.params': {
            "shape": "CosPulse",
            "frequency": 4675808085.0,
            "amp": [[0, 1], [0, 0.8204]],
            "duration": [[0, 1], [2e-08, 2e-08]],
            "phase": [[-1, 1], [-1, 1]],
            "DRAGScaling": 0.1
        },
        'gates.rfUnitary.Q1.params.frequency': 4.4e9
    })

    assert d.query('gates.rfUnitary.Q1.params.frequency') == 4.4e9
    assert d.query('gates.rfUnitary.Q1.params.DRAGScaling') == 0.1


def test_cache():
    log = NamespaceCache()
    log.cache('gate.rfUnitary.Q0',
              {'params': {
                  'frequency': 4.5e9,
                  'amp': 0.4,
                  'duration': 50e-9
              }})

    log.cache('gate.rfUnitary.Q1',
              {'params': {
                  'frequency': 4.6e9,
                  'amp': 0.41,
                  'duration': 52e-9
              }})

    log.cache(
        'gate.CZ', {
            'Q0_Q1': {
                'params': {
                    'amp': 0.25,
                    'duration': 50e-9,
                    'phi1': 0,
                    'phi2': 0
                }
            },
            'Q1_Q2': {
                'params': {
                    'amp': 0.15,
                    'duration': 50e-9,
                    'phi1': 0,
                    'phi2': 0
                }
            },
        })

    log.cache('gate.rfUnitary.Q4.params', (NOTSET, 'gate.rfUnitary.Q4'))

    assert log.query('gate.rfUnitary.Q0.params') == ({
        'frequency': 4.5e9,
        'amp': 0.4,
        'duration': 50e-9
    }, True)
    assert log.query('gate.rfUnitary.Q0.params.frequency') == (4.5e9, True)
    assert log.query('gate.rfUnitary') == ({
        'Q0': {
            'params': {
                'frequency': 4.5e9,
                'amp': 0.4,
                'duration': 50e-9
            }
        },
        'Q1': {
            'params': {
                'frequency': 4.6e9,
                'amp': 0.41,
                'duration': 52e-9
            }
        }
    }, False)
    assert log.query('gate.rfUnitary.Q3') == (None, False)
    assert log.keys('gate') is None
    assert log.keys('gate.rfUnitary') is None
    assert set(log.keys('gate.rfUnitary.Q0')) == {'params'}
    assert set(log.keys('gate.rfUnitary.Q0.params')) == {
        'frequency', 'amp', 'duration'
    }
    assert set(log.keys('gate.rfUnitary.Q0.params.frequency')) == set()
    assert set(log.keys('gate.CZ')) == {'Q0_Q1', 'Q1_Q2'}
    assert set(log.keys('gate.CZ.Q0*')) == {'Q0_Q1'}
    assert set(log.keys('gate.CZ.Q1*')) == {'Q1_Q2'}
