import copy

from waveforms.qlisp import Config

from config import config


def test_config():
    cfg = Config.fromdict(config)
    assert isinstance(cfg, Config)
    cfg2 = copy.deepcopy(cfg)
    assert isinstance(cfg2, dict)
    assert not isinstance(cfg2, Config)

    g = cfg.getGate('rfUnitary', 'Q1')
    assert g.amp(3.1) < g.params.amp[1][-1]
