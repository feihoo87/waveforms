from .quarkconfig import QuarkConfig
from .quarkcontext import QuarkContext


def set_up_backend(host='127.0.0.1'):
    from waveforms.quantum.circuit.qlisp.qlisp import set_context_factory
    from waveforms.quantum.circuit.qlisp.config import set_config_factory

    set_config_factory(lambda: QuarkConfig(host=host))
    set_context_factory(QuarkContext)
