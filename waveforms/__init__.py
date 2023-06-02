from numpy import e, pi

#from .cache import cache
#from .qlisp import Config, compile, libraries, stdlib
#from .qlisp.config import getConfig, setConfig
#from .quantum import qpt, qptInitList, qst, qst_mle, qstOpList
#from .quantum.transmon import Transmon
from .version import __version__
from .waveform import (D, Waveform, chirp, const, cos, cosh, coshPulse,
                       cosPulse, cut, exp, function, gaussian, general_cosine,
                       hanning, interp, mixing, one, poly, registerBaseFunc,
                       registerDerivative, samplingPoints, sign, sin, sinc,
                       sinh, square, step, zero)
from .waveform_parser import wave_eval


def _use_systemq():
    import sys
    import pathlib
    import waveforms.systemq_kernel

    path = str(pathlib.Path(waveforms.systemq_kernel.__file__).parent)
    if path not in sys.path:
        sys.path.insert(0, path)


_use_systemq()
