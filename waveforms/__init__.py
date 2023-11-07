from numpy import e, pi

from .multy_drag import drag_sin, drag_sinx
from .version import __version__
from .waveform import (D, Waveform, WaveVStack, chirp, const, cos, cosh,
                       coshPulse, cosPulse, cut, drag, exp, function, gaussian,
                       general_cosine, hanning, interp, mixing, one, poly,
                       registerBaseFunc, registerDerivative, samplingPoints,
                       sign, sin, sinc, sinh, square, step, t, zero)
from .waveform_parser import wave_eval
