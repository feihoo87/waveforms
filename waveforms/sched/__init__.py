from .base import READ, TRIG, WRITE
from .progress import JupyterProgressBar, Progress, ProgressBar
from .scheduler import bootstrap, login
from .task import App, CalibrationResult, RunCircuits, UserInput
import warnings


def Scheduler(*args, **kwargs):
    warnings.warn(
        "waveforms.sched.Scheduler is deprecated,"
        " use waveforms.sched.bootstrap instead.", DeprecationWarning, 2)
    return bootstrap(*args, **kwargs)
