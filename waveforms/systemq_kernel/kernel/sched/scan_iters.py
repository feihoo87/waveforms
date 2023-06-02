import warnings

warnings.warn(
    ("This module is deprecated and will be removed in a future release. "
     "Please use the new module `waveforms.scan_iter` instead."),
    DeprecationWarning, 2)

try:
    from waveforms.scan_iter import scan_iters, OptimizerConfig, StepStatus, Tracker
except ImportError:
    raise ImportError('Please update the waveforms package.')
