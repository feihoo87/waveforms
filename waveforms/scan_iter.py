import warnings

warnings.warn(
    f"The `waveforms.scan_iter` module is deprecated and will be removed in a future release. "
    f"Please use `waveforms.scan` instead. The `Storage` class is renamed as `BaseDataset` in"
    f"module `waveforms.storage.base_dataset`.", DeprecationWarning, 2)

from waveforms.scan.base import (BaseOptimizer, Begin, End, OptimizerConfig,
                                 StepStatus, Tracker, scan_iters)
from waveforms.storage.base_dataset import BaseDataset as Storage
