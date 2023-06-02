import warnings

warnings.warn(
    ("This module is deprecated and will be removed in a future release. "
     "Please use the new module `storage.hdf5_utils` instead."),
    DeprecationWarning, 2)

from storage.hdf5_utils import (dataarray_from_hdf, dataarray_to_hdf,
                                dataset_from_hdf, dataset_to_hdf, is_dataarray,
                                is_dataset)
