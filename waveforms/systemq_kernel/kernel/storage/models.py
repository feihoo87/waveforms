import warnings

warnings.warn(
    ("This module is deprecated and will be removed in a future release. "
     "Please use the new module `storage.models` instead."),
    DeprecationWarning, 2)

from storage.models import (Attachment, Base, Cell, Comment, InputText,
                            Notebook, Parameter, ParameterEdge, Record, Report,
                            ReportParameters, Role, Sample, SampleAccount,
                            SampleAccountType, SampleTransfer, Snapshot, Tag,
                            User, _load_object, _save_object, create_tables,
                            get_data_path, set_data_path)
