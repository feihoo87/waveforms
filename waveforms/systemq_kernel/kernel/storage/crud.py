import warnings

warnings.warn(
    ("This module is deprecated and will be removed in a future release. "
     "Please use the new module `storage.crud` instead."), DeprecationWarning,
    2)

from storage.crud import (create_cell, create_input_text, create_notebook,
                          create_record, create_role, create_sample,
                          create_sample_account, create_user,
                          get_all_notebooks, get_all_records, get_all_reports,
                          get_all_roles, get_all_samples, get_all_tags,
                          get_all_users, get_object_with_tags, login, tag,
                          tag_it, transform_sample)
