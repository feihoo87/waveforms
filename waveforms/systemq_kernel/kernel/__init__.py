from .bootstrap import bootstrap
from .sched import sched as scheduler
from .sched.sched import (cancel, create_task, exec, get, get_config,
                          get_executor, get_system_info, session, set, submit,
                          update_parameters)
from .sched.task import set_default_lib, update_tags

bootstrap()

executor = get_executor()
cfg = get_config()
