from .bootstrap import bootstrap
from .sched import sched as scheduler
from .sched.sched import (after_task_finished, before_task_start,
                          before_task_step, cancel, create_task, exec, get,
                          get_config, get_executor, get_system_info,
                          register_hook, session, set, submit,
                          unregister_all_hooks, unregister_hook,
                          update_parameters)
from .sched.task import set_default_lib, update_tags

bootstrap()

executor = get_executor()
cfg = get_config()
