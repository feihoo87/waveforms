"""
BIG BROTHER IS WATCHING YOU

Reference:
    arXiv:1803.03226v1

Modified:
    Liu Pei
"""

from collections import defaultdict

from .sched import submit, update_parameters, create_task
from .task import CalibrationResult, Task


class CalibrationError(RuntimeError):
    """
    The error occurs in maintain under Big Brother algorithm.
    """
    pass


def check_state(task: Task) -> bool:
    """
    Check whether the parameter node has been updated, via `task.check()`.

    Args:
        task (Task): current task

    Returns:
        bool: whether the parameter node has been updated.
    """
    last_succeed = task.check(lastest=False)  # 时间集合上的序
    if last_succeed < 0:
        return False
    dependents = task.depends()
    if len(dependents) > 0:
        return all(0 < create_task(*taskInfo).check() < last_succeed
                   for taskInfo in dependents)
    else:
        return True


def scan(task: Task,
         calibration_level: int) -> CalibrationResult:
    """Calibrate a task.

    Args:
        task: a task to be calibrated.
        calibration_level: the calibration level.

    Returns:
        A CalibrationResult.
    """
    args, kwds = task.runtime.prog.task_arguments
    kwds['calibration_level'] = calibration_level
    task = create_task(task.name, args, kwds)
    task = submit(task)
    task.join()
    return task.analyze(task.result())


def check_data(task: Task) -> bool:
    """
    Check data of a task.

    Args:
        scheduler: a scheduler
        task: a task to be checked.

    Returns:
        A CalibrationResult.
    """
    return scan(task, task.check_level())


def calibrate(task: Task,
              calibration_level: int) -> CalibrationResult:
    """Calibrate a task.

    Args:
        task: a task to be calibrated.
        calibration_level: the calibration level.

    Returns:
        A CalibrationResult.
    """
    calibration_level = min(max(calibration_level, 0), 100)
    history = defaultdict(lambda: 0)

    while True:
        result = scan(task, calibration_level)
        calibration_level = int(result.suggested_calibration_level)
        if calibration_level < 0:
            raise CalibrationError(
                f'bad data for task {task}, but all previous tasks were passed.'
            )
        if history[calibration_level] >= 2:
            raise CalibrationError(f'Calibration {task} failed, '
                                   'but all previous tasks were passed.')
        history[calibration_level] += 1
        if calibration_level >= 100:
            break
    return result


def maintain(task: Task) -> Task:
    """Maintain a task.
    """
    # recursive maintain
    for taskInfo in task.depends():
        maintain(create_task(*taskInfo))

    # check state
    success = check_state(task)
    if success:
        return task

    # check data
    result = check_data(task)
    if result.suggested_calibration_level >= 100:
        return task
    # elif result.suggested_calibration_level < 0:
        # for taskInfo in task.depends():
        #     diagnose(create_task(taskInfo))

    # calibrate
    result = calibrate(task, task.calibration_level)
    update_parameters(result.parameters)
    return task


def diagnose(task: Task) -> bool:
    """
    Diagnose a task.

    Returns: True if node or dependent recalibrated.
    """
    # check data
    result = check_data(task)

    # in spec case
    if result.suggested_calibration_level == 100:
        return False

    # bad data case
    if result.suggested_calibration_level < 0:
        recalibrated = [
            diagnose(create_task(*taskInfo))
            for taskInfo in task.depends()
        ]
    if not any(recalibrated):
        return False

    # calibrate
    result = calibrate(task, task.calibration_level)
    update_parameters(result)
    return True
