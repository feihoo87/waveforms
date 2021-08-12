"""
BIG BROTHER IS WATCHING YOU

Reference:
    arXiv:1803.03226v1
"""

from collections import defaultdict

from .scheduler import Scheduler
from .task import CalibrationResult, Task, copy_task, create_task


class CalibrationError(RuntimeError):
    pass


def check_state(task: Task) -> bool:
    last_succeed = task.check()
    if last_succeed < 0:
        return False
    dependents = task.depends()
    if len(dependents) > 0:
        return all(0 < create_task(taskInfo).check() < last_succeed
                   for taskInfo in dependents)
    else:
        return True


def scan(scheduler: Scheduler, task: Task,
         calibration_level: int) -> CalibrationResult:
    """Calibrate a task.

    Args:
        scheduler: a scheduler
        task: a task to be calibrated.
        calibration_level: the calibration level.

    Returns:
        A CalibrationResult.
    """
    task = copy_task(task)
    task.calibration_level = calibration_level
    scheduler.submit(task)
    scheduler.join(task)
    return task.analyze(task.result())


def check_data(scheduler: Scheduler, task: Task) -> bool:
    """
    Check data of a task.

    Args:
        scheduler: a scheduler
        task: a task to be checked.

    Returns:
        A CalibrationResult.
    """
    return scan(scheduler, task, 100)


def calibrate(scheduler: Scheduler, task: Task,
              calibration_level: int) -> CalibrationResult:
    """Calibrate a task.

    Args:
        scheduler: a scheduler
        task: a task to be calibrated.
        calibration_level: the calibration level.

    Returns:
        A CalibrationResult.
    """
    calibration_level = min(max(calibration_level, 0), 100)
    history = defaultdict(lambda: 0)

    while True:
        result = scan(scheduler, task, calibration_level)
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


def maintain(scheduler: Scheduler, task: Task) -> Task:
    """Maintain a task.
        """
    # recursive maintain
    for taskInfo in task.depends():
        maintain(scheduler, create_task(taskInfo))

    # check state
    success = check_state(task)
    if success:
        return task

    # check data
    result = check_data(scheduler, task)
    if result.suggested_calibration_level >= 100:
        return task
    elif result.suggested_calibration_level < 0:
        if len(task.depends()) == 0:
            raise CalibrationError(f'bad data for independent task {task}.')
        for taskInfo in task.depends():
            diagnose(scheduler, create_task(taskInfo))

    # calibrate
    result = calibrate(scheduler, task, result.suggested_calibration_level)
    scheduler.update_parameters(result.parameters)
    return task


def diagnose(scheduler: Scheduler, task: Task) -> bool:
    """
    Diagnose a task.

    Returns: True if node or dependent recalibrated.
    """
    # check data
    result = check_data(scheduler, task)

    # in spec case
    if result.suggested_calibration_level == 100:
        return False

    # bad data case
    if result.suggested_calibration_level < 0:
        recalibrated = [
            diagnose(scheduler, create_task(taskInfo))
            for taskInfo in task.depends()
        ]
    if not any(recalibrated):
        return False

    # calibrate
    result = calibrate(scheduler, task, result.suggested_calibration_level)
    scheduler.update_parameters(result)
    return True
