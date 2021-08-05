from .scheduler import Scheduler
from .task import CalibrationResult, Task, create_task


class CalibrationError(RuntimeError):
    pass


def check_state(task: Task) -> bool:
    last_succeed = task.check()
    if last_succeed < 0:
        return False
    dependents = task.depends()
    if len(dependents) > 0:
        return all(0 < dependent.check() < last_succeed
                   for dependent in dependents)
    else:
        return True


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
    task.calibration_level = calibration_level
    scheduler.submit(task)
    scheduler.join(task)
    return task.analyze(task.result())


def maintain(scheduler: Scheduler, task: Task) -> Task:
    """Maintain a task.
        """
    # recursive maintain
    for n in task.depends():
        maintain(scheduler, create_task(*n))

    # check state
    success = check_state(task)
    if success:
        return task

    # check data
    result = calibrate(scheduler, task, calibration_level=100)

    while result.suggested_calibration_level < 100:
        if result.suggested_calibration_level < 0:
            if len(task.depends()) == 0:
                raise CalibrationError(
                    f'bad data for independent task {task}.')
            for n in task.depends():
                diagnose(scheduler, create_task(*n))
            result.suggested_calibration_level = 0
        else:
            result = calibrate(scheduler, task,
                               result.suggested_calibration_level)

    scheduler.update_parameters(result.parameters)
    return task


def diagnose(scheduler: Scheduler, task: Task) -> bool:
    """
    Diagnose a task.

    Returns: True if node or dependent recalibrated.
    """
    # check data
    result = task.check_data()

    # in spec case
    if result.suggested_calibration_level == 100:
        return False

    # bad data case
    if result.suggested_calibration_level < 0:
        recalibrated = [
            diagnose(scheduler, create_task(*n)) for n in task.depends()
        ]
    if not any(recalibrated):
        return False

    # calibrate
    result.suggested_calibration_level = 0
    while result.suggested_calibration_level < 100:
        result = calibrate(scheduler, task, result.suggested_calibration_level)
        if result.suggested_calibration_level < 0:
            raise CalibrationError(
                f'previous tasks were passed, but {task} failured.')
    scheduler.update_parameters(result)
    return True
