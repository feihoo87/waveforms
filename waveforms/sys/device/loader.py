import importlib
from pathlib import Path
from typing import Type

from .instrument import BaseInstrument

path = {}


def load_driver_from_file(filepath: str | Path,
                          package_name: str) -> Type[BaseInstrument]:
    filepath = Path(filepath)
    module_name = f"{package_name}.{filepath.stem}"
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    return module.Instrument


def create_instrument(driver_name: str, *args, **kwds):
    for package_name, p in path.items():
        driver = load_driver_from_file(
            Path(p) / f"{driver_name}.py", package_name)
        if driver is not None:
            return driver(*args, **kwds)
    try:
        module = importlib.import_module(
            f"waveforms.device.drivers.{driver_name}")
        return module.Instrument(*args, **kwds)
    except:
        raise RuntimeError(f"Can not find driver {driver_name!r}")
