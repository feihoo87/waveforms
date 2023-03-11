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
    importlib.reload(module)
    return module.Instrument


def create_instrument(driver_name: str, *args, **kwds):
    try:
        for package_name, p in path.items():
            driver = load_driver_from_file(
                Path(p) / f"{driver_name}.py", package_name)
            if driver is not None:
                return driver(*args, **kwds)

        module = importlib.import_module(
            f"waveforms.sys.device.drivers.{driver_name}")
        importlib.reload(module)
        return module.Instrument(*args, **kwds)
    except:
        raise RuntimeError(f"Can not find driver {driver_name!r}")
