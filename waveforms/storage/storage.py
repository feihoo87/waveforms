import shutil
import uuid
from pathlib import Path

from .file import load


class Storage():

    def __init__(self, base: Path):
        self.base = base
        self.namespace = uuid.uuid5(
            uuid.UUID('f89f735a-791e-5a43-9ba6-f28d58601544'), base.as_posix())

    def get(self, key: uuid.UUID):
        return load(self.uuid_to_path(key))

    def uuid(self,
             name: str | None = None,
             namespace: uuid.UUID | None = None,
             seq: int = 0) -> uuid.UUID:
        if name is None:
            name = str(uuid.uuid1())
        if namespace is None:
            return uuid.uuid5(self.namespace, f"{name}{seq}")
        else:
            return uuid.uuid5(namespace, f"{name}{seq}")

    def uuid_to_path(self, uuid: uuid.UUID) -> Path:
        return self.base / uuid.hex[:2] / uuid.hex[2:4] / uuid.hex[4:]

    def create_dataset(self):
        from .dataset import Dataset
        id = self.uuid()
        return Dataset(id, self)

    def remove_dataset(self, id: uuid.UUID):
        shutil.rmtree(self.uuid_to_path(id))

    def clear(self):
        shutil.rmtree(self.base)
        self.base.mkdir(parents=True)
