from datetime import datetime

import dill
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .base import Base
from .tag import has_tags


@has_tags
class Record(Base):
    __tablename__ = 'records'

    id = Column(Integer, primary_key=True)
    ctime = Column(DateTime, default=datetime.utcnow)
    mtime = Column(DateTime, default=datetime.utcnow)
    atime = Column(DateTime, default=datetime.utcnow)
    name = Column(String)
    datafile_id = Column(Integer, ForeignKey('files.id'))
    configfile_id = Column(Integer, ForeignKey('files.id'))
    meta_id = Column(Integer, ForeignKey('files.id'))

    datafile = relationship("File", foreign_keys=[datafile_id])
    configfile = relationship("File", foreign_keys=[configfile_id])
    metafile = relationship("File", foreign_keys=[meta_id])

    @property
    def data(self) -> dict:
        result = dill.loads(self.datafile.read())
        result['meta'] = self.meta
        result['meta']['config'] = self.config
        return result

    @data.setter
    def data(self, data: dict):
        meta = data.pop('meta', {})
        self.meta = meta
        self.datafile.write(dill.dumps(data))

    @property
    def config(self) -> dict:
        return dill.loads(self.configfile.read())

    @config.setter
    def config(self, data: dict):
        self.configfile.write(dill.dumps(data))

    @property
    def meta(self) -> dict:
        meta = dill.loads(self.metafile.read())
        meta['id'] = self.id
        meta['name'] = self.name
        meta['ctime'] = self.ctime
        meta['mtime'] = self.mtime
        meta['atime'] = self.atime
        return meta

    @meta.setter
    def meta(self, data: dict):
        if 'config' in data:
            config = data.pop('config', {})
            self.config = config
        self.metafile.write(dill.dumps(data))

    def export(self, path):
        with open(path, 'wb') as f:
            f.write(self.data)

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self.data = data

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'Record({self.name!r})'
