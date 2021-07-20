import hashlib
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import (BLOB, DECIMAL, Column, DateTime, Float, ForeignKey,
                        ForeignKeyConstraint, Integer, Sequence, String, Table,
                        Text, create_engine)
from sqlalchemy.orm import (backref, declarative_base, relationship,
                            sessionmaker)
from sqlalchemy.orm.session import Session
from waveforms.security import InvalidKey, encryptPassword, verifyPassword

Base = declarative_base()

# association table
user_roles = Table('user_roles', Base.metadata,
                   Column('user_id', ForeignKey('users.id'), primary_key=True),
                   Column('role_id', ForeignKey('roles.id'), primary_key=True))

record_reports = Table(
    'record_reports', Base.metadata,
    Column('record_id', ForeignKey('records.id'), primary_key=True),
    Column('report_id', ForeignKey('reports.id'), primary_key=True))

record_tags = Table(
    'record_tags', Base.metadata,
    Column('record_id', ForeignKey('records.id'), primary_key=True),
    Column('tag_id', ForeignKey('tags.id'), primary_key=True))

report_tags = Table(
    'report_tags', Base.metadata,
    Column('report_id', ForeignKey('reports.id'), primary_key=True),
    Column('tag_id', ForeignKey('tags.id'), primary_key=True))


class Role(Base):
    __tablename__ = 'roles'

    id = Column(Integer, primary_key=True)
    name = Column(String)

    users = relationship('User', secondary=user_roles, back_populates='roles')

    def __repr__(self):
        return f"Role(name='{self.name}')"


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    hashed_password = Column(String(64))
    fullname = Column(String)

    roles = relationship('Role', secondary=user_roles, back_populates='users')

    def setPassword(self, password):
        self.hashed_password = encryptPassword(password)

    def verify(self, password):
        try:
            verifyPassword(password, self.hashed_password)
            return True
        except InvalidKey:
            return False

    def __repr__(self):
        return f"User(name='{self.name}')"


class Tag(Base):
    __tablename__ = 'tags'

    id = Column(Integer, primary_key=True)
    text = Column(String, unique=True)

    records = relationship('Record',
                           secondary=record_tags,
                           back_populates='tags')
    reports = relationship('Report',
                           secondary=report_tags,
                           back_populates='tags')

    def __repr__(self):
        return f"Tag(text='{self.text}')"


class InputText(Base):
    __tablename__ = 'inputs'

    id = Column(Integer, primary_key=True)
    hash = Column(String(20))
    text_field = Column(Text)

    @property
    def text(self):
        return self.text_field

    @text.setter
    def text(self, text):
        self.hash = hashlib.sha1(text.encode('utf-8')).digest()
        self.text = text


class Cell(Base):
    __tablename__ = 'cells'

    id = Column(Integer, primary_key=True)
    notebook_id = Column(Integer, ForeignKey("notebooks.id"))
    index = Column(Integer)
    ctime = Column(DateTime, default=datetime.utcnow)
    ftime = Column(DateTime, default=datetime.utcnow)
    input_id = Column(Integer, ForeignKey("inputs.id"))

    notebook = relationship("Notebook", back_populates="cells")
    input = relationship("InputText")


class Notebook(Base):
    __tablename__ = 'notebooks'

    id = Column(Integer, primary_key=True)
    ctime = Column(DateTime, default=datetime.utcnow)
    atime = Column(DateTime, default=datetime.utcnow)

    cells = relationship("Cell",
                         order_by=Cell.index,
                         back_populates="notebook")


class Record(Base):
    __tablename__ = 'records'

    id = Column(Integer, primary_key=True)
    ctime = Column(DateTime, default=datetime.utcnow)
    mtime = Column(DateTime, default=datetime.utcnow)
    atime = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey('users.id'))

    app = Column(String)
    file = Column(String)
    key = Column(String)
    comment = Column(Text)

    reports = relationship('Report',
                           secondary=record_reports,
                           back_populates='records')
    tags = relationship('Tag', secondary=record_tags, back_populates='records')

    def __init__(self,
                 file='test.h5',
                 key=None,
                 indexLabels=[],
                 columnLabels=[]):
        self.file = file
        if key is None:
            self.key = '/Data' + time.strftime("%Y%m%d%H%M%S")
        else:
            self.key = key
        self.df = None
        self.indexLabels = indexLabels
        self.columnLabels = columnLabels

        self._buff = ([], [])

    def flush(self):
        if len(self._buff[0]) == 0:
            return

        self.mtime = datetime.utcnow()
        self.atime = datetime.utcnow()
        index, values = self._buff
        df = pd.DataFrame(values,
                          index=pd.MultiIndex.from_tuples(
                              index, names=self.indexLabels),
                          columns=self.columnLabels)
        if self.df is None:
            self.df = df
        else:
            self.df = self.df.append(df)
        self._buff = ([], [])

    def append(self, index, values):
        self._buff[0].extend(index)
        self._buff[1].extend(values)

        if len(self._buff[0]) > 1000:
            self.flush()

    def set_values(self, *values):
        self.df = pd.DataFrame(dict(zip(self.columnLabels, values)))

    def __getitem__(self, label):
        if label in self.indexLabels:
            return np.asarray(self.df.index.get_level_values(label))
        else:
            return np.asarray(self.df[label])

    def save(self):
        self.flush()
        self.df.to_hdf(self.file, self.key)

    def data(self):
        self.atime = datetime.utcnow()
        return pd.read_hdf(self.file, self.key)


class Report(Base):
    __tablename__ = 'reports'

    id = Column(Integer, primary_key=True)
    ctime = Column(DateTime, default=datetime.utcnow)
    mtime = Column(DateTime, default=datetime.utcnow)
    atime = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey('users.id'))

    records = relationship('Record',
                           secondary=record_reports,
                           back_populates='reports')

    tags = relationship('Tag', secondary=report_tags, back_populates='reports')


def create_tables(engine):
    Base.metadata.create_all(engine)

    root_role = Role(name='root')
    admin_role = Role(name='admin')
    root_user = User(name='root')
    root_user.setPassword('123')
    root_user.roles.append(root_role)
    root_user.roles.append(admin_role)

    Session = sessionmaker(bind=engine)
    session = Session()

    session.add(root_user)
    session.commit()
