import hashlib
import pickle
import time
import zipfile
import itertools
from datetime import datetime
from typing import Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from sqlalchemy import (BLOB, DECIMAL, Column, DateTime, Float, ForeignKey,
                        ForeignKeyConstraint, Integer, Sequence, String, Table,
                        Text, create_engine)
from sqlalchemy.orm import (backref, declarative_base, relationship,
                            sessionmaker)
from sqlalchemy.orm.session import Session
from sqlalchemy.util.compat import u
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

sample_tags = Table(
    'sample_tags', Base.metadata,
    Column('sample_id', ForeignKey('samples.id'), primary_key=True),
    Column('tag_id', ForeignKey('tags.id'), primary_key=True))

sample_reports = Table(
    'sample_reports', Base.metadata,
    Column('sample_id', ForeignKey('samples.id'), primary_key=True),
    Column('report_id', ForeignKey('reports.id'), primary_key=True))

sample_records = Table(
    'sample_records', Base.metadata,
    Column('sample_id', ForeignKey('samples.id'), primary_key=True),
    Column('record_id', ForeignKey('records.id'), primary_key=True))


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
    samples = relationship('Sample',
                           secondary=sample_tags,
                           back_populates='tags')

    def __repr__(self):
        return f"Tag(text='{self.text}')"


class InputText(Base):
    __tablename__ = 'inputs'

    id = Column(Integer, primary_key=True)
    hash = Column(String(20))
    text_field = Column(Text, unique=True)

    @property
    def text(self):
        return self.text_field

    @text.setter
    def text(self, text):
        self.hash = hashlib.sha1(text.encode('utf-8')).digest()
        self.text_field = text


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
    name = Column(String)
    ctime = Column(DateTime, default=datetime.utcnow)
    atime = Column(DateTime, default=datetime.utcnow)

    cells = relationship("Cell",
                         order_by=Cell.index,
                         back_populates="notebook")


class Sample(Base):
    __tablename__ = 'samples'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(Text)

    account_id = Column(Integer, ForeignKey("sample_accounts.id"))

    tags = relationship("Tag", secondary=sample_tags, back_populates="samples")
    records = relationship("Record",
                           secondary=sample_records,
                           back_populates="samples")
    reports = relationship("Report",
                           secondary=sample_reports,
                           back_populates="samples")
    transfers = relationship("SampleTransfer", back_populates="sample")
    account = relationship("SampleAccount", back_populates="samples")


class SampleAccount(Base):
    __tablename__ = 'sample_accounts'

    id = Column(Integer, primary_key=True)
    name = Column(String)

    samples = relationship("Sample", back_populates="account")


class SampleTransfer(Base):
    __tablename__ = 'sample_transfer'

    id = Column(Integer, primary_key=True)
    sample_id = Column(Integer, ForeignKey("samples.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    ctime = Column(DateTime, default=datetime.utcnow)
    debtor_id = Column(Integer, ForeignKey("sample_accounts.id"))
    creditor_id = Column(Integer, ForeignKey("sample_accounts.id"))

    user = relationship("User")
    sample = relationship("Sample", back_populates="transfers")
    debtor = relationship("SampleAccount", foreign_keys=[debtor_id])
    creditor = relationship("SampleAccount", foreign_keys=[creditor_id])


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
    config = Column(String)
    comment = Column(Text)

    user = relationship("User")
    samples = relationship("Sample",
                           secondary=sample_records,
                           back_populates="records")

    reports = relationship('Report',
                           secondary=record_reports,
                           back_populates='records')
    tags = relationship('Tag', secondary=record_tags, back_populates='records')

    def __init__(self,
                 file: str = 'test.h5',
                 key: Optional[str] = None,
                 dims: list[str] = [],
                 vars: list[str] = [],
                 dims_units: list[str] = [],
                 vars_units: list[str] = [],
                 coords: Optional[dict] = None):
        self.file = file
        if key is None:
            self.key = '/Data' + time.strftime("%Y%m%d%H%M%S")
        else:
            self.key = key
        self.ds: xr.Dataset = None
        self.df: pd.DataFrame = None
        self.dims = dims
        self.vars = vars
        self.dims_units = dims_units
        self.vars_units = vars_units
        self.coords = coords

        self._buff = ([], [])

    def flush(self):
        if len(self._buff[0]) == 0:
            if self.df is not None and self.ds is None:
                self._df_to_ds()
            return

        self.mtime = datetime.utcnow()
        self.atime = datetime.utcnow()
        index, values = self._buff
        df = pd.DataFrame(values,
                          index=pd.MultiIndex.from_tuples(index,
                                                          names=self.dims),
                          columns=self.vars)
        if self.df is None:
            self.df = df
        else:
            self.df = self.df.append(df)
        self._df_to_ds()
        self._buff = ([], [])

    def _df_to_ds(self):
        self.ds = xr.Dataset.from_dataframe(self.df)
        for units, var in zip(self.vars_units, self.vars):
            self.ds[var].attrs['units'] = units
        for dim, units in zip(self.dims, self.dims_units):
            self.ds[dim].attrs['units'] = units

    def append(self, index, values):
        if self.coords is None:
            self._buff[0].extend(index)
            self._buff[1].extend(values)
        else:

            def getitem(y, s):
                ret = y
                for i in s:
                    ret = ret[i]
                return ret

            shape = [len(v) for v in self.coords.values()]
            for i, v in zip(index, values):
                for n, c in enumerate(
                        itertools.product(*self.coords.values())):
                    s = np.unravel_index(n, shape)
                    self._buff[0].append(tuple(i) + c)
                    self._buff[1].append(tuple(getitem(y, s) for y in v))

        if len(self._buff[0]) > 1000:
            self.flush()

    def set_values(self, *values):
        self.df = pd.DataFrame(dict(zip(self.columnLabels, values)))

    def __getitem__(self, label):
        return self.ds[label]

    def save(self):
        self.flush()
        Path(self.file).parent.mkdir(parents=True, exist_ok=True)
        self.ds.to_netcdf(self.file,
                          group=self.key,
                          mode='a',
                          format='NETCDF4',
                          engine='netcdf4')

    def data(self):
        self.atime = datetime.utcnow()
        self.flush()
        if self.ds is None:
            self.ds = xr.open_dataset(self.file,
                                      group=self.key,
                                      mode='r',
                                      engine='netcdf4')
        return self.ds


class Report(Base):
    __tablename__ = 'reports'

    id = Column(Integer, primary_key=True)
    ctime = Column(DateTime, default=datetime.utcnow)
    mtime = Column(DateTime, default=datetime.utcnow)
    atime = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey('users.id'))

    category = Column(String)
    title = Column(String)
    content = Column(Text)
    file = Column(String)
    key = Column(String)

    user = relationship("User")
    samples = relationship("Sample",
                           secondary=sample_reports,
                           back_populates="reports")

    records = relationship('Record',
                           secondary=record_reports,
                           back_populates='reports')

    tags = relationship('Tag', secondary=report_tags, back_populates='reports')

    @property
    def data(self):
        with zipfile.ZipFile(self.file, 'r') as z:
            with z.open(self.key) as f:
                return pickle.loads(f.read())

    @data.setter
    def data(self, data):
        Path(self.file).parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.file, 'a') as z:
            with z.open(self.key + "/data.pickle", 'w') as f:
                f.write(pickle.dumps(data))

    @property
    def images(self):
        image_dir = zipfile.Path(self.file) / self.key / 'images'
        return {i.name: i.read_bytes() for i in image_dir.iterdir()}

    def add_image(self, name, image):
        Path(self.file).parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.file, 'a') as z:
            with z.open(self.key + "/images/" + name, 'w') as f:
                f.write(image)


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
