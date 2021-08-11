import hashlib
import itertools
import pickle
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import xarray as xr
from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer, Boolean,
                        LargeBinary, Sequence, String, Table, Text,
                        create_engine)
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

comment_tags = Table(
    'comment_tags', Base.metadata,
    Column('comment_id', ForeignKey('comments.id'), primary_key=True),
    Column('tag_id', ForeignKey('tags.id'), primary_key=True))

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

sample_comments = Table(
    'sample_comments', Base.metadata,
    Column('sample_id', ForeignKey('samples.id'), primary_key=True),
    Column('comment_id', ForeignKey('comments.id'), primary_key=True))

sample_transfer_comments = Table(
    'sample_transfer_comments', Base.metadata,
    Column('transfer_id', ForeignKey('sample_transfer.id'), primary_key=True),
    Column('comment_id', ForeignKey('comments.id'), primary_key=True))

report_comments = Table(
    'report_comments', Base.metadata,
    Column('report_id', ForeignKey('reports.id'), primary_key=True),
    Column('comment_id', ForeignKey('comments.id'), primary_key=True))

record_comments = Table(
    'record_comments', Base.metadata,
    Column('record_id', ForeignKey('records.id'), primary_key=True),
    Column('comment_id', ForeignKey('comments.id'), primary_key=True))


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
    hashed_password = Column(LargeBinary(64))
    fullname = Column(String)

    roles = relationship('Role', secondary=user_roles, back_populates='users')
    attachments = relationship('Attachment', back_populates='user')
    comments = relationship('Comment', back_populates='user')

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

    comments = relationship('Comment',
                            secondary=comment_tags,
                            back_populates='tags')
    records = relationship('Record',
                           secondary=record_tags,
                           back_populates='tags')
    reports = relationship('Report',
                           secondary=report_tags,
                           back_populates='tags')
    samples = relationship('Sample',
                           secondary=sample_tags,
                           back_populates='tags')

    def __init__(self, text) -> None:
        super().__init__()
        self.text = text

    def __repr__(self):
        return f"Tag('{self.text}')"


class Comment(Base):
    __tablename__ = 'comments'

    id = Column(Integer, primary_key=True)
    text = Column(String)
    user_id = Column(Integer, ForeignKey('users.id'))
    ctime = Column(DateTime, default=datetime.utcnow)
    parent_id = Column(Integer, ForeignKey('comments.id'))

    replies = relationship("Comment", lazy="joined", join_depth=2)
    user = relationship('User', back_populates='comments')
    tags = relationship('Tag',
                        secondary=comment_tags,
                        back_populates='comments')
    attachments = relationship('Attachment', back_populates='comment')


class Attachment(Base):
    __tablename__ = 'attachments'

    id = Column(Integer, primary_key=True)
    filename = Column(String)
    mime_type = Column(String, default='application/octet-stream')
    user_id = Column(Integer, ForeignKey('users.id'))
    comment_id = Column(Integer, ForeignKey('comments.id'))
    ctime = Column(DateTime, default=datetime.utcnow)
    size = Column(Integer)
    sha1 = Column(String)
    description = Column(Text)

    user = relationship('User', back_populates='attachments')
    comment = relationship('Comment', back_populates='attachments')

    @property
    def data(self)->bytes:
        self.atime = datetime.utcnow()
        with open(self.filename, 'rb') as f:
            data = f.read()
        return data

    @data.setter
    def data(self, data: bytes):
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filename, 'wb') as f:
            f.write(data)


class InputText(Base):
    __tablename__ = 'inputs'

    id = Column(Integer, primary_key=True)
    hash = Column(LargeBinary(20))
    text_field = Column(Text, unique=True)

    @property
    def text(self):
        return self.text_field

    @text.setter
    def text(self, text):
        self.hash = hashlib.sha1(text.encode('utf-8')).digest()
        self.text_field = text

    def __repr__(self) -> str:
        return self.text


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

    def __repr__(self) -> str:
        return f"Cell(index={self.index}, input='{self.input}')"


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

    account_id = Column(Integer, ForeignKey("sample_accounts.id"))

    tags = relationship("Tag", secondary=sample_tags, back_populates="samples")
    records = relationship("Record",
                           secondary=sample_records,
                           back_populates="samples")
    reports = relationship("Report",
                           secondary=sample_reports,
                           back_populates="samples")
    transfers = relationship("SampleTransfer",
                             order_by="SampleTransfer.ctime",
                             back_populates="sample")
    account = relationship("SampleAccount", back_populates="samples")
    comments = relationship("Comment", secondary=sample_comments)


class SampleAccountType(Base):
    __tablename__ = 'sample_account_types'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(String)

    accounts = relationship("SampleAccount", back_populates="type")


class SampleAccount(Base):
    __tablename__ = 'sample_accounts'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    type_id = Column(Integer, ForeignKey("sample_account_types.id"))
    description = Column(String)

    type = relationship("SampleAccountType", back_populates="accounts")

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
    comments = relationship("Comment", secondary=sample_transfer_comments)


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

    user = relationship("User")
    samples = relationship("Sample",
                           secondary=sample_records,
                           back_populates="records")

    reports = relationship('Report',
                           secondary=record_reports,
                           back_populates='records')
    tags = relationship('Tag', secondary=record_tags, back_populates='records')
    comments = relationship('Comment', secondary=record_comments)

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

    @property
    def data(self):
        self.atime = datetime.utcnow()
        with open(self.file, 'rb') as f:
            data = pickle.load(f)
        return data

    @data.setter
    def data(self, data):
        buf = pickle.dumps(data)
        hashstr = hashlib.sha1(buf).hexdigest()
        file = Path(
            self.file
        ).parent / 'objects' / hashstr[:2] / hashstr[2:4] / hashstr[4:]
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, 'wb') as f:
            f.write(buf)
        self.file = str(file)
        return

    # def flush(self):
    #     if len(self._buff[0]) == 0:
    #         if self.df is not None and self.ds is None:
    #             self._df_to_ds()
    #         return

    #     def getitem(y, s):
    #         ret = y
    #         for i in s:
    #             ret = ret[i]
    #         return ret

    #     self.mtime = datetime.utcnow()
    #     self.atime = datetime.utcnow()
    #     if self.coords is None:
    #         index, values = self._buff
    #     else:
    #         index = []
    #         values = []
    #         shape = [len(v) for v in self.coords.values()]
    #         for i, v in zip(*self._buff):
    #             for n, c in enumerate(
    #                     itertools.product(*self.coords.values())):
    #                 s = np.unravel_index(n, shape)
    #                 index.append(tuple(i) + c)
    #                 values.append(tuple(getitem(y, s) for y in v))

    #     df = pd.DataFrame(values,
    #                       index=pd.MultiIndex.from_tuples(index,
    #                                                       names=self.dims),
    #                       columns=self.vars)

    #     if self.df is None:
    #         self.df = df
    #     else:
    #         self.df = self.df.append(df)
    #     self._df_to_ds()
    #     self._buff = ([], [])

    # def _df_to_ds(self):
    #     self.ds = xr.Dataset.from_dataframe(self.df)
    #     for units, var in zip(self.vars_units, self.vars):
    #         self.ds[var].attrs['units'] = units
    #     for dim, units in zip(self.dims, self.dims_units):
    #         self.ds[dim].attrs['units'] = units

    # def append(self, index, values):
    #     self._buff[0].extend(index)
    #     self._buff[1].extend(values)

    #     if len(self._buff[0]) > 1000:
    #         self.flush()

    # def set_values(self, *values):
    #     self.df = pd.DataFrame(dict(zip(self.columnLabels, values)))

    # def __getitem__(self, label):
    #     return self.ds[label]

    # def save(self):
    #     self.flush()
    #     buf = pickle.dumps(self.ds)
    #     hashstr = hashlib.sha1(buf).hexdigest()
    #     file = Path(self.file).parent / 'objects' / hashstr[:2] / hashstr[2:4] / hashstr[4:]
    #     file.parent.mkdir(parents=True, exist_ok=True)
    #     with open(file, 'wb') as f:
    #         f.write(buf)
    #     self.file = str(file)
    #     return

    #     if Path(self.file).exists():
    #         mode = 'a'
    #     else:
    #         mode = 'w'

    #     self.ds.to_netcdf(self.file,
    #                       group=self.key,
    #                       mode=mode,
    #                       format='NETCDF4',
    #                       engine='netcdf4')

    # def data(self):
    #     self.atime = datetime.utcnow()
    #     self.flush()
    #     if self.ds is None:
    #         with open(self.file, 'rb') as f:
    #             self.ds = pickle.load(f)
    #     return self.ds

    #     if self.ds is None:
    #         self.ds = xr.open_dataset(self.file,
    #                                   group=self.key,
    #                                   mode='r',
    #                                   engine='netcdf4')
    #     return self.ds


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
    comments = relationship('Comment', secondary=report_comments)

    @property
    def obj(self):
        self.atime = datetime.utcnow()
        with open(self.file, 'rb') as f:
            data = pickle.load(f)
        return data

    @obj.setter
    def obj(self, data):
        buf = pickle.dumps(data)
        hashstr = hashlib.sha1(buf).hexdigest()
        file = Path(
            self.file
        ).parent / 'objects' / hashstr[:2] / hashstr[2:4] / hashstr[4:]
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, 'wb') as f:
            f.write(buf)
        self.file = str(file)
        return


def create_tables(engine):
    Base.metadata.create_all(engine)

    sys_role = Role(name='sys')
    kernel = User(name='BIG BROTHER')
    kernel.roles.append(sys_role)
    
    root_role = Role(name='root')
    admin_role = Role(name='admin')
    root_user = User(name='root')
    root_user.setPassword('123')
    root_user.roles.append(root_role)
    root_user.roles.append(admin_role)

    t1 = SampleAccountType(name='factory')
    t2 = SampleAccountType(name='destroyed')
    t3 = SampleAccountType(name='storage')
    t4 = SampleAccountType(name='fridge')
    a = SampleAccount(name='destroyed')
    a.type = t2

    Session = sessionmaker(bind=engine)
    session = Session()

    session.add(root_user)
    session.add(kernel)
    session.add_all([t1, t2, t3, t4, a])
    session.commit()
