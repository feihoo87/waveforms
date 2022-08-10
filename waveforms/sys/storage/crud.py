from typing import Type, Union
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.orm import aliased, Query
from .models import (Cell, Comment, InputText, InvalidKey, Notebook, Record,
                     Report, Role, Sample, SampleAccount, SampleTransfer, Tag,
                     User)


def create_user(session: Session, user_name: str, password: str) -> User:
    """Create a user in the database."""
    user = User(name=user_name)
    user.setPassword(password)
    session.add(user)
    return user


def login(session: Session, user_name: str, password: str) -> User:
    """Login a user in the database."""
    try:
        user = session.query(User).filter(User.name == user_name).one()
    except NoResultFound:
        return None
    try:
        user.verify(password)
    except InvalidKey:
        return None
    return user


def get_all_records(session: Session) -> list:
    """Get all records from the database."""
    return session.query(Record).all()


def get_all_samples(session: Session) -> list:
    """Get all samples from the database."""
    return session.query(Sample).all()


def get_all_tags(session: Session) -> list:
    """Get all tags from the database."""
    return session.query(Tag).all()


def get_all_users(session: Session) -> list:
    """Get all users from the database."""
    return session.query(User).all()


def get_all_roles(session: Session) -> list:
    """Get all roles from the database."""
    return session.query(Role).all()


def get_all_notebooks(session: Session) -> list:
    """Get all notebooks from the database."""
    return session.query(Notebook).all()


def get_all_reports(session: Session) -> list:
    """Get all reports from the database."""
    return session.query(Report).all()


def tag(session: Session, tag_text: str) -> Tag:
    """Get a tag from the database or create a new if not exists."""
    try:
        return session.query(Tag).filter(Tag.text == tag_text).one()
    except NoResultFound:
        tag = Tag(text=tag_text)
        return tag


def tag_it(session: Session, tag_text: str, obj: Union[Sample, Record,
                                                      Report]) -> Tag:
    """Tag an object."""
    if obj.id is None:
        session.add(obj)
        obj.tags.append(tag(session, tag_text))
    else:
        session.query(type(obj)).filter(
            type(obj).id == obj.id).one().tags.append(tag(session, tag_text))
    session.commit()


def get_object_with_tags(session: Session,
                         cls: Union[Type[Comment], Type[Sample], Type[Record],
                                    Type[Report]], *tags: str) -> Query:
    """
    Query objects with the given tags.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The database session.
    cls : :class:`sqlalchemy.orm.Mapper`
        The object class.
    tags : str
        The tags.

    Returns
    -------
    :class:`sqlalchemy.orm.Query`
        The query.
    """
    if isinstance(session, Query):
        q = session
    else:
        q = session.query(cls)
    if not hasattr(cls, 'tags'):
        return []

    aliase = {tag: aliased(Tag) for tag in tags}

    for tag, a in aliase.items():
        q = q.join(a, cls.tags)
        q = q.filter(a.text == tag)
    return q


def create_notebook(session: Session, notebook_name: str) -> Notebook:
    """Create a notebook in the database."""
    notebook = Notebook(name=notebook_name)
    session.add(notebook)
    return notebook


def create_sample(session: Session, sample_name: str, notebook_id: int,
                  record_id: int) -> Sample:
    """Create a sample in the database."""
    sample = Sample(name=sample_name,
                    notebook_id=notebook_id,
                    record_id=record_id)
    session.add(sample)
    return sample


def create_record(session: Session, record_name: str, notebook_id: int,
                  sample_id: int) -> Record:
    """Create a record in the database."""
    record = Record(name=record_name,
                    notebook_id=notebook_id,
                    sample_id=sample_id)
    session.add(record)
    return record


def create_role(session: Session, role_name: str) -> Role:
    """Create a role in the database."""
    role = Role(name=role_name)
    session.add(role)
    return role


def create_cell(session: Session, notebook: Notebook, input_text: str) -> Cell:
    """Create a cell in the database."""
    cell = Cell()
    cell.notebook = notebook
    cell.input = create_input_text(session, input_text)
    cell.index = len(notebook.cells) - 1
    session.add(cell)
    notebook.atime = cell.ctime
    return cell


def create_input_text(session: Session, input_text: str) -> InputText:
    """Create an input text in the database."""
    input = InputText()
    input.text = input_text
    try:
        input = session.query(InputText).filter(
            InputText.hash == input.hash,
            InputText.text_field == input_text).one()
    except NoResultFound:
        session.add(input)
    return input


def create_sample_account(session: Session,
                          account_name: str) -> SampleAccount:
    """Create a sample account in the database."""
    account = SampleAccount(name=account_name)
    session.add(account)
    return account


def transform_sample(session: Session, sample: Sample, source: SampleAccount,
                     dest: SampleAccount) -> None:
    """Transform a sample from one account to another."""
    if (sample.account_id is None and
            source.type.name != 'factory') or sample.account_id != source.id:
        raise ValueError('Sample is not in the source account.')
    try:
        sample.account = dest
        transfer = SampleTransfer()
        transfer.sample = sample
        transfer.debtor = source
        transfer.creditor = dest
        session.add(transfer)
        session.commit()
        return transfer
    except:
        session.rollback()
        raise
