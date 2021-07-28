from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import NoResultFound

from .models import (Cell, InputText, InvalidKey, Notebook, Record, Report,
                     Role, Sample, SampleAccount, SampleTransfer, Tag, User)


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


def get_tag(session: Session, tag_text: str) -> Tag:
    """Get a tag from the database or create a new if not exists."""
    try:
        return session.query(Tag).filter(Tag.text == tag_text).one()
    except NoResultFound:
        return create_tag(session, tag_text)


def create_tag(session: Session, tag_text: str) -> Tag:
    """Create a tag in the database."""
    tag = Tag(text=tag_text)
    session.add(tag)
    return tag


def create_tag(session: Session, tag_text: str) -> Tag:
    """Create a tag in the database."""
    tag = Tag(text=tag_text)
    session.add(tag)
    return tag


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


def create_user(session: Session, user_name: str, password: str) -> User:
    """Create a user in the database."""
    user = User(name=user_name, password=password)
    session.add(user)
    return user


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
    cell.index = len(notebook.cells)
    session.add(cell)
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


def transform_sample(session: Session, sample: Sample, source: SampleAccount,
                     dest: SampleAccount) -> None:
    """Transform a sample from one account to another."""
    if sample.account_id != source.id:
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
    