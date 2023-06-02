from datetime import date, datetime
from typing import Sequence

from sqlalchemy.orm.session import Session
from waveforms.dicttree import foldDict

from ..models import Record
from .basic import get_object_with_tags, tag


def query_record(session: Session,
                 offset: int = 0,
                 limit: int = 10,
                 app: str | None = None,
                 tags: Sequence[str] = (),
                 before: datetime | date | None = None,
                 after: datetime | date | None = None):
    local_tm = datetime.fromtimestamp(0)
    utc_tm = datetime.utcfromtimestamp(0)
    tz_offset = local_tm - utc_tm
    table = {'header': ['ID', 'App', 'tags', 'created time'], 'body': []}
    apps = sorted(
        set([
            n for n, *_ in get_object_with_tags(session.query(Record.app),
                                                Record, *tags).all()
        ]))
    apps = foldDict(dict([(app, None) for app in apps]))

    query = get_object_with_tags(session, Record, *tags)

    if app is not None:
        if app.endswith('*'):
            query = query.filter(Record.app.like(app[:-1] + '%'))
        else:
            query = query.filter(Record.app == app)
    if before is not None:
        if isinstance(before, date):
            before = datetime(before.year, before.month, before.day)
        query = query.filter(Record.ctime <= before - tz_offset)
    if after is not None:
        if isinstance(after, date):
            after = datetime(after.year, after.month, after.day)
        query = query.filter(Record.ctime >= after - tz_offset)
    total = query.count()
    for r in query.order_by(Record.ctime.desc()).limit(limit).offset(offset):
        tags = sorted([t.text for t in r.tags])
        ctime = r.ctime + tz_offset
        row = [r.id, r.app, tags, ctime]
        table['body'].append(row)

    return total, apps, table


def update_tags(session: Session,
                record_id: int,
                tags: Sequence[str],
                append: bool = False):
    record = session.get(Record, record_id)
    if record is None:
        return False
    if append:
        old = [t.text for t in record.tags]
        for t in old:
            if t not in tags:
                tags.append(t)
    record.tags = [tag(session, t) for t in tags]
    try:
        session.commit()
    except Exception:
        session.rollback()
        return False
    return True


def remove_tags(session: Session, record_id: int, tags: Sequence[str]):
    record = session.get(Record, record_id)
    if record is None:
        return False
    old = [t.text for t in record.tags]
    record.tags = [tag(session, t) for t in old if t not in tags]
    try:
        session.commit()
    except Exception:
        session.rollback()
        return False
    return True
