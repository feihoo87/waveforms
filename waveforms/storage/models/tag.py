from sqlalchemy import Column, ForeignKey, Integer, String, Table
from sqlalchemy.orm import relationship

from . import Base


class Tag(Base):
    __tablename__ = 'tags'

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)

    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'Tag({self.name!r})'


def has_tags(cls: Base) -> Base:
    table = Table(
        f'{cls.__tablename__}_tags', Base.metadata,
        Column('item_id',
               ForeignKey(f'{cls.__tablename__}.id'),
               primary_key=True),
        Column('tag_id', ForeignKey('tags.id'), primary_key=True))

    cls.tags = relationship("Tag", secondary=table, backref=cls.__tablename__)

    def add_tag(self, tag: Tag):
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: Tag):
        if tag in self.tags:
            self.tags.remove(tag)

    cls.add_tag = add_tag
    cls.remove_tag = remove_tag

    return cls
