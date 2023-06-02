from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import SingletonThreadPool
from sqlalchemy.orm.session import Session


def session(url: str, debug_mode: bool = False) -> Session:
    """
    Create a database session.

    Parameters
    ----------
    url : str
        The database URL. These URLs follow RFC-1738, and usually can include username, password,
        hostname, database name as well as optional keyword arguments for additional configuration.
        In some cases a file path is accepted, and in others a “data source name” replaces the
        “host” and “database” portions. The typical form of a database URL is:
        ``dialect+driver://username:password@host:port/database``.
        e.g. ``sqlite:////absolute/path/to/foo.db``, ``sqlite:///relative/path/to/foo.db``
        or ``postgresql://scott:tiger@localhost/mydatabase``.
        
        See ``SQLAlchemy documentation <https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls>``_
        for more information.
    
    debug_mode : bool, optional
        if True, the Engine will log all statements
        as well as a ``repr()`` of their parameter lists to the default log
        handler, which defaults to ``sys.stdout`` for output.   If set to the
        string ``"debug"``, result rows will be printed to the standard output
        as well. The ``echo`` attribute of ``Engine`` can be modified at any
        time to turn logging on and off; direct control of logging is also
        available using the standard Python ``logging`` module.
    
    Returns
    -------
    :class:`sqlalchemy.orm.session.Session`
        The database session.
    """
    if url.startswith('sqlite'):
        eng = create_engine(url,
                            echo=debug_mode,
                            poolclass=SingletonThreadPool,
                            connect_args={'check_same_thread': False})
    else:
        eng = create_engine(url, echo=debug_mode)
    return sessionmaker(bind=eng)()
