import contextlib
import time
from datetime import datetime

__current_notebook = None
__session = None


def create_cell(session, notebook, cell):
    raise NotImplementedError


def create_notebook(session, ctime):
    raise NotImplementedError


def get_current_notebook(session=None):
    global __current_notebook
    if __current_notebook is None:
        try:
            if session is None:
                with current_session() as session:
                    __current_notebook = create_notebook(
                        session, time.asctime())
            else:
                __current_notebook = create_notebook(session, time.asctime())
        except RuntimeError:
            return None
    return __current_notebook


def get_inputCells():
    try:
        from IPython import get_ipython

        return get_ipython().user_ns['In']
    except Exception as e:
        return ['']


def save_inputCells(session):
    notebook = get_current_notebook(session)
    aready_saved = len(notebook.cells)
    for cell in get_inputCells()[aready_saved:]:
        create_cell(session, notebook, cell)


def update_inputCells(session):
    notebook = get_current_notebook(session)
    if notebook.cells:
        notebook.cells[-1].ftime = datetime.utcnow()
    session.add(notebook)


__sessionmaker = None


@contextlib.contextmanager
def current_session():
    global __session
    if __sessionmaker is None:
        raise RuntimeError('Sessionmaker is not set')
    if __session is None:
        __session = __sessionmaker()
    try:
        yield __session
        __session.commit()
    except:
        __session.rollback()


def set_sessionmaker(sessionmaker):
    global __sessionmaker
    __sessionmaker = sessionmaker
    __session = __sessionmaker()


def _autosave_cells(ipython):
    try:
        with current_session() as session:
            save_inputCells(session)
    except:
        pass


def _update_timestamp(ipython):
    try:
        with current_session() as session:
            update_inputCells(session)
    except:
        pass


def setup_ipy_events():
    try:
        from IPython import get_ipython

        ipython = get_ipython()

        if ipython is not None:
            ipython.events.register('pre_run_cell', _autosave_cells)
            ipython.events.register('post_run_cell', _update_timestamp)
    except:
        pass
