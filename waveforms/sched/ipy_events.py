import time
import datetime

from waveforms.storage.crud import create_cell, create_notebook
import threading

__current_notebook = None


def get_current_notebook():
    global __current_notebook
    if __current_notebook is None:
        try:
            session = get_current_session()
            __current_notebook = create_notebook(session, time.asctime())
            session.commit()
            session.close()
        except RuntimeError:
            return None
    return __current_notebook


def get_inputCells():
    try:
        from IPython import get_ipython
        
        return get_ipython().user_ns['In']
    except:
        return ['']


def save_inputCells(session):
    try:
        notebook = get_current_notebook()
        aready_saved = len(notebook.cells)
        for cell in get_inputCells()[aready_saved:]:
            create_cell(session, notebook, cell)
    except:
        session.rollback()
    session.commit()


def update_inputCells(session):
    try:
        notebook = get_current_notebook()
        notebook.cells[-1].ftime = datetime.utcnow()
    except:
        session.rollback()
    session.commit()


__session_pool = {}
__sessionmaker = None


def get_current_session():
    if __sessionmaker is None:
        raise RuntimeError('Sessionmaker is not set')
    tid = threading.current_thread().ident
    if tid not in __session_pool:
        __session_pool[tid] = __sessionmaker()
    return __session_pool[tid]


def set_sessionmaker(sessionmaker):
    global __sessionmaker
    __sessionmaker = sessionmaker


def _autosave_cells(ipython):
    try:
        save_inputCells(get_current_session())
    except:
        pass


def _update_timestamp(ipython):
    try:
        update_inputCells(get_current_session())
    except:
        pass


def setup():
    try:
        from IPython import get_ipython

        ipython = get_ipython()

        if ipython is not None:
            ipython.events.register('pre_run_cell', _autosave_cells)
            ipython.events.register('post_run_cell', _update_timestamp)
    except:
        pass
