import asyncio
from collections import deque
from datetime import timedelta
from math import ceil
from time import monotonic

from blinker import Signal

try:
    import ipywidgets as widgets
    from IPython.display import display
except:
    pass


class Progress:
    sma_window = 10  # Simple Moving Average window

    def __init__(self, *, max=10):
        self.start_ts = monotonic()
        self.finished_ts = None
        self.avg = 0
        self._avg_update_ts = self.start_ts
        self._ts = self.start_ts
        self._xput = deque(maxlen=self.sma_window)
        self.max = max
        self.index = 0
        self.updated = Signal()
        self.finished = Signal()

    @property
    def eta(self):
        return int(
            ceil(self.avg * self.remaining -
                 (monotonic() - self._avg_update_ts)))

    @property
    def eta_td(self):
        return timedelta(seconds=self.eta)

    @property
    def percent(self):
        return self.progress * 100

    @property
    def progress(self):
        return min(1, self.index / self.max)

    @property
    def remaining(self):
        return max(self.max - self.index, 0)

    @property
    def elapsed(self):
        if self.finished_ts is not None:
            return int(self.finished_ts - self.start_ts)
        else:
            return int(monotonic() - self.start_ts)

    @property
    def elapsed_td(self):
        return timedelta(seconds=self.elapsed)

    def update_avg(self, n, dt):
        if n > 0:
            xput_len = len(self._xput)
            self._xput.append(dt / n)
            now = monotonic()
            if (xput_len < self.sma_window or now - self._avg_update_ts > 1):
                self.avg = sum(self._xput) / len(self._xput)
                self._avg_update_ts = now

    def next(self, n=1):
        now = monotonic()
        dt = now - self._ts
        self.update_avg(n, dt)
        self._ts = now
        self.index = self.index + n
        self.updated.send(self)

    def goto(self, index):
        incr = index - self.index
        self.next(incr)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.finish(True)
        else:
            self.finish(False)

    def iter(self, it, max=None):
        if max is None:
            try:
                self.max = len(it)
            except TypeError:
                self.max = 0
        else:
            self.max = max

        with self:
            for x in it:
                yield x
                self.next()

    def __repr__(self):
        if self._avg_update_ts == self.start_ts or self.finished_ts is not None:
            eta_td = '--:--:--'
        else:
            eta_td = self.eta_td
        return (f"({self.index}/{self.max}) {self.percent:.0f}%"
                f" Used time: {self.elapsed_td} Remaining time: {eta_td}")

    def start(self):
        self.index = 0
        self.start_ts = monotonic()
        self.avg = 0
        self._avg_update_ts = self.start_ts
        self._ts = self.start_ts

    def finish(self, success=True):
        self.finished_ts = monotonic()
        self.finished.send(self, success=success)


class ProgressBar():

    def listen(self, progress: Progress):
        self.progress = progress
        self.progress.updated.connect(self.update)
        self.progress.finished.connect(self.finish)

    def update(self, sender):
        raise NotImplementedError()

    def finish(self, sender, success):
        self.progress.updated.disconnect(self.update)
        self.progress.finished.disconnect(self.finish)
        try:
            self._update_loop.cancel()
        except:
            pass

    def update_regularly(self, frequency=1):
        self.progress.updated.send(self.progress)
        self._update_loop = asyncio.get_running_loop().call_later(
            frequency, self.update_regularly)


class JupyterProgressBar(ProgressBar):

    def __init__(self, *, description='Progressing', hiden=False):
        self.description = description
        self.hiden = hiden

    def display(self):
        if self.hiden:
            return
        self.progress_ui = widgets.IntProgress(value=0,
                                               min=0,
                                               max=self.progress.max,
                                               step=1,
                                               description='',
                                               bar_style='')

        self.elapsed_ui = widgets.Label(value='Used time: 00:00:00')
        self.eta_ui = widgets.Label(value='Remaining time: --:--:--')
        self.ui = widgets.HBox(
            [widgets.HTML(f'<em>{self.description}</em>'),self.progress_ui, self.elapsed_ui, self.eta_ui])
        display(self.ui)
        self.update_regularly()

    def update(self, sender):
        if self.hiden:
            return
        self.progress_ui.value = sender.index
        self.progress_ui.max = sender.max
        self.elapsed_ui.value = f'({sender.index}/{sender.max}) Used time: {sender.elapsed_td}'
        if sender.eta == sender.start_ts:
            self.eta_ui.value = f'Remaining time: --:--:--'
        else:
            self.eta_ui.value = f'Remaining time: {sender.eta_td}'

    def finish(self, sender, success=True):
        if self.hiden:
            return
        if success:
            self.progress_ui.bar_style = 'success'
            self.progress_ui.value = self.progress_ui.max
        else:
            self.progress_ui.bar_style = 'danger'
        super().finish(sender, success)
