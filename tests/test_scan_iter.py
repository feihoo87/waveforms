import numpy as np
from waveforms.scan_iter import *


def test_scan_iter():

    def f(a):
        for i in range(5):
            yield a + i, i

    def f2():
        for i in range(2):
            yield i

    def f3():
        for i in range(2):
            yield i * 100, -i * 200

    steps = scan_iters(
        {
            'a': [-1, 1],
            ('b', ('c', 'd')): ([4, 5, 6], f),
            (('e', ), ('g', 'h')): (f2, f3)
        },
        filter=lambda x: x < 8,
        additional_kwds={
            'x': lambda a, b: a + b
        })

    def scan_iter2():
        for a in [-1, 1]:
            for b, (c, d) in zip([4, 5, 6], f(a)):
                for e, (g, h) in zip(f2(), f3()):
                    x = a + b
                    if x < 8:
                        yield {
                            'a': a,
                            'b': b,
                            'c': c,
                            'd': d,
                            'e': e,
                            'g': g,
                            'h': h,
                            'x': x
                        }

    for step, args in zip(steps, scan_iter2()):
        for k in ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'x']:
            assert k in step.kwds
            assert step.kwds[k] == args[k]


def test_storage():

    class FindPeak(BaseOptimizer):

        def __init__(self, dimensions):
            self.index = -1
            self.max_args = dimensions[0]
            self.max_value = -np.inf

        def tell(self, suggested, value):
            args, value = value
            if self.max_value <= value:
                self.max_value = value
                self.max_args = args

        def ask(self):
            self.index += 1
            self.max_value = -np.inf
            return self.max_args,

        def get_result(self):

            class Result():
                x = self.max_args,

            return Result()

    def f(b, f):
        c = 1 - b**2
        return np.exp(-((f - c) / 0.01)**2)

    def filt(freq, center=None):
        if center is None:
            return True
        else:
            return -0.1 <= freq - center <= 0.1

    z = np.full((101, 121), np.nan)
    center = None
    bias_list = np.linspace(-0.1, 0.1, 101)
    freq_list = np.linspace(-0.1, 1.1, 121)

    for i, bias in enumerate(bias_list):
        for j, freq in enumerate(freq_list):
            if filt(freq, center):
                z[i, j] = f(bias, freq)
        center = freq_list[np.argmin(np.abs(freq_list - (1 - bias**2)))]

    data = Storage(save_kwds=False)

    for step in scan_iters(
        {
            ('bias', 'center'):
            (bias_list, OptimizerConfig(FindPeak, [None], max_iters=101)),
            'freq':
            freq_list,
        },
            filter=filt,
            trackers=[data]):
        y = f(step.kwds['bias'], step.kwds['freq'])

        step.feed({'z': y}, store=True)
        step.feedback(('center', ), (step.kwds['freq'], y))

    assert set(data.keys()) == {'bias', 'freq', 'z'}
    assert np.all(bias_list == data['bias'])
    assert np.all(freq_list == data['freq'])
    assert data['z'].shape == (101, 121)
    assert np.all((z == data['z'])[np.isnan(z) == False])


def test_level_marker():
    iters = {'a': range(2), 'b': range(2), 'c': range(2)}

    def scan_iter2():
        for a in range(2):
            yield {'a': a}, 0, 'begin'
            for b in range(2):
                yield {'a': a, 'b': b}, 1, 'begin'
                for c in range(2):
                    yield {'a': a, 'b': b, 'c': c}, 2, 'begin'
                    yield {'a': a, 'b': b, 'c': c}, 2, 'step'
                    yield {'a': a, 'b': b, 'c': c}, 2, 'end'
                yield {'a': a, 'b': b}, 1, 'end'
            yield {'a': a}, 0, 'end'

    for step, args in zip(scan_iters(iters, level_marker=True), scan_iter2()):
        kw, level, marker = args
        assert step.kwds == kw
        if marker == 'begin':
            assert isinstance(step, Begin)
            assert step.level == level
        elif marker == 'step':
            assert isinstance(step, StepStatus)
        elif marker == 'end':
            assert isinstance(step, End)
            assert step.level == level
