from typing import Literal, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.signal import correlate
from scipy.sparse import coo_matrix, diags, linalg

from ..graph import graphs, minimum_spanning_tree


def fit_relative_delay(waveform, data, sample_rate, fit=True):
    t = np.linspace(0, len(data) / sample_rate, len(data), endpoint=False)
    if isinstance(waveform, np.ndarray):
        fit = False
        corr = correlate(waveform, data, mode='same', method='fft')
    else:
        corr = correlate(waveform(t), data, mode='same', method='fft')
    delay = 0.5 * (t[0] + t[-1]) - t[np.argmax(corr)]

    if not fit:
        return delay

    def fun(delay, t, waveform, sig):
        if isinstance(delay, (int, float, complex)):
            ret = -correlate((waveform >> delay)(t), sig, mode='valid')[0]
            return ret
        else:
            return np.array([fun(d, t, waveform, sig) for d in delay])

    ret = minimize(fun, [delay], args=(t, waveform, data))
    return ret.x[0]


def calc_delays(relative_delays: dict[tuple[str, str], float],
                reference: float = 0,
                reference_channel: Optional[str] = None,
                full: bool = False) -> dict[str, float]:
    channels = []
    channels_map = {}

    def note_channel(ch):
        if ch not in channels:
            channels_map[ch] = len(channels)
            channels.append(ch)

    matrix = []
    weight = []
    absolute_error = True
    y = []
    for i, ((ch1, ch2), delay) in enumerate(relative_delays.items()):
        note_channel(ch1)
        note_channel(ch2)
        matrix.append([i, channels_map[ch1], 1])
        matrix.append([i, channels_map[ch2], -1])
        if isinstance(delay, (int, float)):
            y.append(delay)
            weight.append(1)
            absolute_error = False
        else:
            y.append(delay[0])
            weight.append(1 / delay[1]**2)

    if reference_channel is None:
        reference_channel = channels[0]
        reference = None
    note_channel(reference_channel)
    matrix.append([i + 1, channels_map[reference_channel], 1])
    y.append(0)
    weight.append(np.max(weight))

    row, col, data = list(zip(*matrix))

    X = coo_matrix((data, (row, col)), shape=(len(y), len(channels)))
    W = diags(weight, 0, shape=(len(y), len(y)))

    M = linalg.inv(X.T @ W @ X)

    beta = M @ X.T @ W @ y
    if reference is None:
        offset = -np.min(beta)
    else:
        offset = reference - beta[channels_map[reference_channel]]

    if full:
        if absolute_error:
            cov = linalg.inv(X.T @ W @ X)
        else:
            r = y - X @ beta
            chi_square = r.T @ W @ r
            dof = len(y) - len(channels)
            cov = chi_square / dof * cov
        errors = np.sqrt(cov.diagonal())
        return {
            ch: (v + offset, e)
            for ch, v, e in zip(channels, beta, errors)
        }
    else:
        return {ch: v + offset for ch, v in zip(channels, beta)}


def relative_delay_to_absolute(relative_delays: dict[tuple[str, str], float],
                               reference: float = 0,
                               reference_channel: Optional[str] = None,
                               method: Literal['mst', 'lsq'] = 'mst',
                               full: bool = False) -> dict[str, float]:
    groups = graphs(relative_delays.keys())
    graph_list = []
    if method == 'mst':
        for group in groups:
            edges = []
            for k in group:
                if isinstance(relative_delays[k], tuple):
                    edges.append((*k, relative_delays[k][1]))
                else:
                    edges.append((*k, 1))
            edges = minimum_spanning_tree(edges)
            graph_list.append({k: relative_delays[k] for k in edges})
    else:
        for group in groups:
            graph_list.append({k: relative_delays[k] for k in group})
    ret = {}
    for graph in graph_list:
        ret.update(calc_delays(graph, reference, reference_channel, full))
    return ret
