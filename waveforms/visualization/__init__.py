import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter


def plotLine(c0, c1, ax, **kwargs):
    t = np.linspace(0, 1, 11)
    c = (c1 - c0) * t + c0
    ax.plot(c.real, c.imag, **kwargs)


def plotCircle(c0, r, ax, **kwargs):
    t = np.linspace(0, 1, 1001) * 2 * np.pi
    s = c0 + r * np.exp(1j * t)
    ax.plot(s.real, s.imag, **kwargs)


def plotEllipse(c0, a, b, phi, ax, **kwargs):
    t = np.linspace(0, 1, 1001) * 2 * np.pi
    c = np.exp(1j * t)
    s = c0 + (c.real * a + 1j * c.imag * b) * np.exp(1j * phi)
    ax.plot(s.real, s.imag, **kwargs)


def plotDistribution(s0, s1, fig=None, axes=None, info=None, hotThresh=10000):
    from waveforms.math.fit import get_threshold_info, mult_gaussian_pdf

    if info is None:
        info = get_threshold_info(s0, s1)
    thr, phi = info['threshold'], info['phi']
    visibility, p0, p1 = info['visibility']
    # print(
    #     f"thr={thr:.6f}, phi={phi:.6f}, visibility={visibility:.3f}, {p0}, {1-p1}"
    # )

    if axes is not None:
        ax1, ax2 = axes
    else:
        if fig is None:
            fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

    if (len(s0) + len(s1)) < hotThresh:
        ax1.plot(np.real(s0), np.imag(s0), '.', alpha=0.2)
        ax1.plot(np.real(s1), np.imag(s1), '.', alpha=0.2)
    else:
        _, *bins = np.histogram2d(np.real(np.hstack([s0, s1])),
                                  np.imag(np.hstack([s0, s1])),
                                  bins=50)

        H0, *_ = np.histogram2d(np.real(s0),
                                np.imag(s0),
                                bins=bins,
                                density=True)
        H1, *_ = np.histogram2d(np.real(s1),
                                np.imag(s1),
                                bins=bins,
                                density=True)
        vlim = max(np.max(np.abs(H0)), np.max(np.abs(H1)))

        ax1.imshow(H1.T - H0.T,
                   alpha=(np.fmax(H0.T, H1.T) / vlim).clip(0, 1),
                   interpolation='nearest',
                   origin='lower',
                   cmap='coolwarm',
                   vmin=-vlim,
                   vmax=vlim,
                   extent=(bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]))

    ax1.axis('equal')
    ax1.set_xticks([])
    ax1.set_yticks([])
    for s in ax1.spines.values():
        s.set_visible(False)

    # c0, c1 = info['center']
    # a0, b0, a1, b1 = info['std']
    params = info['params']
    r0, i0, r1, i1 = params[0][0], params[1][0], params[0][1], params[1][1]
    a0, b0, a1, b1 = params[0][2], params[1][2], params[0][3], params[1][3]
    c0 = (r0 + 1j * i0) * np.exp(1j * phi)
    c1 = (r1 + 1j * i1) * np.exp(1j * phi)
    plotEllipse(c0, 2 * a0, 2 * b0, phi, ax1)
    plotEllipse(c1, 2 * a1, 2 * b1, phi, ax1)

    im0, im1 = info['idle']
    lim = min(im0.min(), im1.min()), max(im0.max(), im1.max())
    t = (np.linspace(lim[0], lim[1], 3) + 1j * thr) * np.exp(-1j * phi)
    ax1.plot(t.imag, t.real, 'k--')

    ax1.plot(np.real(c0), np.imag(c0), 'o', color='C3')
    ax1.plot(np.real(c1), np.imag(c1), 'o', color='C4')

    re0, re1 = info['signal']
    x, a, b, c = info['cdf']

    n0, bins0, *_ = ax2.hist(re0, bins=50, alpha=0.5)
    n1, bins1, *_ = ax2.hist(re1, bins=50, alpha=0.5)
    ax2.plot(
        x,
        np.sum(n0) * (bins0[1] - bins0[0]) * mult_gaussian_pdf(
            x, [r0, r1], [a0, a1], [params[0][4], 1 - params[0][4]]))
    ax2.plot(
        x,
        np.sum(n1) * (bins1[1] - bins1[0]) * mult_gaussian_pdf(
            x, [r0, r1], [a0, a1], [params[0][5], 1 - params[0][5]]))
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Projection Axes')

    ax3 = ax2.twinx()
    ax3.plot(x, a)
    ax3.plot(x, b)
    ax3.plot(x, c)
    ax3.set_ylim(0, 1.1)
    ax3.vlines(thr, 0, 1.1, 'k')
    ax3.set_ylabel('Probility')

    return info


ALLXYSeq = [('I', 'I'), ('X', 'X'), ('Y', 'Y'), ('X', 'Y'), ('Y', 'X'),
            ('X/2', 'I'), ('Y/2', 'I'), ('X/2', 'Y/2'), ('Y/2', 'X/2'),
            ('X/2', 'Y'), ('Y/2', 'X'), ('X', 'Y/2'), ('Y', 'X/2'),
            ('X/2', 'X'), ('X', 'X/2'), ('Y/2', 'Y'), ('Y', 'Y/2'), ('X', 'I'),
            ('Y', 'I'), ('X/2', 'X/2'), ('Y/2', 'Y/2')]


def plotALLXY(data, ax=None):
    assert len(data) % len(ALLXYSeq) == 0

    if ax is None:
        ax = plt.gca()

    ax.plot(np.array(data), 'o-')
    repeat = len(data) // len(ALLXYSeq)
    ax.set_xticks(np.arange(len(ALLXYSeq)) * repeat + 0.5 * (repeat - 1))
    ax.set_xticklabels([','.join(seq) for seq in ALLXYSeq], rotation=60)
    ax.grid(which='major')


def _get_log_ticks(x):
    log10x = np.log10(x)

    major_ticks = np.array(
        range(math.floor(log10x[0]) - 1,
              math.ceil(log10x[-1]) + 1))
    minor_ticks = np.hstack([
        np.log10(np.linspace(2, 10, 9, endpoint=False)) + x
        for x in major_ticks
    ])

    major_ticks = major_ticks[(major_ticks >= log10x[0]) *
                              (major_ticks <= log10x[-1])]
    minor_ticks = minor_ticks[(minor_ticks >= log10x[0]) *
                              (minor_ticks <= log10x[-1])]

    return log10x, major_ticks, minor_ticks


def imshow_logx(x, y, z, x_unit='s', ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    log10x, major_ticks, minor_ticks = _get_log_ticks(x)

    dlogx, dy = log10x[1] - log10x[0], y[1] - y[0]
    extent = (log10x[0] - dlogx / 2, log10x[-1] + dlogx / 2, y[0] - dy / 2,
              y[-1] + dy / 2)

    img = ax.imshow(z, extent=extent, **kwargs)

    ax.set_xticks(major_ticks, minor=False)
    formater = EngFormatter(unit=x_unit)
    ax.set_xticklabels([formater.format_data(10.0**i) for i in major_ticks])

    ax.set_xticks(minor_ticks, minor=True)

    return img


def imshow_logy(x, y, z, y_unit='s', ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    log10y, major_ticks, minor_ticks = _get_log_ticks(y)

    dlogy, dx = log10y[1] - log10y[0], x[1] - x[0]
    extent = (x[0] - dx / 2, x[-1] + dx / 2, log10y[0] - dlogy / 2,
              log10y[-1] + dlogy / 2)

    img = ax.imshow(z, extent=extent, **kwargs)

    ax.set_yticks(major_ticks, minor=False)
    formater = EngFormatter(unit=y_unit)
    ax.set_yticklabels([formater.format_data(10.0**i) for i in major_ticks])
    ax.set_yticks(minor_ticks, minor=True)

    return img


def imshow_loglog(x, y, z, x_unit='s', y_unit='Hz', ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    log10x, x_major_ticks, x_minor_ticks = _get_log_ticks(x)
    log10y, y_major_ticks, y_minor_ticks = _get_log_ticks(y)

    dlogx, dlogy = log10x[1] - log10x[0], log10y[1] - log10y[0]
    extent = (log10x[0] - dlogx / 2, log10x[-1] + dlogx / 2,
              log10y[0] - dlogy / 2, log10y[-1] + dlogy / 2)

    img = ax.imshow(z, extent=extent, **kwargs)

    ax.set_xticks(x_major_ticks, minor=False)
    formater = EngFormatter(unit=x_unit)
    ax.set_xticklabels([formater.format_data(10.0**i) for i in x_major_ticks])
    ax.set_xticks(x_minor_ticks, minor=True)

    ax.set_yticks(y_major_ticks, minor=False)
    formater = EngFormatter(unit=y_unit)
    ax.set_yticklabels([formater.format_data(10.0**i) for i in y_major_ticks])
    ax.set_yticks(y_minor_ticks, minor=True)

    return img
