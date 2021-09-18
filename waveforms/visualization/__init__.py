import matplotlib.pyplot as plt
import numpy as np


def plotLine(c0, c1, ax):
    t = np.linspace(0, 1, 11)
    c = (c1 - c0) * t + c0
    ax.plot(c.real, c.imag)


def plotCircle(c0, r, ax):
    t = np.linspace(0, 1, 1001) * 2 * np.pi
    s = c0 + r * np.exp(1j * t)
    ax.plot(s.real, s.imag)


def plotEllipse(c0, a, b, phi, ax):
    t = np.linspace(0, 1, 1001) * 2 * np.pi
    c = np.exp(1j * t)
    s = c0 + (c.real * a + 1j * c.imag * b) * np.exp(1j * phi)
    ax.plot(s.real, s.imag)


def plotDistribution(s0, s1, fig=None, axes=None, info=None, hotThresh=10000):
    from waveforms.math.fit import getThresholdInfo

    if info is None:
        info = getThresholdInfo(s0, s1)
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

    c0, c1 = info['center']
    a0, b0, a1, b1 = info['std']
    plotEllipse(c0, 2 * a0, 2 * b0, phi, ax1)
    plotEllipse(c1, 2 * a1, 2 * b1, phi, ax1)

    im0, im1 = info['idle']
    lim = min(im0.min(), im1.min()), max(im0.max(), im1.max())
    t = (np.linspace(lim[0], lim[1], 3) + 1j * thr) * np.exp(-1j * phi)
    ax1.plot(t.imag, t.real, 'k--')

    ax1.plot(np.real(c0), np.imag(c0), 'o', color='C3')
    ax1.plot(np.real(c1), np.imag(c1), 'o', color='C4')

    re0, re1 = info['signal']
    ax2.hist(re0, bins=50, alpha=0.5)
    ax2.hist(re1, bins=50, alpha=0.5)
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Projection Axes')

    x, a, b, c = info['cdf']
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