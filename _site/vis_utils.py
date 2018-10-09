import numpy as np

def fd_bins(x):
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25

    binwidth = (2 * iqr / (len(x) ** (1 / 3)))
    nbins = np.ptp(x) / binwidth
    return np.round(nbins).astype(int)


def add_stds(ax, data, n=5, showmean=True):
    mean = np.mean(data)
    std = np.std(data)
    if showmean:
        ax.axvline(mean, c='r', lw=1.5)
    for i in range(n):
        if i % 2 == 0:
            ax.axvspan(mean + i * std, (mean + i * std) + std, alpha=0.6, color='lightgray')
            ax.axvspan(mean - i * std, (mean - i * std) - std, alpha=0.6, color='lightgray')


def add_labels(ax, title=None, x=None, y=None):
    if title:
        ax.set_title(title)
    if x:
        ax.set_xlabel(x)
    if y:
        ax.set_ylabel(y)


def despine(ax, which):
    if isinstance(which, str):
        which = [which]
    for spine in which:
        ax.spines[spine].set_visible(False)


def line_histogram(ax, data, bins, label, precision=None, lw=1, c=None):
    if precision:
        data = np.around(data, precision)
        bins = np.around(bins, precision)
    y, bin_edges = np.histogram(data, bins=bins, density=False, weights=np.zeros_like(data) + 1. / data.size)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    if c:
        ax.plot(bin_centers, y, '-', label=label, lw=lw, c=c)
    else:
        ax.plot(bin_centers, y, '-', label=label, lw=lw)
    ax.set_xticks(bins[:-1])
    ax.grid(axis='y', c='gray', ls='dotted')
    ax.grid(axis='x', c='gray', ls='dotted')
    despine(ax, ('top', 'right'))


