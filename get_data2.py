import numpy as np
import os
import pandas as pd
from sklearn import linear_model
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
from scipy.special import comb
import statsmodels.api as sm
import loc_utils as lut
from standards import *
import matplotlib.pyplot as plt

rx = RAWXix()
r = RAWix()
gcolors = ['#008fd5', '#fc4f30', '#e5ae38', '#6d904f']
colors = ['#375e97', '#f18d9e', '#ffbb00', '#3f681c']
glabels = {0: 'F', 1: 'S'}
clabels = {0: 'i-', 1: 'i+'}


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    see also:
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def gclabel(g, c):
    return '{}/{}'.format(glabels[g], clabels[c])


tlabels = {
    1: '1D',
    2: 'I1D',
    3: '2D',
    4: 'R'}

saveloc = '/Users/alexten/Projects/HFSP/img'
desktop = '/Users/alexten/Desktop/'

data_path = 'pipeline_data/s3/joint_data.pkl'


def save_it(fig, savedir, figname, save_as='svg', dpi=500, compress=True):
    s = savedir + '/{}.{}'.format(figname, save_as)
    fig.savefig(s, format=save_as, dpi=500)
    if compress:
        os.system('scour -i {} -o {}'.format(s, s.replace('img', 'img_compressed')))


def onehot(ind):
    ind = ind - 1
    onehots = np.zeros([ind.size, 4])
    onehots[np.arange(ind.size), ind] = 1
    return onehots


def run_MNlog(x, y, pivot, fullout=0):
    y = y.replace(to_replace=pivot, value=0, inplace=False)
    mdl = sm.MNLogit(y, x)
    mdl_fit = mdl.fit(maxiter=100, full_output=fullout)
    return mdl_fit


def run_skl_MNlog(x, y, pivot):
    # y = y.replace(to_replace=pivot, value=0, inplace=False)
    mdl = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
    mdl_fit = mdl.fit(x, y)
    return mdl_fit


def softmax(x, stable=0):
    """Compute softmax values for each sets of scores in x."""
    if stable:
        e_x = np.exp(x - np.max(x, axis=0))
        return e_x / e_x.sum(axis=0)
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def self_challenging_data(path, alpha=.01, null=.5, rpc_window=5, save_as=False):
    def p_val(n, k, p):
        return comb(n, k) * p ** k * (1 - p) ** (n - k)

    mdata = lut.unpickle(path)['main']
    sids, groups, tasks = lut.get_unique(mdata, [r.ix('sid'), r.ix('group'), r.ix('cat')])

    nontest = lut.get_mask(mdata, {r.ix('stage'): 2}, '!=')
    mdata = mdata[nontest, :]

    outdata = []
    cols = 'sid,grp,stage,trial,blkt,current,nxt,tord:1,tord:2,tord:3,tord:4,pct:1,pct:2,pct:3,pct:4,pc:1,pc:2,pc:3,pc:4,pcr:1,pcr:2,pcr:3,pcr:4,pval:1,pval:2,pval:3,pval:4,sc:1,sc:2,sc:3,sc:4,sw_pred,sw_act,cor'.split(',')
    ix = cols.index

    for grp in groups:
        gmask = lut.get_mask(mdata, {r.ix('group'): grp})
        gsids = lut.get_unique(mdata[gmask, :], r.ix('sid'))

        for sid in gsids:
            sdata = mdata[lut.get_mask(mdata, {r.ix('sid'): sid}), :]
            sid_data = np.zeros([sdata.shape[0] - 1, len(cols)])

            # fill in task on next trial
            sid_data[:, ix('nxt')] = sdata[:, r.ix('cat')][1:]

            # discard the last trial completely
            sdata = sdata[:-1, :]

            # fill in group, sid, stage, and actual switches
            sid_data[:, ix('sid')] = sdata[:, r.ix('sid')]
            sid_data[:, ix('grp')] = grp
            sid_data[:, ix('stage')] = sdata[:, r.ix('stage')]
            sid_data[:, ix('cor')] = sdata[:, r.ix('cor')]
            sid_data[:-1, ix('sw_act')] = sdata[:, r.ix('switch')][1:]
            sid_data[:60, ix('sw_act')] = 0

            # fill in trial numbers and block trial numbers
            sid_data[:, ix('trial')] = sdata[:, r.ix('trial')]
            sid_data[:, ix('blkt')] = sdata[:, r.ix('blkt')] + 1

            # fill in task on current trial
            sid_data[:, ix('current')] = sdata[:, r.ix('cat')]

            ordmap = {0: 1, 15: 2, 30: 3, 45: 4}
            for ti, tsk in enumerate(tasks):
                train_mask = lut.get_mask(sdata, {r.ix('stage'): 0})
                tmask = lut.get_mask(sdata, {r.ix('cat'): tsk})

                train_pc = np.mean(sdata[train_mask & tmask, r.ix('cor')])
                sid_data[:, ix('pct:{}'.format(tsk))] = train_pc

                ordnum = ordmap[np.argmax(tmask)]
                sid_data[:, ix('tord:{}'.format(tsk))] = ordnum

                trials_so_far = np.cumsum(tmask)
                cor_on_task = np.zeros(sid_data.shape[0])
                cor_on_task[tmask] = sdata[tmask, r.ix('cor')]
                cor_so_far = np.cumsum(cor_on_task)

                with np.errstate(divide='ignore', invalid='ignore'):
                    sid_data[:, ix('pc:{}'.format(tsk))] = cor_so_far / trials_so_far
                    sid_data[:, ix('pval:{}'.format(tsk))] = p_val(
                        trials_so_far, cor_so_far, null)

                tempcont = np.zeros(tmask.sum())
                inds = np.append(np.where(sid_data[tmask, ix('blkt')] == 1)[0], -1)
                switch_inds = []
                switch_vals = []
                for beg, end in zip(inds[:-1], inds[1:]):
                    if end == -1:
                        blk_cor = sid_data[tmask, ix('cor')][beg:]
                        rolling_pc = pd.rolling_mean(blk_cor, window=rpc_window, min_periods=1, center=False)
                        tempcont[beg:] = rolling_pc.squeeze()
                    else:
                        blk_cor = sid_data[tmask, ix('cor')][beg:end]
                        rolling_pc = pd.rolling_mean(blk_cor, window=rpc_window, min_periods=1, center=False)
                        tempcont[beg:end] = rolling_pc.squeeze()
                    switch_inds.append(sid_data[tmask, ix('trial')][end if end < 0 else end - 1].astype(int))
                    switch_vals.append(tempcont[[end if end < 0 else end - 1]])
                sid_data[tmask, ix('pcr:{}'.format(tsk))] = tempcont

                nottmask = ~tmask
                tempcont = np.empty(nottmask.sum())
                tempcont[:] = np.nan

                for swt, val in zip(switch_inds, switch_vals):
                    mask = sid_data[nottmask, ix('trial')] > swt
                    tempcont[mask] = val

                sid_data[nottmask, ix('pcr:{}'.format(tsk))] = tempcont
            sid_data = sid_data[58:]

            pcs = sid_data[:, ix('pc:1'):ix('pc:4') + 1].copy()
            pvals = sid_data[:, ix('pval:1'):ix('pval:4') + 1]

            t1 = sid_data[:, ix('nxt')].astype(int) - 1
            t0 = sid_data[:, ix('current')].astype(int) - 1

            actual_next_choice = np.zeros([sid_data.shape[0], 4]).astype(bool)
            actual_next_choice[np.arange(t1.size), t1] = True

            learnt = np.logical_and(pvals < alpha, pcs > null)
            learnt_pcs = pcs.copy()
            learnt_pcs[learnt] -= 1
            predicted_pc = sid_data[
                np.arange(sid_data.shape[0]),
                ix('pc:1') + np.argmax(learnt_pcs, axis=1)]

            sid_data[1:, ix('sw_pred')] = np.logical_not(
                t0[1:] == np.argmax(pcs, axis=1)[:-1])

            all_four = np.all(learnt, axis=1)
            for tsk in tasks:
                sid_data[all_four, ix('sc:{}'.format(tsk))] = np.nan
            sid_data[all_four, ix('sw_pred')] = np.nan

            sid_data[1:, ix('sc:1'):ix('sc:4') + 1] = predicted_pc[1:].reshape([-1, 1]) - pcs[1:]

            outdata.append(sid_data[1:])

    outdata = np.vstack(outdata)

    df = pd.DataFrame(outdata, columns=cols)

    # print(df.loc[df.sw_act == 1, ['t0', 'blkt', 'cor', 'pc:1', 'pc:2', 'pc:3', 'pc:4', 'rpc:1', 'rpc:2', 'rpc:3', 'rpc:4', 'tpc:1', 'tpc:2', 'tpc:3', 'tpc:4']])

    if save_as:
        df.to_csv(save_as)


if 1:
    save_as = '/Users/alexten/Projects/Exploration/R_docs/cmnr_data30cols.csv'

    self_challenging_data(path='pipeline_data/s3/joint_data.pkl',
                          save_as=save_as)