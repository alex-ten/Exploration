import numpy as np
from matplotlib import colors
import statsmodels.api as sm
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

import loc_utils as lut

stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

cols = 'sid,grp,stage,trial,blkt,t0,t1,pc1,pc2,pc3,pc4,p1,p2,p3,p4,sc1,sc2,sc3,sc4,sw_pred,sw_act,sw_lag'.split(',')
ix = cols.index
gcolors = ['#008fd5', '#fc4f30', '#e5ae38', '#6d904f']

def onehot(ind):
    ind = ind - 1
    onehots = np.zeros([ind.size, 4])
    onehots[np.arange(ind.size), ind] = 1
    return onehots


def run_MNlog(x, y, pivot):
    y = y.replace(to_replace=pivot, value=0, inplace=False)
    mdl = sm.MNLogit(y, x)
    mdl_fit = mdl.fit(maxiter=100, full_output=1)
    return mdl_fit


def softmax(x, stable=0):
    """Compute softmax values for each sets of scores in x."""
    if stable:
        e_x = np.exp(x - np.max(x, axis=0))
        return e_x / e_x.sum(axis=0)
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def add_text(ax, arr):
    for (j, i), label in np.ndenumerate(arr):
        ax.text(i, j, label, ha='center', va='center')


def save_it(fig, savedir, figname, save_as='svg', dpi=500, compress=True):
    s = savedir+'/{}.{}'.format(figname, save_as)
    fig.savefig(s, format=save_as, dpi=500)
    if compress:
        os.system('scour -i {} -o {}'.format(s, s.replace('img', 'img_compressed')))


def main(predict=0):
    data = lut.unpickle('pipeline_data/scdata/modeling_data_sw_lag.pkl')

    sids, groups = lut.get_unique(data, [ix('sid'), ix('grp')])
    tasks = [1, 2, 3, 4]

    fig = plt.figure(num='Model summary', figsize=[12, 8])

    images = []
    for grp in groups:
        ii = 50

        gmask = lut.get_mask(arr=data, conds={ix('grp'): grp})
        gdata = data[gmask, :]

        pc = gdata[:, ix('pc1'):ix('pc4') + 1]
        ch = gdata[:, ix('sc1'):ix('sc4') + 1]
        switch = gdata[:, ix('sw_act')]
        rt = np.zeros_like(pc)

        for j, tsk in enumerate([1, 2, 3, 4]):
            tmask = lut.get_mask(gdata, {ix('t0'): tsk})
            rt[tmask, j] = 1

        rt[0, :] = 15
        rt = np.cumsum(rt, axis=0)
        rt = np.transpose(rt.T / np.sum(rt, axis=1))

        ind = onehot(gdata[:, ix('t0')].astype(int)).astype(bool)
        x1 = pc[ind].reshape([-1, 1])*100
        x2 = ch[ind].reshape([-1, 1])*100
        x3 = rt[ind].reshape([-1, 1])*100

        X = sm.add_constant(np.concatenate([x1, x3], 1), prepend=False)
        X = pd.DataFrame(data=X, columns='pc,rt,const'.split(','))

        Y = pd.DataFrame(data=gdata[:, ix('t1')].astype(int), columns=['Choice'])

        N = 100
        exrt = np.linspace(0, 1, 265)
        expc = [.4, .6, .8, 1.0]

        softmax_weights = []
        const_mat = []
        pc_mat = []
        rt_mat = []

        z = np.zeros(X.shape[1])
        for ti, tsk in enumerate(tasks):
            model = run_MNlog(X, Y, pivot=tsk)
            beta_ = model.params.T
            softmax_weights.append(-np.sum(beta_, axis=1))

            beta_ = np.insert(beta_.values, ti, z, axis=0)
            const_mat.append(beta_[:, -1])
            pc_mat.append(beta_[:, 0])
            rt_mat.append(beta_[:, 1])

            # Predict
            if predict:
                logits = np.dot(beta_, X.T)
                Y_hat = np.argmax(softmax(logits, stable=1), axis=0)

            if ti == 3:
                print(model.summary())

        overall = np.stack(softmax_weights, axis=1).T
        onevrest_const = np.stack(const_mat, axis=0)
        onevrest_pc = np.stack(pc_mat, axis=0)
        onevrest_rt = np.stack(rt_mat, axis=0)

        # ax1 = fig.add_subplot(2, 4, 1 + grp * 4)
        # ax1.matshow(overall, aspect='equal', cmap='RdBu')
        # ax1.set_title('Classifier {}'.format('FS'[grp]))
        # ax1.xaxis.set_ticks_position('bottom')
        # ax1.yaxis.set_ticks_position('left')
        # ax1.set_xticks([0, 1, 2])
        # ax1.set_yticks([0, 1, 2, 3])
        # ax1.set_xticklabels('PC,T,Const'.split(','))
        # ax1.set_yticklabels('1D,I1D,2D,R'.split(','))
        # add_text(ax1, np.around(overall, 3))

        ax2 = fig.add_subplot(2, 3, 1 + grp * 3)
        odds = np.exp(onevrest_pc)
        images.append(ax2.matshow(odds, aspect='equal', cmap='viridis'))
        ax2.set_title('Group {}: OvR PC'.format('FS'[grp]), pad=20)
        ax2.xaxis.set_ticks_position('top')
        ax2.yaxis.set_ticks_position('left')
        ax2.set_xticks([0, 1, 2, 3])
        ax2.set_yticks([0, 1, 2, 3])
        ax2.set_xticklabels('1D,I1D,2D,R'.split(','))
        ax2.set_yticklabels('1D,I1D,2D,R'.split(','))
        ax2.set_ylabel('Reference')
        ax2.set_xlabel('Odds: Pr(task)/P(reference)')
        add_text(ax2, np.around(odds, 3))

        ax3 = fig.add_subplot(2, 3, 2 + grp * 3)
        odds = np.exp(onevrest_rt)
        images.append(ax3.matshow(odds, aspect='equal', cmap='viridis'))
        ax3.set_title('Group {}: OvR RT'.format('FS'[grp]), pad=20)
        ax3.xaxis.set_ticks_position('top')
        ax3.yaxis.set_ticks_position('left')
        ax3.set_xticks([0, 1, 2, 3])
        ax3.set_yticks([0, 1, 2, 3])
        ax3.set_xticklabels('1D,I1D,2D,R'.split(','))
        ax3.set_yticklabels('1D,I1D,2D,R'.split(','))
        ax3.set_ylabel('Reference')
        ax3.set_xlabel('Odds: Pr(task)/Pr(reference)')
        add_text(ax3, np.around(odds, 3))

        logits = np.dot(beta_, X.T)
        Y_hat = np.argmax(softmax(logits, stable=1), axis=0) + 1
        CM = confusion_matrix(Y.values.squeeze(), Y_hat)
        CM = np.around(CM / CM.sum(axis=1), 3)
        print(CM)

        ax4 = fig.add_subplot(2, 3, 3 + grp * 3)
        ax4.matshow(CM, aspect='equal', cmap=plt.cm.Blues, vmin=0, vmax=1)
        ax4.set_title('Group {}: Confusion matrix'.format('FS'[grp]), pad=20)
        ax4.xaxis.set_ticks_position('top')
        ax4.yaxis.set_ticks_position('left')
        ax4.set_xticks([0, 1, 2, 3])
        ax4.set_yticks([0, 1, 2, 3])
        ax4.set_xticklabels('1D,I1D,2D,R'.split(','))
        ax4.set_yticklabels('1D,I1D,2D,R'.split(','))
        ax4.set_ylabel('Observed')
        ax4.set_xlabel('Predicted')
        add_text(ax4, CM)

        lut.dopickle('/Users/alexten/Projects/Exploration/model_weights_group{}'.format(grp), beta_)

        # print('Predictions:')
        # output = softmax(np.dot(beta, examples), stable=True)
        # print(np.around(output[:, ii], 5))

    # Find the min and max of all colors for use in setting the color scale.
    vmin = 0 #min(image.get_array().min() for image in images)
    vmax = 2 #max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    # fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)

    # Make images respond to changes in the norm of other images (e.g. via the
    # "edit axis, curves and images parameters" GUI on Qt), but be careful not to
    # recurse infinitely!
    def update(changed_image):
        for im in images:
            if (changed_image.get_cmap() != im.get_cmap()
                    or changed_image.get_clim() != im.get_clim()):
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())

    fig.tight_layout()
    fig.subplots_adjust(hspace=.3)
    # save_it(fig, '/Users/alexten/Projects/HFSP/img', 'MNLogit_pc_rt', save_as='svg', dpi=500, compress=True)


if __name__ == '__main__':
    main()
