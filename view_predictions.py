import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os

import loc_utils as lut
import vis_utils as vut

colors = ['#375e97', '#f18d9e', '#ffbb00', '#3f681c']


def save_it(fig, savedir, figname, save_as='svg', dpi=500, compress=True):
    s = savedir+'/{}.{}'.format(figname, save_as)
    fig.savefig(s, format=save_as, dpi=500)
    if compress:
        os.system('scour -i {} -o {}'.format(s, s.replace('img', 'img_compressed')))


def softmax(x, stable=0):
    """Compute softmax values for each sets of scores in x."""
    if stable:
        e_x = np.exp(x - np.max(x, axis=0))
        return e_x / e_x.sum(axis=0)
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def main(figname):
    exrt = np.linspace(0, 1, 250) * 100
    expc = np.array([.4, .6, .8, 1.0]) * 100
    onez = np.ones_like(exrt)
    zeroz = np.zeros_like(onez).reshape(1, -1)

    fig = plt.figure(num=figname, figsize=[9, 9])

    gs = gridspec.GridSpec(expc.size, 2)

    for grp in [0, 1]:
        beta = lut.unpickle('/Users/alexten/Projects/Exploration/model_weights_group{}'.format(grp))

        for pci, pc in enumerate(expc):
            X = np.stack([
                onez * pc,
                exrt,
                onez
            ], axis=0)

            logits = np.dot(beta, X)
            Y_hat = softmax(logits, stable=1)

            Y_stack = np.cumsum(Y_hat, axis=0)
            Y_stack = np.concatenate([zeroz, Y_stack], axis=0)

            ax = fig.add_subplot(gs[pci, grp])
            for i, row in enumerate(Y_stack[:-1]):
                ax.fill_between(exrt, Y_stack[i, :], Y_stack[i+1, :], facecolor=colors[i], label='1D,I1D,2D,R'.split(',')[i])
            vut.despine(ax, ['top', 'right'])
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1.05)

            if i == 0:
                ax.set_title('Group {}\nAt PC = {}%'.format('FS'[grp], int(pc)))
            else:
                ax.set_title('At PC = {}%'.format(int(pc)))

            ax.set_ylabel('Predicted probability'.format(pc))
            # if pci == 3:
            ax.set_xlabel('RT (%)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            ax.axvline(25, lw=1.2, c='k')

    fig.tight_layout()
    plt.show()

    save_it(fig, '/Users/alexten/Projects/HFSP/img', figname, save_as='svg', dpi=500, compress=True)


if __name__ == '__main__':
    main(
        figname='predictions_for_rt'
    )
