import numpy as np
import argparse
import matplotlib.pyplot as plt

from scipy.special import comb

import loc_utils as lut

colors = ['#375e97', '#f18d9e', '#ffbb00', '#3f681c']

class Explorer(object):
    def __init__(self, alpha, stickiness):
        self.alpha = alpha
        self.stickiness = stickiness

    def make_choice(self,  pcs, pvals):
        islearned = pvals < self.alpha
        inds = np.arange(pcs.size)

        if np.all(islearned):
            choice = np.random.choice(inds, p=np.ones_like(inds)/inds.size)
        else:
            candidates = pcs[~islearned]
            choice = np.argmax(candidates)
        return choice


def pval(n, k, p):
    return comb(n,k) * p**k * (1-p)**(n-k)


def pval_from_pc(pc, trial_num, p):
    n = np.ones_like(pc) * trial_num
    k = n * pc
    return comb(n,k) * p**k * (1-p)**(n-k)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def main():
    null = .5
    alpha = .01
    stickiness = None

    x = Explorer(alpha, stickiness)

    pcs = np.array([.9, .6, .5, .4])
    trials_played = 15
    trials_to_play = 250

    data = []
    for i in range(trials_to_play):
        pvals = pval_from_pc(pc=pcs, trial_num=trials_played, p=null)
        choice = x.make_choice(pcs, pvals)
        outcome = np.random.choice([1, 0], p=[pcs[choice], 1-pcs[choice]])
        pcs[choice] = (pcs[choice] * trials_played + outcome)/(trials_played + 1)

        trials_played += 1
        data.append(
            [choice] + pcs.tolist() + pvals.tolist()
        )
    data = np.stack(data, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for tsk in [0, 1, 2, 3]:
        ax.plot(
            np.arange(data.shape[0]),
            data[:, 1 + tsk],
            lw=2, c=colors[tsk]
        )

        mask = data[:, 0] == tsk
        nans = np.full([data.shape[0], ], np.nan)
        nans[mask] = 
        ax.plot(
            np.arange(data.shape[0]),
            data[:, 1 + tsk],
            lw=2, c=colors[tsk]
        )

    ax.set_xlim(0, trials_to_play)
    ax.set_ylim(0, 1)

    plt.show()


if __name__ == '__main__':
    main()