import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('data', help='data file')
parser.add_argument('-f', '--save_fig', help='save_fig')

ARGS = parser.parse_args()
import loc_utils as lut

from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

def z(arr):
    return (arr - np.mean(arr, axis=0)) / np.std(arr, axis=0)

def main():
    data = lut.unpickle(ARGS.data)

    Y = data[:,0].reshape(-1,1)
    print(Y)
    X = data[:,1:]

    # Y_ = z(data[:,0]).reshape(-1,1)
    X_ = np.concatenate([X[:,0].reshape(-1,1), z(X[:,1:])], axis=1)

    model = sm.Logit(Y, X_)
    results = model.fit()

    print(results.summary())

    betas = results.params[1:].reshape(-1,3)

    tasks = np.tile(np.arange(1,5), 4).reshape(-1,1)
    groups = np.repeat(np.array([0,1]), 8).reshape(-1,1)
    conds = np.tile(np.repeat(np.array([0,1]), 4), 2).reshape(-1,1)

    print([a.shape for a in [betas, tasks, groups, conds]])

    data = np.concatenate([groups, conds, tasks, betas], axis=1)
    df = pd.DataFrame(data=data, columns=['group', 'condition', 'task', 'PC', 'LP', 'LRN'])
    df.group = df.group.replace([0,1], ['F', 'S'])
    df.condition = df.condition.replace([0,1], ['I+', 'I-'])
    df.task = df.task.replace([1,2,3,4], ['1D', 'I1D', '2D', 'R'])
    df['group_cond'] = df.group + '/' + df.condition

    fig = plt.figure(1, figsize=[12,4])
    sns.set_style("whitegrid")

    for i, y in enumerate(['PC', 'LP', 'LRN']):
        ax = fig.add_subplot(1,3,i+1)
        sns.stripplot(x='group_cond', y=y, hue='task', data=df, ax=ax)
        sns.despine()
        plt.ylim(-.7,.7)
        plt.xlabel(y)
        plt.ylabel('beta values' if i==0 else '')
        ax.axhline(0, lw=1, c='k')
        if i <= 1: ax.legend().remove()

    if ARGS.save_fig: fig.savefig('figures/SUR_{}.pdf'.format(ARGS.save_fig))

if __name__=='__main__': main()