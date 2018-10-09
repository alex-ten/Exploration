import numpy as np
import argparse
import statsmodels.api as sm
from tqdm import tqdm
import os
import pandas as pd
import csv
from scipy.stats import rankdata

from standards import *
import loc_utils as lut


def rank_pc(pc):
    if pc < .5:
        if pc < .25:
            return 1
        return 2
    else:
        if pc < .75:
            return 3
        return 4


def rank_lrn4(lrn):
    if lrn < 5:
        if lrn < 2.5:
            return 1
        else:
            return 2
    else:
        if lrn == 5:
            return np.random.randint(2, 4)
        if lrn < 7.5:
            return 3
        else:
            return 4


def rank_lrn3(lrn):
    if lrn < 20/3:
        if lrn <10/3:
            return 1
        else:
            return 2
    else:
        return 3


# Column index
r = RAWix()

# Column eXtra index
rx = RAWXix()

# Column analysis index
s = SURix()


def save_weka(path, data, attributes, relation, title=''):
    header = '% 1. Title: {}\n%\n% 2. Comments:\n'.format(title) + '%\n'*5
    format_dict = {'numeric': float, 'integer': int}
    data = pd.DataFrame(data, columns=[i[0] for i in attributes])
    header += '@RELATION {}\n\n'.format(relation)

    for attr, dtype in attributes:
        if dtype == 'numeric' or dtype == 'integer':
            header += '@ATTRIBUTE {:12} {}\n'.format(attr, dtype.upper())
            data[attr] = data[attr].astype(format_dict[dtype])
        elif '{' in dtype:
            header += '@ATTRIBUTE {:12} {}\n'.format(attr, dtype)
            keys = data[attr].unique()
            vals = dtype.strip('{}').split(',')
            data[attr] = data[attr].replace(to_replace=keys, value=vals)
    header += '\n@DATA\n'

    if os.path.isfile(path):

        overwrite = input('File {} exists. Overwrite? [y/n]\n>>> '.format(path))

        if overwrite.lower() == 'y':
            print('Overwriting data to {}'.format(path))
            with open(path, 'w') as f:
                f.write(header)
            with open(path, 'a') as f:
                data.to_csv(f, header=False, index=False)
            print('Done saving.')
        else:
            print('Data not saved.')
            return
    else:
        lut.may_be_make_dir(os.path.dirname(path))
        print('Saving data to {}'.format(path))
        with open(path, 'w') as f:
            f.write(header)
        with open(path, 'a') as f:
            data.to_csv(f, header=False, index=False)
        print('Done saving.')


def main(path, save_as=''):
    joint_data = lut.unpickle(path)
    mdata = joint_data['main']
    open_loop = lut.get_mask(mdata, {r.ix('trial'): 61}, '<=')
    mdata[:-1, r.ix('switch')] = mdata[:, r.ix('switch')][1:]

    mdata = mdata[open_loop, :]
    xdata = joint_data['extra']

    weka = []

    tasks = lut.get_unique(mdata, r.ix('cat'))
    monsters = lut.get_unique(mdata, r.ix('fam')).tolist()

    x1 = np.arange(0, 15).reshape([-1, 1])
    x0 = np.ones_like(x1)
    X = np.concatenate([x0, x1], axis=1)
    for i, row in enumerate(tqdm(xdata, desc='preparing data')):

        sid = row[rx.ix('sid')]
        grp = row[rx.ix('group')]
        cnd = row[rx.ix('cond')]

        weka_row = [sid, grp, cnd]

        for ti, tsk in enumerate(tasks.astype(int)):
            mask = lut.get_mask(
                arr=mdata,
                conds={
                    r.ix('sid'): sid,
                    r.ix('group'): grp,
                    r.ix('cond'): cnd,
                    r.ix('cat'): tsk,
                    r.ix('stage'): 0
                    }
                )

            correct = mdata[mask, r.ix('cor')]
            pc15 = np.mean(correct)
            pc10 = np.mean(correct[-10:])
            pc5 = np.mean(correct[-5:])

            rpc15 = rank_pc(pc15)
            rpc10 = rank_pc(pc10)
            rpc5 = rank_pc(pc5)

            lp1 = np.mean(correct[-5:]) - np.mean(correct[:5])
            lp2 = sm.OLS(correct.reshape([-1, 1]), X).fit().params[1]

            mi = monsters.index(mdata[mask, r.ix('fam')][0])

            lrn = row[rx.ix('q6m1') + mi * 7]
            rlrn = rank_lrn4(lrn)

            weka_row += [pc15, pc10, pc5, rpc15, rpc10, rpc5, lp1, lp2, lrn, rlrn]

        free_trial = lut.get_mask(mdata, conds={
                    r.ix('sid'): sid,
                    r.ix('group'): grp,
                    r.ix('cond'): cnd,
                    r.ix('stage'): 1})
        weka_row.append(mdata[free_trial, r.ix('cat')])

        weka.append(weka_row)

    if save_as:
        attributes = [
            ('sid', 'integer'),
            ('group', '{F, S}'),
            ('cond', '{i-, i+}'),
            ('1_pc15', 'numeric'),
            ('1_pc10', 'numeric'),
            ('1_pc5', 'numeric'),
            ('1_rpc15', 'integer'),
            ('1_rpc10', 'integer'),
            ('1_rpc5', 'integer'),
            ('1_lp1', 'numeric'),
            ('1_lp2', 'numeric'),
            ('1_lrn', 'numeric'),
            ('1_rlrn', 'integer'),
            ('2_pc15', 'numeric'),
            ('2_pc10', 'numeric'),
            ('2_pc5', 'numeric'),
            ('2_rpc15', 'integer'),
            ('2_rpc10', 'integer'),
            ('2_rpc5', 'integer'),
            ('2_lp1', 'numeric'),
            ('2_lp2', 'numeric'),
            ('2_lrn', 'numeric'),
            ('2_rlrn', 'integer'),
            ('3_pc15', 'numeric'),
            ('3_pc10', 'numeric'),
            ('3_pc5', 'numeric'),
            ('3_rpc15', 'integer'),
            ('3_rpc10', 'integer'),
            ('3_rpc5', 'integer'),
            ('3_lp1', 'numeric'),
            ('3_lp2', 'numeric'),
            ('3_lrn', 'numeric'),
            ('3_rlrn', 'integer'),
            ('4_pc15', 'numeric'),
            ('4_pc10', 'numeric'),
            ('4_pc5', 'numeric'),
            ('4_rpc15', 'integer'),
            ('4_rpc10', 'integer'),
            ('4_rpc5', 'integer'),
            ('4_lp1', 'numeric'),
            ('4_lp2', 'numeric'),
            ('4_lrn', 'numeric'),
            ('4_rlrn', 'integer'),
            ('choice', '{1D, I1D, 2D, R}')
        ]
        weka = np.array(weka)
        save_weka(save_as, weka, attributes, 'open_loop', 'Open Loop Data')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path to data file')
    parser.add_argument('-s', '--save', help='save file')

    ARGS = parser.parse_args()
    main(ARGS.path, ARGS.save)

