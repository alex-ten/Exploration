import pandas as pd
import numpy as np
import pickle
import os
import operator

from standards import RAWix


def may_be_make_dir(path):
    """
    If the directory is created between the `os.path.exists` and the `os.makedirs` calls, the `os.makedirs` will fail
    with an OSError.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def unpickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def dopickle(path, data):
    if os.path.isfile(path):

        overwrite = input('File {} exists. Overwrite? [y/n]\n>>> '.format(path))

        if overwrite.lower() == 'y':
            print('Overwriting data to {}'.format(path))
            with open(path, 'wb') as file:
                pickle.dump(data, file)
            print('Done saving.')
        else:
            print('Data not saved.')
            return

    else:
        may_be_make_dir(os.path.dirname(path))
        print('Saving data to {}'.format(path))
        with open(path, 'wb') as file:
            pickle.dump(data, file)
        print('Done saving.')


def get_unique(arr, cols):
    if not isinstance(cols, list):
        return np.unique(arr[:, cols]).astype(int)
    return [np.unique(arr[:, c]).astype(int) for c in cols]


def get_truth(inp, relate, cut):
    ops = {'>': operator.gt,
           '<': operator.lt,
           '>=': operator.ge,
           '<=': operator.le,
           '==': operator.eq,
           '!=': operator.ne}
    return ops[relate](inp, cut)


def get_mask(arr, conds, op='==',):
    tl = []
    for k, v in conds.items():
        tl.append(get_truth(arr[:, k], op, v))
    return np.all(tl, axis=0)


def report_subject_counts(data):
    r = RAWix()
    groups = get_unique(data, r.ix('group')).astype(int)
    conds = get_unique(data, r.ix('cond')).astype(int)

    for g in groups:
        for c in conds:
            mask = get_mask(data, {r.ix('group'): g, r.ix('cond'): c})
            n = np.unique(data[mask, r.ix('sid')]).size
            print('g{} c{}: {}'.format(g, c, n))
    print('Total:', get_unique(data, r.ix('sid')).size)


def print_arr(arr, cols, nonints=None, group_ind=False, round_=False):
    if not isinstance(nonints, list) and nonints: nonints = [nonints]
    cols = [s.upper() for s in cols]
    df = pd.DataFrame(data=arr, columns=cols)
    if nonints:
        for nonint in nonints: cols.pop(cols.index(nonint.upper()))
    df[cols] = df[cols].astype(int)
    if group_ind: df.set_index(['GROUP', 'SID'], inplace=True)
    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 2000)
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 1000)
    print(df.round(round_) if round_ else df)
    print(df.shape)


def retro_pc(seq, window):
    if seq.size == 0:
        return seq
    elif seq.size < window:
        return np.cumsum(seq) / (np.arange(seq.size) + 1)

    dyn = np.cumsum(seq[:window-1]) / np.arange(1, window)
    s = np.insert(np.cumsum(seq), 0, [0])
    reg = (s[window:] - s[:-window]) * (1. / window)
    return np.concatenate([dyn, reg])


def recency_pc(seq):
    pass