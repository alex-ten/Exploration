import numpy as np
from scipy.stats import rankdata
import argparse

import loc_utils as lut
from standards import *


# Column index
r = RAWix()

def report_analysis(analysis, outliers):
    n = len(outliers)
    if n > 0:
        print('{} detected {} outlier(s):'.format(analysis, n))
        print(outliers)
    else:
        print('No outliers detected by {}'.format(analysis))


def measure_choice_bias(data, collapse_tasks=True):
    sids = lut.get_unique(data, r.ix('sid'))
    tasks = lut.get_unique(data, r.ix('cat'))

    num_tasks = tasks.size
    num_sids = sids.size

    pmax = np.zeros([num_sids, num_tasks])
    for i, sid in enumerate(sids):
        for j, tsk in enumerate(tasks):
            mask = lut.get_mask(data, {r.ix('sid'): sid, r.ix('cat'): tsk})
            choices, counts = np.unique(data[mask, r.ix('food')], return_counts=True)
            pmax[i, j] = np.max(counts) / np.sum(counts)
    return np.mean(pmax, axis=1) if collapse_tasks else pmax


def measure_allocation_variance(data):
    sids = lut.get_unique(data, r.ix('sid'))
    num_sids = sids.size

    free_stage = 1

    stds = np.zeros(num_sids)
    for i, sid in enumerate(sids):
        mask = lut.get_mask(data, {r.ix('sid'): sid, r.ix('stage'): free_stage})
        choices, counts = np.unique(data[mask, r.ix('cat')], return_counts=True)
        stds[i] = np.std(
            np.pad(counts, pad_width=(0, 4-np.size(counts)), mode='constant')
        )
    return stds


def detect_extreme_sticking(data):
    outliers = []
    sids = lut.get_unique(data, r.ix('sid'))
    for sid in sids:
        mask = lut.get_mask(data, {r.ix('sid'): sid, r.ix('stage'): 1})
        tasks_played = data[mask, r.ix('cat')]
        if np.all(tasks_played == tasks_played[0]):
            outliers.append(sid)
    report_analysis('Extreme sticking analysis', outliers)
    return outliers


def detect_choice_bias(data, criterion='sd', critval=2, return_cutoff=False):
    mean_choice_bias = measure_choice_bias(data)

    if criterion.lower() == 'val':
        outliers = np.where(mean_choice_bias > critval)[0].tolist()
    else:
        average_mean_choice_bias = np.mean(mean_choice_bias)
        sd_mean_choice_bias = np.std(mean_choice_bias)
        critval = average_mean_choice_bias + sd_mean_choice_bias * critval
        outliers = np.where(np.abs(mean_choice_bias) > critval)[0].tolist()

    report_analysis('Choice bias analysis', outliers)
    if return_cutoff:
        return outliers, critval
    return outliers


def detect_by_allocation_variance(data, crit):
    outliers = []
    sids = lut.get_unique(data, r.ix('sid'))
    num_sids = sids.size

    free_stage = 1

    stds = np.zeros(num_sids)
    for i, sid in enumerate(sids):
        mask = lut.get_mask(data, {r.ix('sid'): sid, r.ix('stage'): free_stage})
        choices, counts = np.unique(data[mask, r.ix('cat')], return_counts=True)
        stds[i] = np.std(
            np.pad(counts, pad_width=(0, 4-counts.size), mode='constant')
        )
        if stds[i] > crit:
            outliers.append(sid)
    report_analysis('Allocation variance', outliers)
    return outliers


def remove_by_sid(data, sids, assign_new_ids=False):
    outliers_mask = np.isin(data[:, r.ix('sid')], sids)
    filtered_data = data[np.logical_not(outliers_mask), :]
    if assign_new_ids:
        filtered_data[:, r.ix('sid')] = rankdata(filtered_data[:, r.ix('sid')], 'dense') - 1
    return filtered_data


def filter_outliers(path_to_data, save_to=None, extreme_sticking=True,
                    choice_bias_criterion='sd', choice_bias_critval=2,
                    alloc_var_crit=None,
                    assign_new_ids=True,
                    report_counts=True):
    data = lut.unpickle(path_to_data)
    lut.report_subject_counts(data['main'])
    stickers1, stickers2, choice_biased = [], [], []
    if extreme_sticking:
        stickers1 = detect_extreme_sticking(data['main'])

    if choice_bias_criterion:
        choice_biased = detect_choice_bias(data['main'], criterion=choice_bias_criterion, critval=choice_bias_critval)

    if alloc_var_crit:
        stickers2 = detect_by_allocation_variance(data['main'], alloc_var_crit)

    all_outliers = np.unique(stickers1 + stickers2 + choice_biased)
    print('Removing {} outliers'.format(all_outliers.size))

    data['main'] = remove_by_sid(data['main'], all_outliers, assign_new_ids=assign_new_ids)
    data['extra'] = remove_by_sid(data['extra'], all_outliers, assign_new_ids=assign_new_ids)
    lut.report_subject_counts(data['main'])

    if save_to:
        lut.dopickle(save_to, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('-s', '--save_to', help='pickles and saves data with outliers removed', default=None)
    parser.add_argument('-x', '--extreme_sticking', help='whether to remove extreme stickers', default=True)
    parser.add_argument('-c', '--side_bias_crit', help='pickles and saves data with outliers removed', default='sd')
    parser.add_argument('-v', '--side_bias_critval', help='pickles and saves data with outliers removed', default=2,
                        type=int)

    ARGS = parser.parse_args()

    filter_outliers(path_to_data=ARGS.path,
                    save_to=ARGS.save_to,
                    extreme_sticking=ARGS.extreme_sticking,
                    choice_bias_criterion=ARGS.side_bias_crit,
                    choice_bias_critval=ARGS.side_bias_critval)

    # # detect_extreme_sticking by groups
    # for grp in [0,1]:
    #     for cnd in [0,1]:
    #         mask = lut.get_mask(main_data, {r.ix('group'):grp, r.ix('cond'): cnd})
    #         detect_extreme_sticking(main_data[mask, :])