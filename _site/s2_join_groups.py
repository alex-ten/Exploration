import numpy as np
import loc_utils as lut

from standards import *
import argparse


# Chech and remove outliers
def join_groups(data_dir, files_dict, output_df=False, save_to=False):

    # Column index
    # TODO figure out a good way to implement standard column names
    r = RAWix(include_group=False)
    rx = RAWXix(include_group=False)

    # Load data
    data_free = lut.unpickle(data_dir + '/' + files_dict['main_free'])
    data_strat = lut.unpickle(data_dir + '/' + files_dict['main_strat'])
    data_free_x = lut.unpickle(data_dir + '/' + files_dict['extra_free'])
    data_strat_x = lut.unpickle(data_dir + '/' + files_dict['extra_strat'])

    # Fix SIDs in the strategic group to be consistent inside the joint data set
    last = np.max(data_free[:, r.ix('sid')]) + 1
    data_strat[:, r.ix('sid')] = data_strat[:, r.ix('sid')] + last

    last = np.max(data_free_x[:, r.ix('sid')]) + 1
    data_strat_x[:, r.ix('sid')] = data_strat_x[:, r.ix('sid')] + last

    # Join data sets (main and extra data, respectively)
    data_free = np.concatenate([np.zeros([data_free.shape[0], 1]), data_free], axis=1)
    data_strat = np.concatenate([np.ones([data_strat.shape[0], 1]), data_strat], axis=1)
    data = np.concatenate([data_free, data_strat], axis=0)
    del data_free, data_strat

    data_free_x = np.concatenate([np.zeros([data_free_x.shape[0], 1]), data_free_x], axis=1)
    data_strat_x = np.concatenate([np.ones([data_strat_x.shape[0], 1]), data_strat_x], axis=1)
    data_x = np.concatenate([data_free_x, data_strat_x], axis=0)
    del data_free_x, data_strat_x

    # TODO perform the selection below elsewhere (when you compute vars at stage 4)
    # Select data to work with from the main data set
    # train_trials = data[:, r.ix('stage')] == 0
    # init_free_trials = data[:, r.ix('trial')] == 61
    # open_loop_trials = np.logical_or(train_trials, init_free_trials)
    # data = data[open_loop_trials, :]

    # Display data if needed
    if output_df == 'main':
        lut.print_arr(data, r.cols, nonints=['rt'])
    elif output_df == 'extra':
        lut.print_arr(data_x, rx.cols)
    elif output_df == 'both':
        lut.print_arr(data, r.cols)
        lut.print_arr(data_x, rx.cols)

    # Save data if needed to the path provided
    if save_to:
        lut.dopickle(save_to, {'main': data, 'extra': data_x})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('parent_dir', help='path to parent data directory with a strict structure')
    parser.add_argument('files_dict', help='a dict mapping data sets to filenames, e.g. "{\'main_free\':\'file.pkl\'}"')
    parser.add_argument('-s', '--save_to', help='pickles and saves preprocessed and joint data to path provided', default=False)
    parser.add_argument('-c', '--save_csv', help='saves preprocessed and joint data to path provided as csv', default=False)
    parser.add_argument('-o', '--output_df', help='prints a pandas DF to stdout', default=False)
    ARGS = parser.parse_args()

    join_groups(data_dir=ARGS.parent_dir,
                files_dict=ARGS.files_dict,
                output_df=ARGS.output_df,
                save_to=ARGS.save_to)