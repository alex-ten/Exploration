import pandas as pd
import numpy as np
import os
import argparse

from loc_utils import may_be_make_dir


def preprocess_raw_main(path, save_to, id_map=None):
    # Read csv into pandas DataFrame
    df = pd.read_csv(path)

    if id_map:
        catmap = None
        names, nums = [], []
        for name, num in id_map.items():
            names.append(name)
            nums.append(num)

        df.replace(
            to_replace=names,
            value=nums,
            inplace=True
        )
    else:
        catmap = {}
        for num, name in enumerate(df['participant:assignmentId'].astype('category').cat.categories):
            catmap[name] = num
        df['participant:assignmentId'] = df['participant:assignmentId'].astype('category').cat.codes


    # Add a switch column
    df.loc[:, 'switch'] = pd.Series(np.array(df.blockTrial == 1).astype(int), index=df.index)  # Find switch trials
    df.loc[:, 'blockTrial'] = df.blockTrial - 1

    # Parse the monster characteristics (family, and the two dimensions are stored in separate cols)
    df.loc[:, 'D1'] = df.monster.str.split('_', expand=True)[1]
    df.loc[:, 'D2'] = df.monster.str.split('_', expand=True)[2]

    # Convert text categories, state, and family to numbers
    df.replace(
        to_replace=['category2D', 'categoryRandom', 'categoryIgnore1D', 'category1D',
                    'train', 'free', 'test',
                    'Bear', 'Bunny', 'GreenMonster', 'Squid',
                    'bananas', 'broccoli', 'carrot', 'grilled_cheese', 'oranges', 'pancakes', 'tacos', 'waffles'],
        value=[3, 4, 2, 1,
               0, 1, 2,
               1, 2, 3, 4,
               1, 2, 3, 4, 5, 6, 7, 8],
        inplace=True
    )

    # Convert boolean correct to int (for presentation)
    df.correct = df.correct.astype(int)

    # Remove unwanted columns
    to_remove = ['trialStartTime', 'monster', 'preferredFood']
    df.drop(to_remove, axis=1, inplace=True)
    df.rename(columns={'state': 'stage', 'participant:assignmentId': 'sid'}, inplace=True)

    # Rearrange dataframe
    new_order = ['sid', 'trial',
                 'condition', 'stage', 'blockTrial',
                 'family', 'D1', 'D2', 'category',
                 'choice', 'correct', 'switch', 'rt']
    df = df[new_order]

    # Save preprocessed csv
    may_be_make_dir(os.path.dirname(save_to))

    df.to_csv(
        path_or_buf=save_to,
        sep=',',
        na_rep='',
        header=True,
        index=False,
        mode='w',
        line_terminator='\n',
    )

    if id_map:
        return save_to
    else:
        return save_to, catmap


def preprocess_raw_extra(path, save_to, id_map=None):
    # Read csv into pandas DataFrame
    df = pd.read_csv(path, usecols=list(range(30)))
    if id_map:
        catmap = 0
        names, nums = [], []
        for name, num in id_map.items():
            names.append(name)
            nums.append(num)
        df.replace(
            to_replace=names,
            value=nums,
            inplace=True
        )
    else:
        catmap = {}
        for num, name in enumerate(df['participant:assignmentId'].astype('category').cat.categories):
            catmap[name] = num
        df['participant:assignmentId'] = df['participant:assignmentId'].astype('category').cat.codes


    # Remove unwanted columns
    df.rename(columns={'participant:assignmentId': 'sid'}, inplace=True)

    df.to_csv(
        path_or_buf=save_to,
        sep=',',
        na_rep='',
        header=True,
        index=False,
        mode='w',
        line_terminator='\n',
    )

    if id_map:
        return save_to
    else:
        return save_to, catmap


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path to raw data file')
    parser.add_argument('dataset', help='which data set to preprocess [main/extra]')
    parser.add_argument('save_to', help='path to preprocessed data file to write')
    parser.add_argument('-v', '--verbose', help='boolean flag to turn on ')

    ARGS = parser.parse_args()

    if ARGS.dataset == 'main':
        preprocess_raw_main(path=ARGS.path, save_to=ARGS.save_to)
    elif ARGS.dataset == 'extra':
        preprocess_raw_extra(path=ARGS.path, save_to=ARGS.save_to)
