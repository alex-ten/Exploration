import os
import numpy as np
import argparse

import loc_utils as lut


def csv_to_pickle(path, save_to):
    data = np.loadtxt(path, dtype=int, delimiter=',', skiprows=1)
    lut.dopickle(save_to, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path to csv file')
    parser.add_argument('-s', '--save_to', help='pickles and saves preprocessed data to path provided', default=False)
    ARGS = parser.parse_args()

    csv_to_pickle(path=ARGS.path, save_to=ARGS.save_to)