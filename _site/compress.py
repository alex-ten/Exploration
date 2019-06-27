from loc_utils import may_be_make_dir
import os
import argparse


def compress(path):
    if os.path.isdir(path):
        output_dir = may_be_make_dir(path + '_compressed')
        for file in os.listdir(path):
            if 'svg' in file:
                os.system('scour -i {} -o {}'.format(os.path.join(path, file), os.path.join(output_dir , file)))

    else:
        os.system('scour -i {} -o {}'.format(os.path.join(path), os.path.join(path.strip('.svg') + '_compressed.svg')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path to raw data file')

    ARGS = parser.parse_args()

    compress(ARGS.path)


from datetime import date
from argparse

d0 = date(2018, 10, 1)
d1 = date(2008, 9, 26)
delta = d1 - d0
print delta.days