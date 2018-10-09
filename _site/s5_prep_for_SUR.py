import numpy as np
import argparse

from standards import *
import loc_utils as lut

parser = argparse.ArgumentParser()
parser.add_argument('x', help='puts 3 predictors separated by "/" into the data table')
parser.add_argument('-s', '--save_to', help='pickles and saves preprocessed data to path provided')
parser.add_argument('-o', '--output_df', help='prints a pandas DF to stdout', action='store_true')

ARGS = parser.parse_args()

def main():

    s = SURix()

    data = lut.unpickle('open_loop_data/SUR_data_2.pkl')
    data = lut.unpickle('open_loop_data/temp.pkl') # TODO delete later

    # Separate data of each batch (group x condition)
    sep_data = [] # order: 0,0; 0,1; 1,0; 1,1
    for grp_ind in [0,1]:
        for cnd_ind in [0,1]:
            bix = np.all([
                data[:, s.ix('group')] == grp_ind,
                data[:, s.ix('cond')] == cnd_ind
            ], axis=0)
            sep_data.append(data[bix,:])

    # 1. For each subject and each **TASK** within a subject, add a y value encoding whether their free trial choice
    # was the same as the **TASK**.
    # 2. Reorder each batch by task
    for i, grp in enumerate(sep_data):
        y = np.array([grp[:, s.ix('task')]==grp[:, s.ix('choice')]])
        aug = np.concatenate([grp, y.reshape(-1, 1)], axis=1)
        ordix = np.argsort(aug[:, s.ix('task')])
        sep_data[i] = aug[ordix]
    s.cols.append('y')

    num_rows = np.sum([grp.shape[0] for grp in sep_data])
    X = np.zeros([num_rows, 48])

    tasks = lut.get_unique(data, s.ix('task'))
    start = 0

    x1, x2, x3 = ARGS.x.split(sep='/')
    for i, grp in enumerate(sep_data):
        for j, tsk in enumerate(tasks):
            mask = lut.get_mask(grp, {s.ix('task'): tsk})
            X[start:int(start + grp.shape[0] / len(tasks)), i*12+j*3+0] = grp[mask, s.ix(x1)]
            X[start:int(start + grp.shape[0] / len(tasks)), i*12+j*3+1] = grp[mask, s.ix(x2)]
            X[start:int(start + grp.shape[0] / len(tasks)), i*12+j*3+2] = grp[mask, s.ix(x3)]
            start += int(grp.shape[0] / len(tasks))

    Y = np.concatenate([d[:, -1] for d in sep_data], axis=0).reshape(-1, 1)
    print(Y)
    D = np.concatenate([Y, np.ones_like(Y), X], axis=1)

    c = ['Y', 'const']
    for grp in ['F', 'S']:
        for cnd in ['I+', 'I-']:
            for tsk in ['1D', 'I1D', '2D', 'R']:
                for iv in ['PC', 'LP', 'LRN']:
                    c.append('{}:{}:{}:{}'.format(iv,tsk,cnd,grp))


    if ARGS.save_to: lut.dopickle(ARGS.save_to, D)
    if ARGS.output_df: lut.print_arr(D, c, nonints=c[1:], round_=2)


if __name__=='__main__': main()