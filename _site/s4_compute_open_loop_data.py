import numpy as np
import argparse
import statsmodels.api as sm

from standards import *
import loc_utils as lut

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save_to', help='pickles and saves preprocessed data to path provided')
parser.add_argument('-o', '--output_df', help='prints a pandas DF to stdout', action='store_true')
parser.add_argument('-r', '--save_raw', help='pickles and saves the unified raw data', action='store_true')

ARGS = parser.parse_args()

def rank_pc(pc):
    if pc < .5:
        if pc <.25: return 1
        return 2
    else:
        if pc < .75: return 3
        return 4

def rank_lrn4(lrn):
    if lrn < 5:
        if lrn <2.5: return 1
        else: return 2
    else:
        if lrn == 5: return np.random.randint(2,4)
        if lrn < 7.5: return 3
        else: return 4

def rank_lrn3(lrn):
    if lrn < 20/3:
        if lrn <10/3: return 1
        else: return 2
    else: return 3

def main():
    '''
    Here we will take the clean pickled data and preprocess it for the SUR analysis. The resulting data set will contain
    4*400 rows, each row representing one task for one subject in one of the groups. Column headers are listed in the
    `sur_cols` list. The program can be run with the -o flag to see the resulting data set
    '''

    # Column index
    r = RAWix()

    # Column eXtra index
    rx = RAWXix()

    # Column analysis index
    s = SURix()

    data = lut.unpickle('pickled_data/temp.pkl')
    dataX = lut.unpickle('pickled_data/temp_extra.pkl')

    # Compute for analysis # TODO change the routine description "Compute for analysis"
    out_data = np.zeros([4*(np.alen(dataX)), len(s.cols)])
    out_data[:, 0] += np.arange(0, out_data.shape[0], 1)

    cats = lut.get_unique(data, r.ix('cat'))
    for i, row in enumerate(dataX):
        sid = row[rx.ix('sid')]
        grp = row[rx.ix('group')]
        cnd = row[rx.ix('cond')]
        for ci, cat in enumerate(cats.astype(int)):
            ii = i * 4 + ci
            out_data[ii, s.ix('sid')] = sid
            out_data[ii, s.ix('group')] = grp
            out_data[ii, s.ix('cond')] = cnd
            out_data[ii, s.ix('task')] = cat
            out_data[ii, s.ix('choice')] = data[(i+1)*61-1, r.ix('cat')]

            mask = lut.get_mask(
                arr=data,
                conds={
                    r.ix('sid'): sid,
                    r.ix('group'): grp,
                    r.ix('cond'): cnd,
                    r.ix('cat'): cat,
                    r.ix('cor'): 1
                    }
                )
            pc = np.sum(mask) / 15

            out_data[ii, s.ix('pc')] = pc
            out_data[ii, s.ix('pc_rank')] = rank_pc(pc)
            mi = monsters.index(mdata[mask, r.ix('fam')][0])

            out_data[ii, s.ix('lrn')] = row[rx.ix('q6m1')+ci*7]
            out_data[ii, s.ix('lrn_rank3')] = rank_lrn3(row[rx.ix('q6m1')+ci*7])
            out_data[ii, s.ix('lrn_rank4')] = rank_lrn4(row[rx.ix('q6m1')+ci*7])

    sids = lut.get_unique(out_data, s.ix('sid'))
    tasks = lut.get_unique(out_data, s.ix('task'))


    train_trials = data[:, r.ix('stage')]==0
    data = data[train_trials, :]

    x1 = np.arange(0, 15).reshape([-1, 1])
    x0 = np.ones_like(x1)
    for sid in sids:
        for i, tsk in enumerate(tasks):
            ii = sid * 4 + i
            # select one subject and one task
            mask = np.all([data[:, r.ix('sid')] == sid, data[:, r.ix('cat')] == tsk], axis=0)
            hits = data[mask, r.ix('cor')]
            cs = np.cumsum(hits)
            pc = cs[:-1] / np.arange(1, 15)

            # LP variant 1
            pc1 = np.mean(pc[:5])  # mean PC across trials 2 through 6
            pc2 = np.mean(pc[-5:])  # mean PC across trials 11 through 15
            out_data[ii, s.ix('lp1')] = pc1 / pc2

            # LP variant 2
            out_data[ii, s.ix('lp2')] = pc2 - pc1

            # LP variant 3
            pc1, pc2 = pc[:-1], pc[1:]
            q = np.divide(pc1, pc2, out=np.zeros_like(pc1), where=pc1 != 0)
            out_data[ii, s.ix('lp3')] = np.mean(q)

            # LP variant 4
            Y = hits.reshape([-1, 1])
            X = np.concatenate([x0, x1], axis=1)
            linmodel = sm.OLS(Y, X)
            out_data[ii, s.ix('lp4')] = linmodel.fit().params[1]

            # LP variant 5
            out_data[ii, s.ix('lp5')] = np.sum(hits[-5:]) - np.sum(hits[:5])

    if ARGS.output_df: lut.print_arr(out_data, s.cols, nonints=['pc'])
    if ARGS.save_to: lut.dopickle(ARGS.save_to, out_data)
    if ARGS.save_raw: lut.dopickle('pickled_data/combined.pkl', data)


if __name__=='__main__': main()