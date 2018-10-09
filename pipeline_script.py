import os.path as osp

from s0_preprocess_raw import preprocess_raw_main, preprocess_raw_extra
from s1_csv_to_pickle import csv_to_pickle
from s2_join_groups import join_groups
from s3_remove_outliers import filter_outliers

STAGES = list('3')
PARENT_DIR = 'pipeline_data'

if '0' in STAGES:
    raw_paths = {
        'main_free': 'monster task data/free_trained/monsters_data_free_familiarize_05232017.csv',
        'main_strat': 'monster task data/strategic/monsters_data_strategic_052217.csv',
        'extra_free': 'monster task data/free_trained/monsters_extra_data_free_familiarize_05232017.csv',
        'extra_strat': 'monster task data/strategic/monsters_extra_data_strategic_052217.csv'
    }

    pdir = PARENT_DIR
    sdir = 's0'
    main_free_prep, free_map = preprocess_raw_main(path=raw_paths['main_free'],
                                                   save_to=osp.join(pdir, sdir, 'main_free.csv'))
    main_strat_prep, strat_map = preprocess_raw_main(path=raw_paths['main_strat'],
                                                     save_to=osp.join(pdir, sdir, 'main_strat.csv'))

    extra_free_prep = preprocess_raw_extra(path=raw_paths['extra_free'],
                                           save_to=osp.join(pdir, sdir, 'extra_free.csv'),
                                           id_map=free_map)
    extra_strat_prep = preprocess_raw_extra(path=raw_paths['extra_strat'],
                                            save_to=osp.join(pdir, sdir, 'extra_strat.csv'),
                                            id_map=strat_map)
else:
    main_free_prep = osp.join(PARENT_DIR, 's0', 'main_free.csv')
    main_strat_prep = osp.join(PARENT_DIR, 's0', 'main_strat.csv')
    extra_free_prep = osp.join(PARENT_DIR, 's0', 'main_strat.csv')
    extra_strat_prep = osp.join(PARENT_DIR, 's0', 'extra_strat.csv')


if '1' in STAGES:
    prep_paths = main_free_prep, main_strat_prep, extra_free_prep, extra_strat_prep
    pdir = PARENT_DIR
    sdir = 's1'
    for path in prep_paths:
        csv_to_pickle(path, save_to=osp.join(pdir, sdir, osp.splitext(osp.basename(path))[0]+'.pkl'))


if '2' in STAGES:
    fdict = {
        'main_free': 'main_free.pkl',
        'main_strat': 'main_strat.pkl',
        'extra_free': 'extra_free.pkl',
        'extra_strat': 'extra_strat.pkl'
    }
    pdir = PARENT_DIR
    ddir = 's1'
    sdir = 's2'
    filename = 'joint_data.pkl'
    join_groups(data_dir=osp.join(pdir, ddir),
                files_dict=fdict,
                save_to=osp.join(pdir, sdir, filename))


if '3' in STAGES:
    pdir = PARENT_DIR
    ddir = 's2'
    sdir = 's3'
    filename = 'joint_data.pkl'
    path = osp.join(pdir, ddir, filename)
    save_to = osp.join(pdir, sdir, filename)
    filter_outliers(path_to_data=path,
                    save_to=save_to,
                    extreme_sticking=True,
                    choice_bias_criterion='sd',
                    choice_bias_critval=2,
                    alloc_var_crit=100,
                    assign_new_ids=True)