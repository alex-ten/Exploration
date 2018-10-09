import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
import seaborn as sns

import numpy as np
import os

import loc_utils as lut
import vis_utils as vut
from standards import *

rx = RAWXix()
r = RAWix()

colors = ['#008fd5', '#fc4f30', '#e5ae38', '#6d904f']

glabels = {0: 'F', 1: 'S'}
clabels = {0: 'i-', 1: 'i+'}

def gclabel(g, c):
    return '{}/{}'.format(glabels[g], clabels[c])

tlabels = {
        1: '1D',
        2: 'I1D',
        3: '2D',
        4: 'R'}

saveloc = '/Users/alexten/Projects/HFSP/img'
desktop = '/Users/alexten/Desktop/'

def individual_sc(path, sid, save=True):
    cols = 'sid,grp,stage,trial,t0,t1,pc1,pc2,pc3,pc4,p1,p2,p3,p4,sc,switch'.split(',')
    ix = cols.index
    data = lut.unpickle(path)
    
    mask = lut.get_mask(data, {ix('sid'): sid})
    data = data[mask, :]
    
    grp = data[0, ix('grp')]
    figname = 'g{}s{:03}'.format(int(grp), sid)
    fig = plt.figure(figname, figsize=[6,3])
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 320)
    ax.set_ylim(-.75, 1.02)

    ax.grid(True)
    ax.axvline(60, c='k')
    for ti, tsk in enumerate([1,2,3,4]):
        tmask = lut.get_mask(data, {ix('t0'): tsk})
                
        trials = data[tmask, ix('trial')]
        ps = data[tmask, ix('p{}'.format(tsk))]
        nans = np.full([tmask.size,], np.nan)
        nans[tmask] = ps
        
        pcs = data[:, ix('pc{}'.format(tsk))]
        
        ax.plot(np.arange(60, 60+pcs.size), pcs, c=colors[ti], lw=2, alpha=.7,
               label=tlabels[tsk])
        ax.plot(np.arange(60, 60+pcs.size), nans, c=colors[ti], ls='--', alpha=.7)
    
    l = ['stay', 'switch']
    s = ['-', '--']
    for swt in [0,1]:
        mask = lut.get_mask(data, {ix('switch'): swt})
        nans = np.full([mask.size,], np.nan)
        nans[mask] = data[mask, ix('sc')]
        
        ax.plot(np.arange(60, 60+pcs.size), nans, ls=s[swt], lw=1, label=l[swt], c='k')
    
    ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    fig.tight_layout()
    
    if save:
        s = desktop+'ind_sc/{}.{}'.format(figname, 'svg')
        fig.savefig(s, format='svg', dpi=100)
        os.system('scour -i {} -o {}'.format(s, s.replace('ind_sc', 'ind_sc_compressed')))
        

if __name__=='__main__':
    for i in range(400):
        try:
            individual_sc('pipeline_data/scdata/joint_data5.pkl', i, save=True)
            print(i)
        except IndexError:
            continue
    