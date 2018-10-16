import loc_utils as lut
import pandas as pd
import numpy as np

tasks = [1, 2, 3, 4]
mcols = 'sid,grp,stage,trial,blkt,current,nxt,pc:1,pc:2,pc:3,pc:4,p:1,p:2,p:3,p:4,sc:1,sc:2,sc:3,sc:4,sw_pred,sw_act,sw_lag,relt:1,relt:2,relt:3,relt:4'.split(',')
ix = mcols.index

mdata = lut.unpickle('../pipeline_data/scdata/modeling_data_sw_lag.pkl')

sids, groups = lut.get_unique(mdata, [ix('sid'), ix('grp')])
rt = np.zeros([mdata.shape[0], 4])

for j, tsk in enumerate(tasks):
    tmask = lut.get_mask(mdata, {ix('current'): tsk})
    rt[:-1][tmask[1:], j] = 1

for sid in sids:
    smask = lut.get_mask(mdata, {ix('sid'): sid})
    srt = rt[smask, :]
    srt[0, :] = 15
    srt = np.cumsum(srt, axis=0)
    rt[smask] = srt

rt = np.transpose(rt.T / np.sum(rt, axis=1))
mdata = np.concatenate([mdata, rt], axis=1)

df1 = pd.DataFrame(mdata, columns=mcols)
neworder = 'sid,grp,stage,trial,blkt,current,nxt,pc:1,pc:2,pc:3,pc:4,p:1,p:2,p:3,p:4,sc:1,sc:2,sc:3,sc:4,relt:1,relt:2,relt:3,relt:4,sw_pred,sw_act,sw_lag'.split(',')
df1 = df1[neworder]

xdata = lut.unpickle('../pipeline_data/selection_rates/data_v3.pkl')

A  = ['sid', 'grp']
B  = ['ps:{}'.format(t) for t in tasks]
C  = ['pct:{}'.format(t) for t in tasks]
D  = ['pcf:{}'.format(t) for t in tasks]
D1 = ['pctst:{}'.format(t) for t in tasks]
E  = ['lrn:{}'.format(t) for t in tasks]
F  = ['int:{}'.format(t) for t in tasks]
G  = ['comp:{}'.format(t) for t in tasks]
H  = ['time:{}'.format(t) for t in tasks]
I  = ['prog:{}'.format(t) for t in tasks]
J  = ['rule:{}'.format(t) for t in tasks]
K  = ['lrn2:{}'.format(t) for t in tasks]
L  = ['alv']

allcols = [*A, *B, *C, *D, *D1, *E, *F, *G, *H, *I, *J, *K, *L]
xcols = [*A, *E, *F, *G, *H, *I, *J, *K]
colinds = [allcols.index(cname) for cname in xcols]
df2 = pd.DataFrame(xdata[:, colinds], columns=xcols)


df1.to_csv('/Users/alexten/Projects/Exploration/R_docs/cmnr_mdata.csv')
df2.to_csv('/Users/alexten/Projects/Exploration/R_docs/cmnr_xdata.csv')