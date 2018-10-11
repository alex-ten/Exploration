import loc_utils as lut
import pandas as pd

tasks = [1, 2, 3, 4]
mcols = 'sid,grp,stage,trial,blkt,t0,t1,pc1,pc2,pc3,pc4,p1,p2,p3,p4,sc1,sc2,sc3,sc4,sw_pred,sw_act,sw_lag'.split(',')

mdata = lut.unpickle('../pipeline_data/scdata/modeling_data_sw_lag.pkl')
df1 = pd.DataFrame(mdata, columns=mcols)


xdata = lut.unpickle('../pipeline_data/selection_rates/data_v3.pkl')

A = ['sid', 'grp']
B = ['ps_{}'.format(t) for t in tasks]
C = ['pct_{}'.format(t) for t in tasks]
D = ['pcf_{}'.format(t) for t in tasks]
D1 = ['pctst_{}'.format(t) for t in tasks]
E = ['lrn_{}'.format(t) for t in tasks]
F = ['int_{}'.format(t) for t in tasks]
G = ['comp_{}'.format(t) for t in tasks]
H = ['time_{}'.format(t) for t in tasks]
I = ['prog_{}'.format(t) for t in tasks]
J = ['rule_{}'.format(t) for t in tasks]
K = ['lrn2_{}'.format(t) for t in tasks]
L = ['alv']

allcols = [*A, *B, *C, *D, *D1, *E, *F, *G, *H, *I, *J, *K, *L]
xcols = [*A, *E, *F, *G, *H, *I, *J, *K]
colinds = [allcols.index(cname) for cname in xcols]
df2 = pd.DataFrame(xdata[:, colinds], columns=xcols)


df1.to_csv('/Users/alexten/Projects/Exploration/R_docs/mnr_mdata.csv')
df2.to_csv('/Users/alexten/Projects/Exploration/R_docs/mnr_xdata.csv')