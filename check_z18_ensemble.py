import sys, numpy as np, time
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nfi_core import stream_sums, fit_alpha
MG=[50,100,200,400,800,1200,1600]
res={}
for N in [2000,10000,40000,160000,400000]:
    al=[]
    nrep = 5 if N<=40000 else 2
    for s in range(nrep):
        rng=np.random.default_rng(500+s)
        S=stream_sums(lambda x: np.mod(x+x**1.8,1.0), rng.uniform(0.01,0.99,N), 2000, MG)
        v=[np.var(S[m],ddof=1) for m in MG]; al.append(fit_alpha(MG,v)[0])
    res[N]=al; print(N, np.round(al,3), 'mean %.3f'%np.mean(al), flush=True)
import pickle; pickle.dump(res,open('data/z18_ens.pkl','wb'))
