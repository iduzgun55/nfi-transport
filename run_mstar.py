"""Finite-time (m) dependence of scale and shape diagnostics up to m = 12800
(reviewer 2, point 5), plus burn-in and ensemble-size checks."""
import os
import numpy as np, pickle, time, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nfi_core import *

NT = 10000
MS = np.unique(np.round(25 * 2 ** (np.arange(0, 19) / 2)).astype(int))   # 25 ... 12800 (factor sqrt2)
MS = MS[MS <= 12800]
out = {}
t0 = time.time()

def shape_vs_m(S, nboot=200, seed=1):
    rows = {}
    rng = np.random.default_rng(seed)
    for m, s in S.items():
        Z = standardize(s)
        if Z is None:
            continue
        st = shape_stats(Z); null = gaussian_null(len(Z))
        st['kl_exc'] = st['kl'] - float(np.mean(null['kl'])); st['p'] = null_pvalues(st, null)
        bs = {k: [] for k in ('skew', 'kurt', 'kl')}
        for _ in range(nboot):
            idx = rng.integers(0, len(s), len(s)); sb = shape_stats(standardize(s[idx]))
            for k in bs: bs[k].append(sb[k])
        for k in bs: st['se_' + k] = float(np.std(bs[k], ddof=1))
        rows[m] = st
    return rows

def var_boot(S, nboot=200, seed=2):
    ms = np.array(sorted(S)); rng = np.random.default_rng(seed); N = len(S[ms[0]])
    v = np.array([np.var(S[m], ddof=1) for m in ms])
    vb = np.empty((nboot, len(ms)))
    for b in range(nboot):
        idx = rng.integers(0, N, N); vb[b] = [np.var(S[m][idx], ddof=1) for m in ms]
    return ms, v, vb

cases = [('pm', 1.2), ('pm', 1.3), ('pm', 1.5), ('pm', 1.8), ('pm', 2.0), ('log', 3.6), ('log', 4.0)]
for kind, p in cases:
    rng = np.random.default_rng(777 + int(p * 100))
    x0 = rng.uniform(0.01, 0.99, NT)
    if kind == 'pm':
        step = (lambda z: (lambda x: np.mod(x + x ** z, 1.0)))(p); burn = 2000
    else:
        step = None; burn = 800
    S = stream_logistic(p, x0, burn, MS, rng) if kind == 'log' else stream_sums(step, x0, burn, MS)
    ms, v, vb = var_boot(S)
    out[(kind, p)] = dict(ms=ms, var=v, var_boot=vb, shape=shape_vs_m(S))
    lm, ls = local_slopes(ms, v)
    print(kind, p, 'local slopes:', np.round(ls, 3), round(time.time() - t0), flush=True)
    sh = out[(kind, p)]['shape']
    print('   skew:', [round(sh[m]['skew'], 2) for m in ms if m in sh])
    print('   kurt:', [round(sh[m]['kurt'], 2) for m in ms if m in sh])
    print('   klexc:', [round(sh[m]['kl_exc'], 3) for m in ms if m in sh], flush=True)

# burn-in sensitivity (m* = 800, m-grid 50..1600)
MG = [50, 100, 200, 400, 800, 1200, 1600]
bi = {}
for kind, p, burns in [('pm', 1.3, [2000, 20000]), ('pm', 1.5, [2000, 20000]), ('log', 4.0, [800, 8000]), ('log', 3.6, [800, 8000])]:
    for burn in burns:
        rng = np.random.default_rng(99)
        x0 = rng.uniform(0.01, 0.99, NT)
        S = stream_sums((lambda z: (lambda x: np.mod(x + x ** z, 1.0)))(p), x0, burn, MG) if kind == 'pm' else stream_logistic(p, x0, burn, MG, rng)
        v = [np.var(S[m], ddof=1) for m in MG]; a, R2, _ = fit_alpha(MG, v)
        st = shape_stats(standardize(S[800]))
        bi[(kind, p, burn)] = dict(alpha=a, R2=R2, **st, kl_exc=st['kl'] - float(np.mean(gaussian_null(NT)['kl'])))
        print('burn', kind, p, burn, {k: round(v_, 4) for k, v_ in bi[(kind, p, burn)].items()}, flush=True)
out['burn'] = bi

# ensemble-size check for z = 1.3
es = {}
for N in [2000, 10000, 40000]:
    rng = np.random.default_rng(31)
    S = stream_sums(lambda x: np.mod(x + x ** 1.3, 1.0), rng.uniform(0.01, 0.99, N), 2000, [800])
    st = shape_stats(standardize(S[800])); null = gaussian_null(N)
    st['kl_exc'] = st['kl'] - float(np.mean(null['kl'])); st['p'] = null_pvalues(st, null)
    es[N] = st
    print('ens z=1.3 N=', N, {k: (round(v_, 4) if isinstance(v_, float) else v_) for k, v_ in st.items()}, flush=True)
out['ens'] = es
pickle.dump(out, open('data/mstar.pkl', 'wb'))
print('DONE', round(time.time() - t0))
