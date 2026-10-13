"""Supporting analyses: replicate ensembles, fine r-scan, noise, binning, weighting,
two-band check at r=3.6, floating-point collapse checks, correlation times."""
import os
import numpy as np, pickle, time, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nfi_core import *
from scipy import stats
t0 = time.time(); out = {}
MG = [50, 100, 200, 400, 800, 1200, 1600]

def pmstep(z): return lambda x: np.mod(x + x ** z, 1.0)
def logstep(r): return lambda x: r * x * (1 - x)

# 1. replicate ensembles (N = 1e4, independent seeds) at m* = 800
rep = {}
for name, step, burn in [('pm1.3', pmstep(1.3), 2000), ('pm1.2', pmstep(1.2), 2000), ('log4', logstep(4.0), 800),
                         ('hen14_01', None, 1500), ('pm1.5', pmstep(1.5), 2000)]:
    rows = []
    for s in range(20):
        rng = np.random.default_rng(1000 + s)
        if name == 'hen14_01':
            G = run_henon(1.4, 0.1, 10000, 800, rng=rng); S = G.sum(axis=1)
        elif name == 'log4':
            S = stream_logistic(4.0, rng.uniform(0.01, 0.99, 10000), 800, [800], rng)[800]
        else:
            S = stream_sums(step, rng.uniform(0.01, 0.99, 10000), burn, [800])[800]
        Z = standardize(S); st = shape_stats(Z); null = gaussian_null(10000)
        st['kl_exc'] = st['kl'] - float(np.mean(null['kl'])); st['p'] = null_pvalues(st, null)
        st['q_lo'] = float(np.mean(Z < -3)); st['q_hi'] = float(np.mean(Z > 3))
        st['bowley'] = float((np.percentile(Z, 75) + np.percentile(Z, 25) - 2 * np.median(Z)) / (np.percentile(Z, 75) - np.percentile(Z, 25)))
        rows.append(st)
    rep[name] = rows
    def med(k): return np.median([r[k] for r in rows])
    def iqr(k): return np.percentile([r[k] for r in rows], [25, 75])
    print('rep', name, {k: (round(med(k), 4), np.round(iqr(k), 4).tolist()) for k in ['kl_exc', 'ad', 'w1', 'skew', 'kurt', 'bowley', 'q_lo', 'q_hi']},
          'frac p_ad<0.01=', np.mean([r['p']['ad'] < 0.01 for r in rows]), 'frac p_kl<0.01=', np.mean([r['p']['kl'] < 0.01 for r in rows]), round(time.time() - t0), flush=True)
out['rep'] = rep

# 2. fine r-scan (dr = 0.001), N = 2000
rs = np.round(np.arange(3.500, 4.0005, 0.001), 3); scan = []
for r in rs:
    rng = np.random.default_rng(int(r * 1e4))
    x = rng.uniform(0.01, 0.99, 2000)
    for _ in range(800):
        x = r * x * (1 - x); bad = (x <= 0) | (x >= 1)
        if bad.any(): x[bad] = rng.uniform(0.01, 0.99, bad.sum())
    S = np.zeros(2000); lam = np.zeros(2000); Sm = {}; okm = np.ones(2000, bool)
    for k in range(1, 1601):
        x = r * x * (1 - x); okm &= (x > 0) & (x < 1); S += x; lam += np.log(np.abs(r * (1 - 2 * x)) + 1e-300)
        if k in MG: Sm[k] = S.copy()
    Sm = {k_: v_[okm] for k_, v_ in Sm.items()}
    lam = float(np.mean(lam / 1600))
    v = np.array([np.var(Sm[m], ddof=1) for m in MG]); a, R2, spr = fit_alpha(MG, v)
    Z = standardize(Sm[800])
    row = dict(r=r, lam=lam, alpha=a, R2=R2, spread=spr, D=v[4] / 800)
    if Z is None or lam < 0:
        row.update(periodic=True)
    else:
        st = shape_stats(Z); null = gaussian_null(2000)
        row.update(periodic=False, kl=st['kl'], kl_exc=st['kl'] - float(np.mean(null['kl'])), skew=st['skew'], kurt=st['kurt'], ad=st['ad'], p_ad=null_pvalues(st, null)['ad'])
    scan.append(row)
out['scan'] = scan
print('scan done', len(scan), 'periodic:', sum(r_['periodic'] for r_ in scan), round(time.time() - t0), flush=True)

# 3. observational noise (added to the observable)
noise = {}
eps_list = [0, 0.001, 0.005, 0.01, 0.05, 0.1]
for name, kind, p in [('log3.6', 'log', 3.6), ('log4.0', 'log', 4.0), ('pm1.3', 'pm', 1.3)]:
    rng = np.random.default_rng(55)
    G = run_logistic(p, 10000, 1600, rng=rng)[0] if kind == 'log' else run_pm(p, 10000, 1600, rng=rng)[0]
    rows = []
    for eps in eps_list:
        Gn = G + eps * np.random.default_rng(7).standard_normal(G.shape)
        d = diagnose(Gn, MG, 800, n_boot=100, null_reps=1000)
        rows.append(dict(eps=eps, alpha=d['alpha'], se_alpha=d['se_alpha'], kl_exc=d['kl_exc'], se_kl=d['se_kl'], skew=d['skew'], kurt=d['kurt'], ad=d['ad']))
    noise[name] = rows
    print('noise', name, [(r_['eps'], round(r_['alpha'], 3), round(r_['kl_exc'], 4), round(r_['skew'], 3)) for r_ in rows], round(time.time() - t0), flush=True)
    del G
out['noise'] = noise

# 4. binning: raw and null-corrected KL vs B
Blist = [20, 30, 40, 60, 80, 100, 130, 160, 200]
bins = {}
for name, kind, p in [('log3.6', 'log', 3.6), ('log4.0', 'log', 4.0), ('hen1.4', 'hen', (1.4, 0.3)), ('pm1.3', 'pm', 1.3), ('pm1.5', 'pm', 1.5)]:
    rng = np.random.default_rng(77)
    if kind == 'log': S = stream_logistic(p, rng.uniform(0.01, 0.99, 10000), 800, [800], rng)[800]
    elif kind == 'pm': S = stream_sums(pmstep(p), rng.uniform(0.01, 0.99, 10000), 2000, [800])[800]
    else: S = run_henon(*p, 10000, 800, rng=rng).sum(axis=1)
    Z = standardize(S)
    raw = [kl_hist(Z, B) for B in Blist]
    exc = [kl_hist(Z, B) - float(np.mean(gaussian_null(10000, B, 500)['kl'])) for B in Blist]
    bins[name] = dict(raw=raw, exc=exc)
    print('bins', name, np.round(raw, 4), np.round(exc, 4), flush=True)
bins['null'] = {B: (float(np.mean(gaussian_null(10000, B, 500)['kl'])), float(np.std(gaussian_null(10000, B, 500)['kl']))) for B in Blist}
bins['null2000'] = {B: float(np.mean(gaussian_null(2000, B, 500)['kl'])) for B in Blist}
bins['B'] = Blist
out['bins'] = bins

# 5. two-band check at r = 3.6: Var(S_m) for every m, fit Var = A + 2 D m on even m
rng = np.random.default_rng(36)
G = run_logistic(3.6, 10000, 12800, rng=rng)[0][:, :]
C = np.cumsum(G - G.mean(), axis=1); vall = np.var(C, axis=0, ddof=1); del C, G
m = np.arange(1, 12801); ev = (m % 2 == 0) & (m >= 50)
A, B2 = np.polyfit(m[ev], vall[ev], 1)[::-1]
out['band'] = dict(var=vall, A=A, slope=B2)
lm = np.array([50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]); ls = np.diff(np.log(vall[lm - 1])) / np.diff(np.log(lm))
print('band r=3.6: A=%.4g 2D=%.4g ; local slopes even grid' % (A, B2), np.round(ls, 3), flush=True)

# 6. floating-point collapse onto x=0 (logistic r=4) and PM
col = {}
for name, step, burn in [('log4', logstep(4.0), 800), ('pm1.3', pmstep(1.3), 2000), ('pm2.5', pmstep(2.5), 2000)]:
    rng = np.random.default_rng(3); x = rng.uniform(0.01, 0.99, 10000)
    first = np.full(10000, -1)
    for k in range(burn + 12800):
        x = step(x)
        hit = (x == 0) & (first < 0); first[hit] = k
    col[name] = dict(n_by_2400=int(np.sum((first >= 0) & (first < 2400))), n_by_end=int(np.sum(first >= 0)))
    print('collapse', name, col[name], flush=True)
out['collapse'] = col

# 7. correlation diagnostics (N = 2000): first lag with |rho| envelope < 0.05, and statistical inefficiency at m*
cor = {}
for name, kind, p in [('log3.6', 'log', 3.6), ('log3.8', 'log', 3.8), ('log4.0', 'log', 4.0), ('hen1.4_0.3', 'hen', (1.4, 0.3)), ('hen1.4_0.1', 'hen', (1.4, 0.1)),
                      ('pm1.2', 'pm', 1.2), ('pm1.3', 'pm', 1.3), ('pm1.5', 'pm', 1.5), ('pm1.8', 'pm', 1.8), ('pm2.0', 'pm', 2.0), ('pm2.5', 'pm', 2.5)]:
    rng = np.random.default_rng(11)
    if kind == 'log': G = run_logistic(p, 2000, 4000, rng=rng)[0]
    elif kind == 'pm': G = run_pm(p, 2000, 4000, rng=rng)[0]
    else: G = run_henon(*p, 2000, 4000, rng=rng)
    c = G - G.mean(); c0 = np.mean(c * c)
    lags = np.unique(np.round(np.logspace(0, np.log10(2000), 60)).astype(int))
    rho = np.array([np.mean(c[:, :-l] * c[:, l:]) / c0 for l in lags])
    above = np.where(np.abs(rho) >= 0.05)[0]
    tau = int(lags[above[-1] + 1]) if len(above) and above[-1] + 1 < len(lags) else (int(lags[0]) if not len(above) else -1)
    S800 = c[:, :800].sum(axis=1); ineff = float(np.var(S800, ddof=1) / (800 * c0))
    cor[name] = dict(lags=lags, rho=rho, tau05=tau, ineff=ineff)
    print('corr', name, 'tau05=', tau, 'ineff=%.2f' % ineff, flush=True)
    del G, c
out['corr'] = cor
pickle.dump(out, open('data/extra.pkl', 'wb'))
print('DONE', round(time.time() - t0))
