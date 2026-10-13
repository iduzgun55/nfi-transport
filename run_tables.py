"""Tables I-IV of the revised manuscript (N_traj = 10^4, 500 bootstrap resamples)."""
import os
import numpy as np, pickle, json, time, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nfi_core import *

NT = 10000; MG = [50, 100, 200, 400, 800, 1200, 1600]; MSTAR = 800; NB = 500
rng = np.random.default_rng(42)
R = {}

def strip(d):
    return {k: v for k, v in d.items() if k != 'Z'}

t0 = time.time()
# ---------------- logistic ----------------
for r in [3.50, 3.60, 3.80, 4.00]:
    for g in ['x', 'x2']:
        G, lam = run_logistic(r, NT, 1600, burn=800, rng=np.random.default_rng(int(r * 1000)), g=g)
        nd = run_logistic.n_dropped
        d = diagnose(G, MG, MSTAR, n_boot=NB)
        d['lam'] = lam; d['n_dropped'] = nd
        d['tau'], d['tau_lag'] = tau_int(G[:2000], 400)
        d['var_all'] = np.var(np.cumsum(G - G.mean(), axis=1), axis=0, ddof=1)  # Var(S_m) for every m
        R[('log', r, g)] = d
        print('log', r, g, 'lam=%.3f a=%.3f R2=%.4f spr=%.2f' % (lam, d['alpha'], d['R2'], d['loc_spread']),
              '' if d['degenerate'] else 'KL=%.4f exc=%.4f skew=%.3f kurt=%.3f p=%s' % (d['kl'], d['kl_exc'], d['skew'], d['kurt'], d['p']),
              'tau=%.1f' % d['tau'], round(time.time() - t0), flush=True)
        del G
# ---------------- Henon ----------------
for (a, b, g) in [(1.4, 0.3, 'x'), (1.4, 0.3, 'x2'), (1.4, 0.1, 'x'), (1.4, 0.1, 'x2'), (1.3, 0.3, 'x'), (1.3, 0.3, 'x2')]:
    G = run_henon(a, b, NT, 1600, burn=1500, rng=np.random.default_rng(int(a * 100 + b * 10)), g=g)
    d = diagnose(G, MG, MSTAR, n_boot=NB)
    d['lam'], d['se_lam'] = henon_lyap(a, b)
    d['tau'], d['tau_lag'] = tau_int(G[:2000], 400)
    d['nper'] = int(len(np.unique(np.round(G[0, -400:], 8))))
    d['var_all'] = np.var(np.cumsum(G - G.mean(), axis=1), axis=0, ddof=1)
    R[('hen', a, b, g)] = d
    print('hen', a, b, g, 'lam=%.3f a=%.3f R2=%.4f spr=%.2f distinct=%d' % (d['lam'], d['alpha'], d['R2'], d['loc_spread'], d['nper']),
          '' if d['degenerate'] else 'KL=%.4f exc=%.4f skew=%.3f kurt=%.3f p=%s' % (d['kl'], d['kl_exc'], d['skew'], d['kurt'], d['p']),
          round(time.time() - t0), flush=True)
    del G
# ---------------- PM ----------------
for z in [1.2, 1.3, 1.5, 1.8, 2.0, 2.5]:
    G, lam = run_pm(z, NT, 1600, burn=2000, rng=np.random.default_rng(int(z * 1000)))
    d = diagnose(G, MG, MSTAR, n_boot=NB)
    d['lam'] = lam
    d['tau'], d['tau_lag'] = tau_int(G[:2000], 1500)
    d['var_all'] = np.var(np.cumsum(G - G.mean(), axis=1), axis=0, ddof=1)
    R[('pm', z)] = d
    print('pm', z, 'lam=%.3f a=%.3f [%.3f,%.3f] R2=%.4f spr=%.2f' % (lam, d['alpha'], *d['ci_alpha'], d['R2'], d['loc_spread']),
          'KL=%.4f exc=%.4f [%.4f,%.4f] skew=%.3f+-%.3f kurt=%.3f+-%.3f W1=%.4f AD=%.2f p=%s' % (
              d['kl'], d['kl_exc'], *d['ci_kl_exc'], d['skew'], d['se_skew'], d['kurt'], d['se_kurt'], d['w1'], d['ad'], d['p']),
          'tau=%.1f' % d['tau'], round(time.time() - t0), flush=True)
    del G

pickle.dump(R, open('data/tables.pkl', 'wb'))
print('DONE', round(time.time() - t0))
