"""
Bootstrap SE computation for all tables in the revised manuscript.
Covers: Logistic (Table I), Henon (Table II), PM (Table III) including z=1.30.
"""
import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings; warnings.filterwarnings('ignore')
np.random.seed(99)

# ---- Maps ----
def logistic_map(r, x0, N, burn=800):
    x = np.empty(burn+N); x[0]=x0
    for i in range(1, burn+N): x[i]=r*x[i-1]*(1-x[i-1])
    return x[burn:]

def henon_map(a, b, x0, y0, N, burn=1500):
    x=np.empty(burn+N); y=np.empty(burn+N); x[0],y[0]=x0,y0
    for i in range(1,burn+N):
        x[i]=1-a*x[i-1]**2+y[i-1]; y[i]=b*x[i-1]
    return x[burn:], y[burn:]

def pomeau_manneville(z, x0, N, burn=2000):
    x=np.empty(burn+N); x[0]=x0
    for i in range(1,burn+N): x[i]=(x[i-1]+x[i-1]**z)%1.0
    return x[burn:]

# ---- Core metrics ----
mg = np.array([50,100,200,400,800,1200,1600])
ms = 800
nb = 80

def compute_var(trajs, m_vals):
    c = trajs - np.mean(trajs)
    v = np.zeros(len(m_vals))
    for i,m in enumerate(m_vals):
        if m>trajs.shape[1]: v[i]=np.nan; continue
        Sm = np.sum(c[:,:m], axis=1); v[i]=np.var(Sm, ddof=1)
    return v

def fit_alpha(m_vals, v):
    ok = (v>0) & np.isfinite(v)
    if ok.sum()<3: return 0., 0.
    s,_,_,_,_ = stats.linregress(np.log(m_vals[ok]), np.log(v[ok]))
    return s, 0.

def compute_Zm(trajs, m):
    c = trajs - np.mean(trajs)
    Sm = np.sum(c[:,:m], axis=1)
    s = np.std(Sm, ddof=1)
    return Sm/s if s>1e-15 else None

def kl_hist(Zm, nb=80):
    if Zm is None: return np.nan
    h,e = np.histogram(Zm, bins=nb, range=(-5,5))
    h = h+1; p = h/h.sum()
    bc = 0.5*(e[:-1]+e[1:]); bw = e[1]-e[0]
    q = stats.norm.pdf(bc)*bw; q = q/q.sum()
    return sum(p[i]*np.log(p[i]/q[i]) for i in range(len(p)) if p[i]>0 and q[i]>0)

def compute_all(trajs, m_vals=mg, mstar=ms):
    v = compute_var(trajs, m_vals)
    a, _ = fit_alpha(m_vals, v)
    Zm = compute_Zm(trajs, mstar)
    kl = kl_hist(Zm)
    sk = float(skew(Zm)) if Zm is not None else np.nan
    ku = float(kurtosis(Zm)) if Zm is not None else np.nan  # excess kurtosis
    return a, kl, sk, ku

def bootstrap_se(trajs, n_boot=60, m_vals=mg, mstar=ms, pm_mg=None):
    """Bootstrap SE for alpha, KL, skewness, excess kurtosis."""
    if pm_mg is not None:
        m_vals = pm_mg
    Nt = trajs.shape[0]
    boot_a = np.zeros(n_boot)
    boot_kl = np.zeros(n_boot)
    boot_sk = np.zeros(n_boot)
    boot_ku = np.zeros(n_boot)
    for b in range(n_boot):
        idx = np.random.choice(Nt, Nt, replace=True)
        t_b = trajs[idx]
        a_b, kl_b, sk_b, ku_b = compute_all(t_b, m_vals, mstar)
        boot_a[b]=a_b; boot_kl[b]=kl_b
        boot_sk[b]=sk_b; boot_ku[b]=ku_b
    return (np.std(boot_a, ddof=1), np.std(boot_kl, ddof=1),
            np.std(boot_sk, ddof=1), np.std(boot_ku, ddof=1))

# ============================================================
# 1. LOGISTIC MAP  (60 bootstrap resamples)
# ============================================================
print("="*60)
print("TABLE I — LOGISTIC MAP")
print("="*60)
Nt_log = 2000; Ns = 2000
rvals = [3.50, 3.60, 3.80, 4.00]
regimes = {3.50:'periodic', 3.60:'weak chaos', 3.80:'chaotic', 4.00:'fully chaotic'}
gx  = lambda t: t
gx2 = lambda t: t**2

for r in rvals:
    print(f"\n  Generating logistic r={r} trajectories...")
    trajs = np.zeros((Nt_log, Ns))
    for i in range(Nt_log):
        trajs[i] = logistic_map(r, np.random.uniform(0.01,0.99), Ns)

    for gname, gfunc in [('x', gx), ('x^2', gx2)]:
        t = gfunc(trajs)
        a, kl, sk, ku = compute_all(t)
        if r == 3.50:
            print(f"  r={r} g={gname}: DEGENERATE (periodic) — all metrics undefined")
            continue
        se_a, se_kl, se_sk, se_ku = bootstrap_se(t, n_boot=60)
        nfi_sc = abs(a-1)
        print(f"  r={r} g={gname} [{regimes[r]}]:")
        print(f"    alpha   = {a:.4f}  SE={se_a:.4f}")
        print(f"    KL      = {kl:.4f}  SE={se_kl:.4f}")
        print(f"    NFIsc   = {nfi_sc:.4f}  SE={se_a:.4f}")
        print(f"    skew    = {sk:.3f}   SE={se_sk:.4f}")
        print(f"    kurt    = {ku:.3f}   SE={se_ku:.4f}")

# ============================================================
# 2. HÉNON MAP  (30 bootstrap resamples)
# ============================================================
print("\n" + "="*60)
print("TABLE II — HÉNON MAP")
print("="*60)
Nt_h = 1500
henon_params = [(1.4,0.3,'chaotic'), (1.4,0.1,'Fickian'), (1.3,0.3,'ordered')]

for a_h, b_h, label in henon_params:
    print(f"\n  Generating Hénon ({a_h},{b_h}) trajectories...")
    xs = []; ok=0
    while ok < Nt_h:
        try:
            xh, _ = henon_map(a_h, b_h,
                              np.random.uniform(-0.5,0.5),
                              np.random.uniform(-0.5,0.5), Ns)
            if np.all(np.abs(xh)<100): xs.append(xh); ok+=1
        except: pass
    trajs = np.array(xs)

    for gname, gfunc in [('x', gx)] + ([('x^2', gx2)] if (a_h,b_h)==(1.4,0.3) else []):
        t = gfunc(trajs)
        a, kl, sk, ku = compute_all(t)
        se_a, se_kl, se_sk, se_ku = bootstrap_se(t, n_boot=30)
        nfi_sc = abs(a-1)
        print(f"  ({a_h},{b_h}) g={gname} [{label}]:")
        print(f"    alpha   = {a:.4f}  SE={se_a:.4f}")
        print(f"    KL      = {kl:.4f}  SE={se_kl:.4f}")
        print(f"    NFIsc   = {nfi_sc:.4f}  SE={se_a:.4f}")

# ============================================================
# 3. POMEAU–MANNEVILLE  (30 bootstrap resamples) — includes z=1.30
# ============================================================
print("\n" + "="*60)
print("TABLE III — POMEAU–MANNEVILLE MAP")
print("="*60)
Nt_pm = 1500
pm_mg = np.array([50,100,200,400,800])
zvals = [1.20, 1.30, 1.50, 1.80, 2.00, 2.50]

for z in zvals:
    print(f"\n  Generating PM z={z} trajectories...")
    trajs = np.zeros((Nt_pm, Ns))
    for i in range(Nt_pm):
        trajs[i] = pomeau_manneville(z, np.random.uniform(0.01,0.99), Ns)

    t = gx(trajs)
    a, kl, sk, ku = compute_all(t, m_vals=pm_mg)
    se_a, se_kl, se_sk, se_ku = bootstrap_se(t, n_boot=30, pm_mg=pm_mg)
    nfi_sc = abs(a-1)
    print(f"  z={z}:")
    print(f"    alpha   = {a:.4f}  SE={se_a:.4f}")
    print(f"    KL      = {kl:.4f}  SE={se_kl:.4f}")
    print(f"    NFIsc   = {nfi_sc:.4f}  SE={se_a:.4f}")
    print(f"    skew    = {sk:.3f}   SE={se_sk:.4f}")
    print(f"    kurt    = {ku:.3f}   SE={se_ku:.4f}")

print("\n[DONE] All bootstrap SEs computed.")
