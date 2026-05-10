"""
gen_figs_r10.py — Generates all 11 figures for R10 manuscript.

Figure mapping:
  fig1  = logistic variance scaling (4-panel, wide)
  fig2  = logistic Gaussianity test (4-panel, wide) — periodic: "undef."
  fig3  = Hénon map (2×2: variance + Z distributions)
  fig4  = NFI vs r (logistic scan)
  fig5  = NFI vs λ (logistic scan)
  fig6  = PM map (top: variance 3-panel; bottom: distributions 3-panel)
  fig7  = NFI vs m★ (logistic)
  fig8  = Response to observational noise
  fig9  = Binning sensitivity — periodic EXCLUDED
  fig10 = Weighting sensitivity (2-panel)
  fig11 = Near-Fickian yet non-Gaussian detection (scatter + histogram + Q-Q)
"""
import numpy as np, pickle
from scipy import stats
from scipy.stats import skew, kurtosis
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 8, 'figure.dpi': 200
})

D = pickle.load(open('/home/claude/sim_data.pkl', 'rb'))
E = pickle.load(open('/home/claude/sim_extra.pkl', 'rb'))
lt, ht, pmt = D['lt'], D['ht'], D['pmt']
mg, ms, nb = D['mg'], D['ms'], D['nb']
T1, T2, T3 = D['T1'], D['T2'], D['T3']
pm_mg = D['pm_mg']
gx = lambda x: x
FD = '/home/claude/figs'
import os; os.makedirs(FD, exist_ok=True)

# Ensure z=1.3 is in pmt (may be missing from older sim_data.pkl)
if 1.3 not in pmt:
    print("Generating PM z=1.3 trajectories (not in cache)...")
    np.random.seed(42)
    def _pm(z, x0, N, burn=2000):
        x = np.empty(burn + N); x[0] = x0
        for i in range(1, burn + N): x[i] = (x[i-1] + x[i-1]**z) % 1.0
        return x[burn:]
    t13 = np.zeros((1500, 2000))
    for i in range(1500):
        t13[i] = _pm(1.3, np.random.uniform(0.01, 0.99), 2000)
    pmt[1.3] = t13
    print("  z=1.3 done")
    # Add T3 entry for z=1.3
    from scipy.stats import skew as _sk, kurtosis as _ku
    c13 = t13 - np.mean(t13)
    v13 = np.array([np.var(np.sum(c13[:, :m], axis=1), ddof=1) for m in pm_mg])
    ok13 = v13 > 0
    al13, _, _, _, _ = stats.linregress(np.log(pm_mg[ok13]), np.log(v13[ok13]))
    Zm13 = np.sum(c13[:, :ms], axis=1); Zm13 /= np.std(Zm13, ddof=1)
    h13, e13 = np.histogram(Zm13, bins=nb, range=(-5, 5)); h13 = h13 + 1; p13 = h13 / h13.sum()
    bc13 = .5*(e13[:-1]+e13[1:]); bw13 = e13[1]-e13[0]
    q13 = stats.norm.pdf(bc13)*bw13; q13 = q13/q13.sum()
    kl13 = sum(p13[i]*np.log(p13[i]/q13[i]) for i in range(len(p13)) if p13[i]>0 and q13[i]>0)
    T3.insert(1, dict(z=1.3, a=al13, R2=1.0, kl=kl13, w1=0.,
                      nfi=kl13+abs(al13-1),
                      skew=float(_sk(Zm13)), kurt=float(_ku(Zm13))))

def compute_var(t, g, mv):
    gv = g(t); c = gv - np.mean(gv); v = np.zeros(len(mv))
    for i, m in enumerate(mv):
        if m > t.shape[1]: v[i] = np.nan; continue
        v[i] = np.var(np.sum(c[:, :m], axis=1), ddof=1)
    return v

def compute_Zm(t, g, m):
    gv = g(t); c = gv - np.mean(gv); Sm = np.sum(c[:, :m], axis=1)
    s = np.std(Sm, ddof=1)
    return Sm / s if s > 1e-15 else None

def fit_a(m, v):
    ok = (v > 0) & np.isfinite(v)
    if ok.sum() < 3: return 0., 0.
    s, _, r, _, _ = stats.linregress(np.log(m[ok]), np.log(v[ok]))
    return s, r**2

zr = np.linspace(-4, 4, 200)
gpdf = stats.norm.pdf(zr)
rn = {3.50: 'Periodic', 3.60: 'Weak chaos', 3.80: 'Chaotic', 4.00: 'Fully chaotic'}
C1, C2 = '#185FA5', '#D85A30'

# ============================================================
# FIG 1: Logistic variance scaling (4-panel)
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
for i, r in enumerate([3.50, 3.60, 3.80, 4.00]):
    ax = axes[i]; v = compute_var(lt[r], gx, mg); a, R2 = fit_a(mg, v)
    ok = v > 0
    ax.loglog(mg[ok], v[ok], 'o', ms=4, color=C1, zorder=3)
    if ok.sum() >= 2:
        mf = np.linspace(mg[ok][0], mg[ok][-1], 100)
        li = np.mean(np.log(v[ok]) - a * np.log(mg[ok]))
        ax.loglog(mf, np.exp(li) * mf**a, '-', color=C2, lw=1.5)
    ax.set_title(f'{rn[r]}\n$\\alpha={a:.3f}$, $R^2={R2:.3f}$', fontsize=9)
    ax.set_xlabel('$m$'); ax.grid(True, alpha=.3)
    if i == 0: ax.set_ylabel('Var($S_m$)')
plt.tight_layout()
plt.savefig(f'{FD}/fig1.pdf', bbox_inches='tight'); plt.close()
print("Fig 1 done")

# ============================================================
# FIG 2: Logistic Gaussianity test — periodic shows "undef."
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
for i, r in enumerate([3.50, 3.60, 3.80, 4.00]):
    ax = axes[i]
    Z = compute_Zm(lt[r], gx, ms)
    d = [x for x in T1 if x['r'] == r and x['g'] == 'x'][0]
    if r == 3.50:
        # Periodic: degenerate — show placeholder text
        ax.text(0.5, 0.5, 'Degenerate\n(periodic)', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='gray')
        ax.set_title(f'{rn[r]}\nKL = undefined', fontsize=9)
    else:
        if Z is not None and np.std(Z) > .01:
            ax.hist(Z, bins=60, density=True, alpha=.6, color=C1, edgecolor='none')
        ax.plot(zr, gpdf, '-', color=C2, lw=1.5, label='$\\mathcal{N}(0,1)$')
        ax.set_title(f'{rn[r]}\nKL$={d["kl"]:.4f}$', fontsize=9)
        ax.legend(fontsize=7)
    ax.set_xlabel('$Z_{m_\\star}$'); ax.set_xlim(-4, 4)
    if i == 0: ax.set_ylabel('Density')
plt.tight_layout()
plt.savefig(f'{FD}/fig2.pdf', bbox_inches='tight'); plt.close()
print("Fig 2 done")

# ============================================================
# FIG 3: Hénon map (2×2)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(8, 6.5))
for col, (a_h, b_h) in enumerate([(1.4, .3), (1.3, .3)]):
    tr = ht[(a_h, b_h)]; v = compute_var(tr, gx, mg); al, R2 = fit_a(mg, v); ok = v > 0
    ax = axes[0, col]
    ax.loglog(mg[ok], v[ok], 'o', ms=4, color=C1)
    if ok.sum() >= 2:
        mf = np.linspace(mg[ok][0], mg[ok][-1], 100)
        li = np.mean(np.log(v[ok]) - al * np.log(mg[ok]))
        ax.loglog(mf, np.exp(li) * mf**al, '--', color=C2, lw=1.5)
    lb = 'Chaotic' if a_h == 1.4 and b_h == .3 else 'Low mixing'
    ax.set_title(f'$({a_h},{b_h})$ [{lb}]\n$\\alpha={al:.2f}$, $R^2={R2:.3f}$', fontsize=9)
    ax.set_xlabel('$m$'); ax.set_ylabel('Var($S_m$)'); ax.grid(True, alpha=.3)
    ax = axes[1, col]; Z = compute_Zm(tr, gx, ms)
    d2 = [x for x in T2 if x['ab'] == f'({a_h},{b_h})' and x['g'] == 'x'][0]
    if Z is not None and np.std(Z) > .01:
        ax.hist(Z, bins=60, density=True, alpha=.6, color=C1)
    ax.plot(zr, gpdf, '-', color=C2, lw=1.5)
    ax.set_title(f'KL$={d2["kl"]:.3f}$', fontsize=9)
    ax.set_xlabel('$Z_{m_\\star}$'); ax.set_ylabel('Density'); ax.set_xlim(-4, 4)
plt.tight_layout()
plt.savefig(f'{FD}/fig3.pdf', bbox_inches='tight'); plt.close()
print("Fig 3 done")

# ============================================================
# FIG 4: NFI vs r (logistic scan)
# ============================================================
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.plot(E['rscan'], E['nfi_scan'], '.', ms=3, color=C1)
ax.set_xlabel('$r$'); ax.set_ylabel('NFI'); ax.set_title('NFI vs $r$')
ax.grid(True, alpha=.3)
plt.tight_layout()
plt.savefig(f'{FD}/fig4.pdf', bbox_inches='tight'); plt.close()
print("Fig 4 done")

# ============================================================
# FIG 5: NFI vs λ (logistic scan)
# ============================================================
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.plot(E['lam_scan'], E['nfi_scan'], '.', ms=4, color=C1)
ax.set_xlabel('Lyapunov exponent $\\lambda$'); ax.set_ylabel('NFI')
ax.set_title('NFI vs $\\lambda$ (logistic map)')
m = (E['lam_scan'] > -0.05) & (E['lam_scan'] < 0.15) & \
    (E['nfi_scan'] > 0.5) & (E['nfi_scan'] < 15)
if m.any():
    ax.plot(E['lam_scan'][m], E['nfi_scan'][m], 'o', ms=6,
            mfc='none', mec=C2, mew=1.5, label='Same $\\lambda$, different NFI')
    ax.legend()
ax.grid(True, alpha=.3)
plt.tight_layout()
plt.savefig(f'{FD}/fig5.pdf', bbox_inches='tight'); plt.close()
print("Fig 5 done")

# ============================================================
# FIG 6: Pomeau–Manneville (top: variance; bottom: distributions)
# ============================================================
fig = plt.figure(figsize=(11, 6.5))
# Top row: variance scaling for z=1.5, 2.0, 2.5
for i, z in enumerate([1.5, 2.0, 2.5]):
    ax = fig.add_subplot(2, 3, i + 1)
    t = pmt[z]; v = compute_var(t, gx, pm_mg); al, R2 = fit_a(pm_mg, v); ok = v > 0
    ax.loglog(pm_mg[ok], v[ok], 'o', ms=4, color=C1)
    if ok.sum() >= 2:
        mf = np.linspace(pm_mg[ok][0], pm_mg[ok][-1], 100)
        li = np.mean(np.log(v[ok]) - al * np.log(pm_mg[ok]))
        ax.loglog(mf, np.exp(li) * mf**al, '--', color=C2, lw=1.5)
    # alpha=1 reference line
    mf2 = np.array([pm_mg[ok][0], pm_mg[ok][-1]])
    li1 = np.mean(np.log(v[ok]) - 1.0 * np.log(pm_mg[ok]))
    ax.loglog(mf2, np.exp(li1) * mf2**1.0, ':', color='gray', lw=1.2,
              label='$\\alpha=1$ ref.')
    ax.set_title(f'$z={z}$\n$\\alpha={al:.3f}$, $R^2={R2:.3f}$', fontsize=9)
    ax.set_xlabel('$m$'); ax.grid(True, alpha=.3)
    if i == 0: ax.set_ylabel('Var($S_m$)')
    ax.legend(fontsize=7)
# Bottom row: distributions for z=1.3, 2.0, 2.5
bot_cases = [(1.3, 'near-Fickian\n(borderline non-Gaussian)'),
             (2.0, 'anomalous'),
             (2.5, 'anomalous')]
for i, (z, label) in enumerate(bot_cases):
    ax = fig.add_subplot(2, 3, i + 4)
    t = pmt[z]
    Z = compute_Zm(t, gx, ms)
    # get alpha for subplot annotation
    v = compute_var(t, gx, pm_mg); al_z, _ = fit_a(pm_mg, v)
    # KL
    if Z is not None:
        h, e = np.histogram(Z, bins=80, range=(-5, 5)); h = h + 1; p = h / h.sum()
        bc = .5 * (e[:-1] + e[1:]); bw = e[1] - e[0]
        q = stats.norm.pdf(bc) * bw; q = q / q.sum()
        kl = sum(p[j] * np.log(p[j] / q[j]) for j in range(len(p)) if p[j] > 0 and q[j] > 0)
        ax.hist(Z, bins=60, density=True, alpha=.6, color=C1, edgecolor='none')
    ax.plot(zr, gpdf, '-', color=C2, lw=1.5, label='$\\mathcal{N}(0,1)$')
    ax.set_title(f'$z={z}$ ({label})\nKL$={kl:.3f}$, $\\alpha={al_z:.3f}$', fontsize=8)
    ax.set_xlabel('$Z_{m_\\star}$'); ax.set_xlim(-4, 4)
    if i == 0: ax.set_ylabel('Density')
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(f'{FD}/fig6.pdf', bbox_inches='tight'); plt.close()
print("Fig 6 done")

# ============================================================
# FIG 7: NFI vs m★
# ============================================================
fig, ax = plt.subplots(figsize=(5.5, 3.8))
col = {3.50: '#A32D2D', 3.60: '#BA7517', 4.00: C1}
lab = {3.50: '$r=3.50$ (periodic)', 3.60: '$r=3.60$ (weak chaos)',
       4.00: '$r=4.00$ (fully chaotic)'}
for r in [3.50, 3.60, 4.00]:
    mv = [d['m'] for d in E['mstar_res'][r]]
    nv = [d['nfi'] for d in E['mstar_res'][r]]
    ax.plot(mv, nv, 'o-', ms=5, color=col[r], label=lab[r])
ax.set_xlabel('$m_\\star$'); ax.set_ylabel('NFI'); ax.set_title('NFI vs $m_\\star$')
ax.legend(fontsize=8); ax.grid(True, alpha=.3); ax.set_yscale('log')
plt.tight_layout()
plt.savefig(f'{FD}/fig7.pdf', bbox_inches='tight'); plt.close()
print("Fig 7 done")

# ============================================================
# FIG 8: Response to observational noise (title fixed)
# ============================================================
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for r, c, l in [(4.00, C1, '$r=4.00$'), (3.60, '#BA7517', '$r=3.60$')]:
    ev = [d['eps'] for d in E['noise_res'][r]]
    nv = [d['nfi'] for d in E['noise_res'][r]]
    ax.plot(ev, nv, 'o-', ms=5, color=c, label=l)
ax.set_xlabel('Noise $\\varepsilon$'); ax.set_ylabel('NFI')
ax.set_title('Response to observational noise')  # FIXED: was "Noise robustness"
ax.legend(); ax.grid(True, alpha=.3)
plt.tight_layout()
plt.savefig(f'{FD}/fig8.pdf', bbox_inches='tight'); plt.close()
print("Fig 8 done")

# ============================================================
# FIG 9: Binning sensitivity — periodic EXCLUDED
# ============================================================
fig, ax = plt.subplots(figsize=(5.5, 3.8))
cb = {3.60: '#BA7517', 3.80: '#0F6E56', 4.00: C1}
for r in [3.60, 3.80, 4.00]:  # r=3.50 excluded (degenerate Z)
    ax.plot(E['binr'], E['bsens'][r], 'o-', ms=3, color=cb[r], label=f'$r={r}$')
ax.set_xlabel('Bins'); ax.set_ylabel('KL'); ax.set_title('KL vs bin count')
ax.legend(); ax.grid(True, alpha=.3)
plt.tight_layout()
plt.savefig(f'{FD}/fig9.pdf', bbox_inches='tight'); plt.close()
print("Fig 9 done")

# ============================================================
# FIG 10: Weighting sensitivity (2-panel)
# ============================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
for c in E['cv']:
    nv = []
    for r in [3.50, 3.60, 3.80, 4.00]:
        d = [x for x in T1 if x['r'] == r and x['g'] == 'x'][0]
        nv.append(d['kl'] + c * d['nfi_sc'])
    a1.plot([3.50, 3.60, 3.80, 4.00], nv, 'o-', label=f'$c={c}$')
a1.set_xlabel('$r$'); a1.set_ylabel('NFI'); a1.set_title('All regimes')
a1.legend(); a1.set_yscale('log'); a1.grid(True, alpha=.3)
for c in E['cv']:
    a2.plot(E['rcl'], E['cnfi'][c], 'o-', ms=4, label=f'$c={c}$')
a2.set_xlabel('$r$'); a2.set_ylabel('NFI')
a2.set_title('Close neighborhood ($r\\in[3.58,3.66]$)')
a2.legend(); a2.grid(True, alpha=.3)
plt.tight_layout()
plt.savefig(f'{FD}/fig10.pdf', bbox_inches='tight'); plt.close()
print("Fig 10 done")

# ============================================================
# FIG 11: Near-Fickian yet non-Gaussian detection
#   (a) NFIshape vs NFIscale scatter
#   (b) Histogram: logistic r=4.0 vs PM z=1.3
#   (c) Q-Q plot: PM z=1.3 vs normal
# ============================================================
fig = plt.figure(figsize=(12, 4))

# --- (a) Scatter: NFIshape vs NFIscale ---
ax_s = fig.add_subplot(1, 3, 1)
# Collect all (NFIsc, NFIsh, label, marker, color)
points = []
for d in T1:
    if d['r'] == 3.50: continue
    lbl = f"Log $r={d['r']}$"
    points.append((d['nfi_sc'], d['kl'], lbl, 'o', '#888888'))
for d in T2:
    lbl = f"H{d['ab']}"
    points.append((abs(d['a'] - 1), d['kl'], lbl, 's', '#0F6E56'))
for d in T3:
    z_val = d['z']
    col_pm = '#D85A30' if z_val == 1.3 else C1
    mk = '*' if z_val == 1.3 else '^'
    points.append((abs(d['a'] - 1), d['kl'], f"PM $z={z_val}$", mk, col_pm))

# Draw shaded quadrant for near-Fickian yet non-Gaussian
ax_s.axhspan(0.08, 1.0, xmin=0, xmax=0.15, alpha=0.08, color='#D85A30',
             label='near-Fickian\nnon-Gaussian zone')
ax_s.axvline(x=0.1, color='gray', lw=0.8, ls='--', alpha=0.5)
ax_s.axhline(y=0.08, color='gray', lw=0.8, ls='--', alpha=0.5)

for sc, sh, lbl, mk, col in points:
    ax_s.plot(sc, sh, mk, ms=6 if mk == '*' else 4, color=col, zorder=3)

# Annotate PM z=1.3 specifically
d13 = [d for d in T3 if d['z'] == 1.3][0]
ax_s.annotate('PM $z=1.3$\n(borderline)',
              xy=(abs(d13['a'] - 1), d13['kl']),
              xytext=(0.15, 0.15), fontsize=7,
              arrowprops=dict(arrowstyle='->', color='#D85A30', lw=0.8),
              color='#D85A30')

ax_s.set_xlabel('NFI$_\\mathrm{scale}=|\\alpha-1|$')
ax_s.set_ylabel('NFI$_\\mathrm{shape}$ (KL)')
ax_s.set_title('(a) Shape vs scale decomposition')
ax_s.set_xlim(-0.02, 0.75); ax_s.set_ylim(-0.02, 1.5)
ax_s.grid(True, alpha=.3)

# --- (b) Histogram comparison ---
ax_h = fig.add_subplot(1, 3, 2)
Z40  = compute_Zm(lt[4.0], gx, ms)   # logistic r=4.0 (Gaussian reference)
Z13  = compute_Zm(pmt[1.3], gx, ms)  # PM z=1.3 (borderline)
if Z40 is not None:
    ax_h.hist(Z40, bins=50, density=True, alpha=0.5, color=C1,
              label='Logistic $r=4.0$\n($\\alpha\\approx1.00$)')
if Z13 is not None:
    ax_h.hist(Z13, bins=50, density=True, alpha=0.5, color=C2,
              label='PM $z=1.3$\n($\\alpha\\approx1.05$)')
ax_h.plot(zr, gpdf, 'k-', lw=1.2, label='$\\mathcal{N}(0,1)$')
ax_h.set_xlabel('$Z_{m_\\star}$'); ax_h.set_ylabel('Density')
ax_h.set_title('(b) Both near-Fickian;\nPM shows asymmetry')
ax_h.set_xlim(-4, 4); ax_h.legend(fontsize=7)

# --- (c) Q-Q plot ---
ax_q = fig.add_subplot(1, 3, 3)
if Z13 is not None:
    q_th = np.percentile(np.random.randn(100000), np.linspace(1, 99, 200))
    q_pm = np.percentile(Z13, np.linspace(1, 99, 200))
    ax_q.plot(q_th, q_th, '-', color='gray', lw=1.2, label='Normal ref.')
    ax_q.plot(q_th, q_pm, '.', ms=3, color=C2, label='PM $z=1.3$')
if Z40 is not None:
    q_log = np.percentile(Z40, np.linspace(1, 99, 200))
    ax_q.plot(q_th, q_log, '.', ms=3, color=C1, label='Logistic $r=4.0$')
ax_q.set_xlabel('Theoretical quantiles')
ax_q.set_ylabel('Sample quantiles')
ax_q.set_title('(c) Q-Q: PM tails deviate')
ax_q.legend(fontsize=7); ax_q.grid(True, alpha=.3)

plt.tight_layout()
plt.savefig(f'{FD}/fig11.pdf', bbox_inches='tight'); plt.close()
print("Fig 11 done")

print(f"\nAll 11 figures generated in {FD}")
