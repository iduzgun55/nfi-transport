"""Figures for the revised manuscript (R1). Reads data/tables.pkl, data/mstar.pkl,
data/extra.pkl produced by run_tables.py, run_mstar.py and run_extra.py."""
import os
import numpy as np, pickle, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nfi_core import gaussian_null, standardize

D = 'data/'; FD = 'figs/'
T = pickle.load(open(D + 'tables.pkl', 'rb'))
M = pickle.load(open(D + 'mstar.pkl', 'rb'))
E = pickle.load(open(D + 'extra.pkl', 'rb'))

# Okabe-Ito (validated categorical order), secondary encoding by marker/linestyle
C = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000']
MK = ['o', 's', '^', 'D', 'v', 'P', 'X']
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 7.5,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': 0.7, 'lines.linewidth': 1.4,
                     'savefig.dpi': 300, 'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5})
ZLAB = r'$Z_{m^{*}}$'

def save(fig, name):
    fig.savefig(FD + name + '.pdf', bbox_inches='tight'); fig.savefig(FD + name + '.png', bbox_inches='tight', dpi=200); plt.close(fig)

def panel_label(ax, s):
    t = ax.get_title()
    if t:
        ax.set_title(s + ' ' + t, fontsize=8.5)
    else:
        ax.text(-0.2, 1.015, s, transform=ax.transAxes, va='bottom', ha='left', fontweight='bold', fontsize=9)

# ---------------------------------------------------------------- Fig 1
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.2))
for i, (r, lab) in enumerate([(3.5, 'periodic'), (3.6, 'two-band chaotic'), (3.8, 'chaotic'), (4.0, 'fully chaotic')]):
    ax = axs[i]; d = T[('log', r, 'x')]; va = d['var_all']; m = np.arange(1, len(va) + 1)
    ax.set_title(r'$r=%.2f$ (%s)' % (r, lab))
    if r == 3.5:
        ax.plot(m[:40], va[:40], '-', color=C[0], lw=0.8, marker='.', ms=4)
        ax.set_xlabel(r'$m$'); ax.set_ylabel(r'Var$(S_m)$')
        ax.text(0.5, 0.42, 'bounded, period 4 in $m$\n(zero for $m\\equiv0$ mod 4)', transform=ax.transAxes, ha='center', fontsize=6.5, bbox=dict(fc='white', ec='none', alpha=0.9))
        ax.set_ylim(-0.003, 0.05)
    else:
        ax.loglog(m[10:], va[10:], '-', color='0.6', lw=0.6, label='all $m$')
        mg = np.array(d['m_grid']); v = np.array(d['var'])
        ax.loglog(mg, v, MK[0], color=C[0], ms=4.5, label='grid')
        fit = np.exp(np.polyfit(np.log(mg), np.log(v), 1)); mm = np.linspace(40, 1700, 50)
        a_ = np.polyfit(np.log(mg), np.log(v), 1)
        ax.loglog(mm, np.exp(a_[1]) * mm ** a_[0], '--', color=C[1], lw=1.1, label=r'fit $\alpha=%.3f$' % d['alpha'])
        ax.loglog(mm, v[4] * (mm / 800.0), ':', color='k', lw=1.0, label=r'$\alpha=1$')
        if r == 3.6:
            b = E['band']; ax.loglog(mm, b['A'] + b['slope'] * mm, '-.', color=C[2], lw=1.1, label=r'$A+2Dm$')
        ax.set_xlabel(r'$m$')
        if i == 1: ax.set_ylabel(r'Var$(S_m)$')
        if r in (3.6, 4.0): ax.legend(loc='upper left', frameon=False, handlelength=1.6, fontsize=6.5)
    panel_label(ax, '(%s)' % 'abcd'[i])
fig.tight_layout(w_pad=0.6); save(fig, 'fig1')

# ---------------------------------------------------------------- Fig 2: shape test logistic
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=True)
zz = np.linspace(-4.5, 4.5, 300)
for i, r in enumerate([3.6, 3.8, 4.0]):
    ax = axs[i]; d = T[('log', r, 'x')]; Z = d['Z']
    ax.hist(Z, bins=np.linspace(-5, 5, 81), density=True, color=C[0], alpha=0.55, edgecolor='white', linewidth=0.3)
    ax.plot(zz, stats.norm.pdf(zz), color=C[1], lw=1.3, label=r'$\mathcal{N}(0,1)$')
    ax.axvline(0, color='k', lw=0.6, ls=':')
    ax.set_title(r'$r=%.2f$' % r); ax.set_xlabel(ZLAB); ax.set_xlim(-4.5, 4.5)
    ax.text(0.98, 0.95, r'$\Delta D_{\rm KL}=%.4f$' % d['kl_exc'] + '\n' + r'$p_{\rm KL}=%.2g$' % d['p']['kl'] + '\n' + r'skew $=%.2f$' % d['skew'],
            transform=ax.transAxes, ha='right', va='top', fontsize=7)
    panel_label(ax, '(%s)' % 'abc'[i])
axs[0].set_ylabel('density'); axs[0].legend(loc='center left', frameon=False, bbox_to_anchor=(0.0, 0.62))
fig.tight_layout(w_pad=0.4); save(fig, 'fig2')

# ---------------------------------------------------------------- Fig 3: fine r-scan
sc = E['scan']; r_ = np.array([s['r'] for s in sc]); lam = np.array([s['lam'] for s in sc])
per = np.array([s['periodic'] for s in sc])
nsc = np.array([abs(s['alpha'] - 1) if not s['periodic'] else np.nan for s in sc])
kex = np.array([s['kl_exc'] if not s['periodic'] else np.nan for s in sc])
R2 = np.array([s['R2'] for s in sc])
fig, axs = plt.subplots(3, 1, figsize=(3.4, 5.2), sharex=True)
for ax in axs:
    for rr, pp in zip(r_, per):
        if pp: ax.axvspan(rr - 0.0005, rr + 0.0005, color='0.85', lw=0)
axs[0].plot(r_, lam, '-', color=C[0], lw=0.7); axs[0].axhline(0, color='k', lw=0.6); axs[0].set_ylabel(r'$\lambda$')
good = (~per) & (R2 >= 0.99)
axs[1].plot(r_[~per], nsc[~per], '.', color='0.6', ms=2.2, label=r'$R^2<0.99$')
axs[1].plot(r_[good], nsc[good], '.', color=C[0], ms=2.6, label=r'$R^2\geq0.99$')
axs[1].set_yscale('log'); axs[1].set_ylabel(r'NFI$_{\rm scale}=|\alpha-1|$'); axs[1].axhline(0.05, color=C[1], ls='--', lw=0.8)
axs[1].legend(frameon=False, loc='lower left', markerscale=3)
axs[2].plot(r_[~per], kex[~per], '.', color=C[2], ms=2.6)
null = gaussian_null(2000); q99 = np.percentile(null['kl'], 99) - np.mean(null['kl'])
axs[2].axhline(q99, color=C[1], ls='--', lw=0.8, label='null 99%')
axs[2].set_yscale('symlog', linthresh=1e-2); axs[2].axhline(0, color='k', lw=0.5); axs[2].set_ylabel(r'NFI$_{\rm shape}=\Delta D_{\rm KL}$'); axs[2].set_xlabel(r'$r$')
axs[2].legend(frameon=False, loc='upper right')
for i, ax in enumerate(axs): panel_label(ax, '(%s)' % 'abc'[i])
axs[2].set_xlim(3.5, 4.0)
fig.tight_layout(h_pad=0.3); save(fig, 'fig3')

# ---------------------------------------------------------------- Fig 4: components vs lambda
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
ok = ~per
s1 = axs[0].scatter(lam[ok], nsc[ok], c=r_[ok], cmap='viridis', s=7, lw=0)
axs[0].set_yscale('log'); axs[0].set_xlabel(r'$\lambda$'); axs[0].set_ylabel(r'NFI$_{\rm scale}$'); axs[0].axhline(0.05, color=C[1], ls='--', lw=0.8)
axs[1].scatter(lam[ok], kex[ok], c=r_[ok], cmap='viridis', s=7, lw=0)
axs[1].set_yscale('symlog', linthresh=1e-2); axs[1].axhline(0, color='k', lw=0.5); axs[1].set_xlabel(r'$\lambda$'); axs[1].set_ylabel(r'NFI$_{\rm shape}=\Delta D_{\rm KL}$'); axs[1].axhline(q99, color=C[1], ls='--', lw=0.8)
cb = fig.colorbar(s1, ax=axs, pad=0.01, fraction=0.03); cb.set_label(r'$r$')
panel_label(axs[0], '(a)'); panel_label(axs[1], '(b)')
save(fig, 'fig4')

# ---------------------------------------------------------------- Fig 5: Henon
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
ax = axs[0]
for j, (key, lab) in enumerate([(('hen', 1.4, 0.3, 'x'), r'$(1.4,0.3)$, $g=x$'), (('hen', 1.4, 0.3, 'x2'), r'$(1.4,0.3)$, $g=x^2$'),
                                 (('hen', 1.4, 0.1, 'x'), r'$(1.4,0.1)$, $g=x$'), (('hen', 1.3, 0.3, 'x'), r'$(1.3,0.3)$, $g=x$')]):
    d = T[key]; va = d['var_all']; m = np.arange(1, len(va) + 1)
    ax.loglog(m[9:], va[9:] / m[9:], '-', color=C[j], lw=1.0, label=lab)
ax.set_xlabel(r'$m$'); ax.set_ylabel(r'Var$(S_m)/m$'); ax.legend(frameon=False, fontsize=6.5, loc='lower left')
ax = axs[1]; d = T[('hen', 1.4, 0.3, 'x')]
ax.hist(d['Z'], bins=np.linspace(-5, 5, 81), density=True, color=C[0], alpha=0.55, edgecolor='white', linewidth=0.3)
ax.plot(zz, stats.norm.pdf(zz), color=C[1], lw=1.3); ax.axvline(0, color='k', lw=0.6, ls=':')
ax.set_xlabel(ZLAB); ax.set_ylabel('density'); ax.set_title(r'$(1.4,0.3)$, $g=x$'); ax.set_xlim(-4.5, 4.5)
ax.text(0.98, 0.95, r'$\Delta D_{\rm KL}=%.4f$' % d['kl_exc'] + '\n' + r'$p_{\rm KL}=%.2g$' % d['p']['kl'], transform=ax.transAxes, ha='right', va='top', fontsize=7)
ax = axs[2]; d = T[('hen', 1.3, 0.3, 'x')]
vals, cnt = np.unique(np.round(d['Z'], 3), return_counts=True)
ax.vlines(vals, 0, cnt / cnt.sum(), color=C[3], lw=1.5); ax.axvline(0, color='k', lw=0.6, ls=':')
ax.set_xlabel(ZLAB); ax.set_ylabel('probability'); ax.set_title(r'$(1.3,0.3)$: period-7 orbit'); ax.set_xlim(-2.5, 2.5)
for i, ax in enumerate(axs): panel_label(ax, '(%s)' % 'abc'[i])
fig.tight_layout(w_pad=0.5); save(fig, 'fig5')

# ---------------------------------------------------------------- Fig 6: PM overview
zs = [1.2, 1.3, 1.5, 1.8, 2.0, 2.5]
fig = plt.figure(figsize=(7.2, 4.4)); gs = fig.add_gridspec(2, 4)
ax = fig.add_subplot(gs[0, :2])
for j, z in enumerate(zs):
    d = T[('pm', z)]; va = d['var_all']; m = np.arange(1, len(va) + 1)
    ax.loglog(m[9:], va[9:] / m[9:], '-', color=C[j], lw=1.1, label=r'$z=%.1f$' % z)
ax.set_xlabel(r'$m$'); ax.set_ylabel(r'Var$(S_m)/m$'); ax.legend(frameon=False, ncol=3, fontsize=6.5, loc='upper left')
ax.text(0.98, 0.04, r'flat $\Leftrightarrow\ \alpha=1$', transform=ax.transAxes, ha='right', fontsize=7)
ax.set_title('variance growth (normalized)', fontsize=8); panel_label(ax, '(a)')
ax = fig.add_subplot(gs[0, 2:])
for j, z in enumerate(zs):
    d = T[('pm', z)]
    ax.errorbar(z, d['alpha'], yerr=[[d['alpha'] - d['ci_alpha'][0]], [d['ci_alpha'][1] - d['alpha']]], fmt=MK[j], color=C[j], ms=5, capsize=2)
zt = np.linspace(1.5, 2.0, 50)
ax.plot(zt, 2 * (zt - 1), '--', color='k', lw=0.9, label=r'$2(z-1)$ (fluctuation scale)')
zt2 = np.linspace(1.5, 2.0, 50); ax.plot(zt2, 3 - 1 / (zt2 - 1), '-.', color='0.4', lw=0.9, label=r'$3-1/(z-1)$ (second moment)')
ax.plot([1.5], [1 + 1 / np.log(283)], 'x', color='k', ms=6, label=r'$z=1.5$: $1+1/\ln m$')
ax.plot([1.1, 1.5], [1, 1], ':', color='k', lw=1.0, label=r'CLT: $\alpha=1$')
ax.axvspan(2.0, 2.6, color='0.9', lw=0); ax.text(2.3, 1.0, 'infinite\ninvariant\nmeasure', ha='center', fontsize=6.5)
ax.set_xlabel(r'$z$'); ax.set_ylabel(r'$\alpha$ ($m\in[50,1600]$)'); ax.legend(frameon=False, loc='upper left', fontsize=6); ax.set_xlim(1.1, 2.6); ax.set_ylim(0.9, 2.15)
ax.set_title(r'fitted exponent vs. asymptotic theory', fontsize=8); panel_label(ax, '(b)')
for j, z in enumerate([1.3, 1.5, 1.8, 2.0]):
    ax = fig.add_subplot(gs[1, j]); d = T[('pm', z)]
    ax.hist(d['Z'], bins=np.linspace(-6, 6, 97), density=True, color=C[zs.index(z)], alpha=0.6, edgecolor='white', linewidth=0.2)
    ax.plot(zz, stats.norm.pdf(zz), color='k', lw=1.0)
    ax.axvline(0, color='k', lw=0.8, ls=':'); ax.set_yscale('log'); ax.set_ylim(1e-4, 5); ax.set_xlim(-6, 6)
    ax.set_xlabel(ZLAB); ax.set_title(r'$z=%.1f$' % z)
    if j == 0: ax.set_ylabel('density (log)')
    ax.text(0.97, 0.95, 'skew %.2f' % d['skew'], transform=ax.transAxes, ha='right', va='top', fontsize=6.5)
    panel_label(ax, '(%s)' % 'cdef'[j])
fig.tight_layout(h_pad=0.6, w_pad=0.3); save(fig, 'fig6')

# ---------------------------------------------------------------- Fig 7: PM finite-time dependence (R2 point 5)
fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0))
cases = [(('pm', 1.2), r'$z=1.2$'), (('pm', 1.3), r'$z=1.3$'), (('pm', 1.5), r'$z=1.5$'), (('pm', 1.8), r'$z=1.8$'), (('pm', 2.0), r'$z=2.0$'), (('log', 4.0), r'logistic $r=4$')]
for j, (key, lab) in enumerate(cases):
    o = M[key]; ms = o['ms']; v = o['var']; vb = o['var_boot']
    # windowed fit over 5 consecutive grid points (factor 4 in m)
    w = 5; mc, al, lo, hi = [], [], [], []
    for k in range(len(ms) - w + 1):
        x = np.log(ms[k:k + w]); a0 = np.polyfit(x, np.log(v[k:k + w]), 1)[0]
        ab = [np.polyfit(x, np.log(vb[b, k:k + w]), 1)[0] for b in range(vb.shape[0])]
        mc.append(np.exp(x.mean())); al.append(a0); lo.append(np.percentile(ab, 2.5)); hi.append(np.percentile(ab, 97.5))
    col = C[j]; ls = '-' if key[0] == 'pm' else '--'
    axs[0, 0].plot(mc, al, ls, color=col, marker=MK[j], ms=3.5, label=lab); axs[0, 0].fill_between(mc, lo, hi, color=col, alpha=0.18, lw=0)
    sh = o['shape']; mm = np.array(sorted(sh))
    kx = np.array([sh[m]['kl_exc'] for m in mm]); sk = np.array([sh[m]['skew'] for m in mm]); ses = np.array([sh[m]['se_skew'] for m in mm])
    axs[0, 1].plot(mm, kx, ls, color=col, marker=MK[j], ms=3.5)
    axs[1, 0].plot(mm, sk, ls, color=col, marker=MK[j], ms=3.5); axs[1, 0].fill_between(mm, sk - 2 * ses, sk + 2 * ses, color=col, alpha=0.18, lw=0)
    ku = np.array([sh[m]['kurt'] for m in mm])
    axs[1, 1].plot(mm, ku, ls, color=col, marker=MK[j], ms=3.5)
mt = np.logspace(np.log10(50), np.log10(10000), 50)
axs[0, 0].plot(mt, 1 + 1 / np.log(mt), ':', color=C[2], lw=1.2, label=r'$1+1/\ln m$ ($z=1.5$)')
axs[0, 0].axhline(1.6, color=C[3], ls=':', lw=1.2, label=r'$2(z-1)=1.6$ ($z=1.8$)')
axs[0, 0].axhline(1.75, color=C[3], ls='-.', lw=0.9, label=r'$3-1/(z-1)=1.75$ ($z=1.8$)')
axs[0, 0].axhline(1.0, color='k', ls=':', lw=0.8)
axs[0, 0].set_xscale('log'); axs[0, 0].set_ylabel(r'local exponent $\alpha_{\rm loc}(m)$'); h_, l_ = axs[0, 0].get_legend_handles_labels(); fig.legend(h_, l_, loc='upper center', ncol=5, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, 1.07)); axs[0, 0].set_ylim(0.85, 1.85)
nl = gaussian_null(10000); axs[0, 1].axhline(np.percentile(nl['kl'], 99) - np.mean(nl['kl']), color='k', ls='--', lw=0.8, label='null 99%')
axs[0, 1].set_xscale('log'); axs[0, 1].set_yscale('symlog', linthresh=1e-3); axs[0, 1].axhline(0, color='k', lw=0.5); axs[0, 1].set_ylabel(r'$\Delta D_{\rm KL}$'); axs[0, 1].legend(frameon=False, fontsize=7, loc='lower left')
axs[1, 0].axhline(0, color='k', lw=0.6); axs[1, 0].set_xscale('log'); axs[1, 0].set_ylabel('skewness of ' + ZLAB.replace('m^{*}', 'm'))
axs[1, 1].axhline(0, color='k', lw=0.6); axs[1, 1].set_xscale('log'); axs[1, 1].set_yscale('symlog', linthresh=1); axs[1, 1].set_ylabel('excess kurtosis')
for i, ax in enumerate(axs.flat): ax.set_xlabel(r'$m$'); panel_label(ax, '(%s)' % 'abcd'[i])
fig.tight_layout(h_pad=0.6); save(fig, 'fig7')

# ---------------------------------------------------------------- Fig 8: diagnostic plane
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax = axs[0]
pts = [(('log', 3.6, 'x'), 'L 3.6', 0), (('log', 3.8, 'x'), 'L 3.8', 0), (('log', 4.0, 'x'), 'L 4.0', 0),
       (('hen', 1.4, 0.3, 'x'), 'H (1.4,0.3)', 1), (('hen', 1.4, 0.1, 'x'), 'H (1.4,0.1)', 1)] + [(('pm', z), 'PM %.1f' % z, 2) for z in zs]
famc = {0: C[0], 1: C[2], 2: C[1]}; famm = {0: 'o', 1: 's', 2: '^'}
for key, lab, f in pts:
    d = T[key]; xs = abs(d['alpha'] - 1); ys = max(d['kl_exc'], 2e-4)
    xe = [[xs - min(abs(c - 1) for c in d['ci_alpha']) if not (d['ci_alpha'][0] < 1 < d['ci_alpha'][1]) else xs], [max(abs(c - 1) for c in d['ci_alpha']) - xs]]
    ylo = max(d['ci_kl_exc'][0] - d['kl'] + d['kl_exc'], 2e-4)
    ax.errorbar(xs, ys, xerr=np.abs(xe), fmt=famm[f], color=famc[f], ms=5, capsize=1.5, lw=0.8)
    ax.annotate(lab, (xs, ys), textcoords='offset points', xytext=(4, 3), fontsize=6.3)
ax.axvline(0.05, color='k', ls='--', lw=0.8); ax.axhline(np.percentile(nl['kl'], 99) - np.mean(nl['kl']), color='k', ls=':', lw=0.8)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(2e-3, 1.2); ax.set_ylim(1.5e-4, 1.5)
ax.set_xlabel(r'NFI$_{\rm scale}=|\alpha-1|$'); ax.set_ylabel(r'NFI$_{\rm shape}=\Delta D_{\rm KL}$')
ax.text(0.03, 0.6, 'near-Fickian\nscale,\nnon-Gaussian', transform=ax.transAxes, fontsize=6.5, color='0.3')
ax.text(0.03, 0.05, 'near-Fickian', transform=ax.transAxes, fontsize=6.5, color='0.3')
ax.text(0.97, 0.05, 'anomalous scale,\nGaussian shape', transform=ax.transAxes, fontsize=6.5, color='0.3', ha='right')
ax.set_title(r'$m^{*}=800$, $N_{\rm traj}=10^4$', fontsize=8); panel_label(ax, '(a)')
ax = axs[1]
for j, key in enumerate([('pm', 1.3), ('pm', 1.5), ('pm', 1.8), ('log', 3.6), ('log', 4.0)]):
    o = M[key]; ms = o['ms']; v = o['var']; sh = o['shape']
    w = 5; xs_, ys_ = [], []
    for k in range(len(ms) - w + 1):
        mtop = ms[k + w - 1]
        if key == ('log', 3.6):
            sel = [i for i in range(k, k + w) if ms[i] % 2 == 0]
            if len(sel) < 3: continue
            a0 = np.polyfit(np.log(ms[sel]), np.log(v[sel]), 1)[0]
        else:
            a0 = np.polyfit(np.log(ms[k:k + w]), np.log(v[k:k + w]), 1)[0]
        xs_.append(max(abs(a0 - 1), 2e-3)); ys_.append(max(sh[mtop]['kl_exc'], 2e-4))
    lab = {('pm', 1.2): 'PM 1.2', ('pm', 1.3): 'PM 1.3', ('pm', 1.5): 'PM 1.5', ('pm', 1.8): 'PM 1.8', ('log', 3.6): 'L 3.6 (even $m$)', ('log', 4.0): 'L 4.0'}[key]
    jj = [1, 2, 3, 4, 5][j]
    ax.plot(xs_, ys_, '-', color=C[jj], lw=1.0, marker=MK[jj], ms=3, label=lab)
    ax.plot(xs_[0], ys_[0], 'o', mfc='white', mec=C[jj], ms=6)
    ax.annotate('', xy=(xs_[-1], ys_[-1]), xytext=(xs_[-2], ys_[-2]), arrowprops=dict(arrowstyle='-|>', color=C[jj], lw=1.2))
ax.axvline(0.05, color='k', ls='--', lw=0.8); ax.axhline(np.percentile(nl['kl'], 99) - np.mean(nl['kl']), color='k', ls=':', lw=0.8)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(2e-3, 1.2); ax.set_ylim(1.5e-4, 1.5)
ax.set_xlabel(r'NFI$_{\rm scale}(m)$'); ax.set_ylabel(r'NFI$_{\rm shape}(m)$'); ax.legend(frameon=False, fontsize=6.3, loc='upper left', ncol=2)
ax.set_title(r'trajectories in $m$ (open circle: $m\approx70$; arrow: $m=1.3\times10^4$)', fontsize=8); panel_label(ax, '(b)')
fig.tight_layout(w_pad=0.8); save(fig, 'fig8')

# ---------------------------------------------------------------- Fig 9: z=1.3 statistical evidence (R2 point 4)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
ax = axs[0]; Z = np.sort(T[('pm', 1.3)]['Z']); n = len(Z)
pp = (np.arange(1, n + 1) - 0.5) / n; tq = stats.norm.ppf(pp)
rng = np.random.default_rng(3); sims = np.sort(np.array([standardize(rng.standard_normal(n)) for _ in range(400)]), axis=1)
lo, hi = np.percentile(sims, 0.5, axis=0), np.percentile(sims, 99.5, axis=0)
ax.fill_between(tq, lo - tq, hi - tq, color='0.8', lw=0, label='Gaussian null 99%')
ax.plot(tq, Z - tq, '-', color=C[1], lw=1.1, label=r'PM $z=1.3$')
Z4 = np.sort(T[('log', 4.0, 'x')]['Z']); n4 = len(Z4); tq4 = stats.norm.ppf((np.arange(1, n4 + 1) - 0.5) / n4)
ax.plot(tq4, Z4 - tq4, '-', color=C[0], lw=1.0, label=r'logistic $r=4$')
ax.axhline(0, color='k', lw=0.6); ax.set_xlim(-4, 4); ax.set_ylim(-1.5, 0.8)
ax.set_xlabel('Gaussian quantile'); ax.set_ylabel('sample $-$ Gaussian quantile'); ax.legend(frameon=False, fontsize=7, loc='lower right')
panel_label(ax, '(a)')
ax = axs[1]; rep = E['rep']
names = [('log4', r'L $r=4$'), ('hen14_01', r'H $(1.4,0.1)$'), ('pm1.2', r'PM $1.2$'), ('pm1.3', r'PM $1.3$')]
data = [[max(r['kl_exc'], 1e-5) for r in rep[k]] for k, _ in names]
nl0 = gaussian_null(10000)['kl'] - np.mean(gaussian_null(10000)['kl'])
bp = ax.boxplot([np.maximum(nl0, 1e-5)] + data, widths=0.5, patch_artist=True, showfliers=False)
for patch, col in zip(bp['boxes'], ['0.8', C[0], C[2], C[4], C[1]]): patch.set_facecolor(col); patch.set_alpha(0.7)
for i, dd in enumerate(data):
    ax.plot(np.full(len(dd), i + 2) + np.random.default_rng(i).uniform(-0.12, 0.12, len(dd)), dd, '.', color='k', ms=2.5)
ax.set_xticks(range(1, 6)); ax.set_xticklabels(['Gaussian\nnull'] + [b for _, b in names], fontsize=7)
ax.set_yscale('log'); ax.set_ylabel(r'$\Delta D_{\rm KL}$ (20 independent ensembles)'); ax.axhline(np.percentile(nl0, 99), color='k', ls=':', lw=0.8)
panel_label(ax, '(b)')
fig.tight_layout(w_pad=0.8); save(fig, 'fig9')

# ---------------------------------------------------------------- Fig 10: robustness (binning + noise)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
b = E['bins']; B = b['B']
for j, (k, lab) in enumerate([('log3.6', r'L $r=3.6$'), ('log4.0', r'L $r=4$'), ('hen1.4', r'H $(1.4,0.3)$'), ('pm1.3', r'PM $1.3$'), ('pm1.5', r'PM $1.5$')]):
    axs[0].plot(B, b[k]['raw'], '--', color=C[j], marker=MK[j], ms=3, lw=0.8)
    axs[0].plot(B, np.maximum(b[k]['exc'], 1e-4), '-', color=C[j], marker=MK[j], ms=3, label=lab)
axs[0].plot(B, [b['null'][x][0] for x in B], ':', color='k', lw=1.2, label='null mean')
axs[0].set_yscale('log'); axs[0].set_xlabel('number of bins $B$'); axs[0].set_ylabel(r'$D_{\rm KL}$ (dashed) / $\Delta D_{\rm KL}$ (solid)')
axs[0].legend(frameon=False, fontsize=6.3, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.28)); panel_label(axs[0], '(a)')
nz = E['noise']
for j, (k, lab) in enumerate([('log3.6', r'L $r=3.6$'), ('log4.0', r'L $r=4$'), ('pm1.3', r'PM $1.3$')]):
    eps = [x['eps'] for x in nz[k]]; ee = np.array(eps); ee[0] = 3e-4
    axs[1].plot(ee, [abs(x['alpha'] - 1) for x in nz[k]], '-', color=C[[0, 1, 3][j]], marker=MK[j], ms=3.5, label=lab)
    axs[2].plot(ee, [x['kl_exc'] for x in nz[k]], '-', color=C[[0, 1, 3][j]], marker=MK[j], ms=3.5, label=lab)
for ax, yl in [(axs[1], r'NFI$_{\rm scale}$'), (axs[2], r'$\Delta D_{\rm KL}$')]:
    ax.set_xscale('log'); ax.set_yscale('log' if ax is axs[1] else 'symlog', **({} if ax is axs[1] else dict(linthresh=1e-3))); ax.set_xlabel(r'noise $\varepsilon$ (leftmost: $\varepsilon=0$)'); ax.set_ylabel(yl)
axs[1].legend(frameon=False, fontsize=6.5); axs[2].set_ylim(-2e-3, 3e-2); panel_label(axs[1], '(b)'); panel_label(axs[2], '(c)')
fig.tight_layout(w_pad=0.5); save(fig, 'fig10')

# ---------------------------------------------------------------- Supplementary figures
# S1 composite score weighting (same log scale in both panels)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
regs = [(3.6, 'L 3.6'), (3.8, 'L 3.8'), (4.0, 'L 4.0')]
for j, cc in enumerate([0.5, 1, 2]):
    vals = [T[('log', r, 'x')]['kl_exc'] + cc * abs(T[('log', r, 'x')]['alpha'] - 1) for r, _ in regs]
    axs[0].plot([r for r, _ in regs], np.maximum(vals, 1e-4), '-', color=C[j], marker=MK[j], label=r'$c=%.1f$' % cc)
    sel = (r_ >= 3.58) & (r_ <= 3.66) & (~per)
    axs[1].plot(r_[sel], np.maximum(kex[sel] + cc * nsc[sel], 1e-4), '-', color=C[j], marker=MK[j], ms=2.5, lw=0.8, label=r'$c=%.1f$' % cc)
for ax in axs: ax.set_yscale('log'); ax.set_xlabel(r'$r$')
axs[0].set_ylabel(r'$\Delta D_{\rm KL}+c\,|\alpha-1|$'); axs[0].legend(frameon=False); axs[1].set_title(r'$r\in[3.58,3.66]$, $\Delta r=0.001$', fontsize=8)
panel_label(axs[0], '(a)'); panel_label(axs[1], '(b)')
fig.tight_layout(); save(fig, 'figS1')

# S2 correlation functions
fig, ax = plt.subplots(1, 1, figsize=(5.4, 2.8))
for j, (k, lab) in enumerate([('log4.0', 'L 4.0'), ('log3.8', 'L 3.8'), ('hen1.4_0.3', 'H (1.4,0.3)'), ('pm1.3', 'PM 1.3'), ('pm1.5', 'PM 1.5'), ('pm1.8', 'PM 1.8'), ('pm2.0', 'PM 2.0'), ('log3.6', 'L 3.6 (two-band)'), ('hen1.4_0.1', 'H (1.4,0.1) (two-band)')]):
    c = E['corr'][k]; ax.loglog(c['lags'], np.maximum(np.abs(c['rho']), 1e-4), '-' if j < 7 else '--', color=C[j % 7] if j < 7 else '0.5', marker=MK[j % 7], ms=2.5, lw=0.8, label=lab)
ax.axhline(0.05, color='k', ls=':', lw=0.8); ax.set_xlabel(r'lag $\ell$'); ax.set_ylabel(r'$|C(\ell)/C(0)|$'); ax.legend(frameon=False, fontsize=6.5, ncol=1, loc='center left', bbox_to_anchor=(1.01, 0.5))
fig.tight_layout(); save(fig, 'figS2')

# S3 two-band logistic r=3.6
b = E['band']; va = b['var']; m = np.arange(1, len(va) + 1)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
ev = m % 2 == 0
axs[0].loglog(m[ev], va[ev], '-', color=C[0], lw=1.0, label='even $m$')
axs[0].loglog(m[~ev], va[~ev], '-', color=C[1], lw=1.0, label='odd $m$')
mm = np.logspace(1, np.log10(12800), 100); axs[0].loglog(mm, b['A'] + b['slope'] * mm, '--', color='k', lw=0.9, label=r'$A+2Dm$ (even $m$)')
axs[0].set_xlabel('$m$'); axs[0].set_ylabel(r'Var$(S_m)$'); axs[0].legend(frameon=False, fontsize=7)
lm = np.array([50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]); ls_ = np.diff(np.log(va[lm - 1])) / np.diff(np.log(lm))
axs[1].semilogx(np.sqrt(lm[1:] * lm[:-1]), ls_, 'o-', color=C[0]); axs[1].axhline(1, color='k', ls=':', lw=0.8)
mm2 = np.sqrt(lm[1:] * lm[:-1]); x = b['slope'] * mm2 / (b['A'] + b['slope'] * mm2); axs[1].semilogx(mm2, x, '--', color='k', lw=0.9, label=r'$2Dm/(A+2Dm)$')
axs[1].set_xlabel('$m$ (even grid)'); axs[1].set_ylabel(r'local exponent'); axs[1].legend(frameon=False, fontsize=7)
panel_label(axs[0], '(a)'); panel_label(axs[1], '(b)')
fig.tight_layout(); save(fig, 'figS3')
print('figures done')
