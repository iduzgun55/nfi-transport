"""LaTeX tables for the revised manuscript, generated directly from data/*.pkl."""
import os
import pickle, numpy as np
T = pickle.load(open('data/tables.pkl', 'rb')); E = pickle.load(open('data/extra.pkl', 'rb'))
def pfmt(p): return r'$<10^{-3}$' if p < 1e-3 else ('%.3f' % p if p < 0.01 else '%.2f' % p)
def ci(d): return '%.3f [%.3f, %.3f]' % (d['alpha'], d['ci_alpha'][0], d['ci_alpha'][1])
def flag(d): return '' if (d['R2'] >= 0.99 and round(d['loc_spread'], 2) <= 0.25) else r'$^\dagger$'
def sgn(x, f='%.4f'):
    s = f % x
    if float(s) == 0: s = s.lstrip('-')
    return s.replace('-', r'$-$')
L = []
# ---------------- Table I logistic
L.append(r'''\begin{table*}[t]
\caption{\label{tab:logistic}Logistic map, $N_{\rm traj}=10^4$, $m^{*}=800$, $m$-grid $\{50,\dots,1600\}$. $\alpha$: OLS exponent with bootstrap 95\% CI (500 resamples); $\Delta\alpha_{\rm loc}$: spread of local slopes between successive grid points; $\dagger$: single power law not adequate ($R^2<0.99$ or $\Delta\alpha_{\rm loc}>0.25$). $\mathrm{NFI}_{\rm shape}=\Delta D_{\rm KL}$ is the null-corrected divergence (null mean $\bar D_0=0.0092$, null s.d.\ $0.0006$); $p$ values are from the Gaussian null (2000 replicates of $10^4$ samples). Skewness and excess kurtosis are given $\pm$ bootstrap SE. Periodic case: $S_m$ is bounded, no transport.}
\footnotesize\setlength{\tabcolsep}{2pt}
\begin{ruledtabular}
\begin{tabular}{cclcccccccc}
$r$ & $g$ & regime & $\lambda$ & $\alpha$ [95\% CI] & $R^2$ & $\Delta\alpha_{\rm loc}$ & $\mathrm{NFI}_{\rm scale}$ & $\mathrm{NFI}_{\rm shape}$ ($p$) & skewness & ex.\ kurtosis\\ \hline''')
lab = {3.5: 'periodic (period 4)', 3.6: 'two-band chaotic', 3.8: 'chaotic', 4.0: 'fully chaotic'}
for r in [3.5, 3.6, 3.8, 4.0]:
    for g, gl in [('x', '$x$'), ('x2', '$x^2$')]:
        d = T[('log', r, g)]
        if d['degenerate']:
            L.append(r'%.2f & %s & %s & $%s$ & \multicolumn{7}{c}{bounded $S_m$ (no transport); NFI not applicable}\\' % (r, gl, lab[r], '-0.873'.replace('-', '-')) )
            continue
        L.append(r'%.2f & %s & %s & %.3f & %s%s & %.4f & %.2f & %.3f & %s (%s) & %s $\pm$ %.2f & %s $\pm$ %.2f\\' % (
            r, gl, lab[r], d['lam'], ci(d), flag(d), d['R2'], d['loc_spread'], abs(d['alpha'] - 1), sgn(d['kl_exc']), pfmt(d['p']['kl']),
            sgn(d['skew'], '%.2f'), d['se_skew'], sgn(d['kurt'], '%.2f'), d['se_kurt']))
L.append(r'''\end{tabular}
\end{ruledtabular}
\end{table*}
''')
# ---------------- Table II Henon
L.append(r'''\begin{table*}[t]
\caption{\label{tab:henon}H\'{e}non map, same settings as Table~\ref{tab:logistic}; $\lambda_1$: largest Lyapunov exponent (QR method, five initial conditions, s.e.\ $<0.002$). For $(a,b)=(1.3,0.3)$, $\lambda_1<0$ and more than $99.6\%$ of the trajectories lie on a period-7 orbit: $S_m$ is bounded and the diagnostic is not applicable.}
\footnotesize\setlength{\tabcolsep}{2pt}
\begin{ruledtabular}
\begin{tabular}{cclcccccccc}
$(a,b)$ & $g$ & regime & $\lambda_1$ & $\alpha$ [95\% CI] & $R^2$ & $\Delta\alpha_{\rm loc}$ & $\mathrm{NFI}_{\rm scale}$ & $\mathrm{NFI}_{\rm shape}$ ($p$) & skewness & ex.\ kurtosis\\ \hline''')
for (a, b, g, gl, lb) in [(1.4, 0.3, 'x', '$x$', 'chaotic'), (1.4, 0.3, 'x2', '$x^2$', 'chaotic'), (1.4, 0.1, 'x', '$x$', 'two-band chaotic'), (1.4, 0.1, 'x2', '$x^2$', 'two-band chaotic')]:
    d = T[('hen', a, b, g)]
    L.append(r'$(%.1f,%.1f)$ & %s & %s & %.3f & %s%s & %.4f & %.2f & %.3f & %s (%s) & %s $\pm$ %.2f & %s $\pm$ %.2f\\' % (
        a, b, gl, lb, d['lam'], ci(d), flag(d), d['R2'], d['loc_spread'], abs(d['alpha'] - 1), sgn(d['kl_exc']), pfmt(d['p']['kl']),
        sgn(d['skew'], '%.2f'), d['se_skew'], sgn(d['kurt'], '%.2f'), d['se_kurt']))
d = T[('hen', 1.3, 0.3, 'x')]
L.append(r'$(1.3,0.3)$ & $x$, $x^2$ & periodic (period 7) & $-$%.3f & \multicolumn{7}{c}{bounded $S_m$ (no transport); NFI not applicable}\\' % abs(d['lam']))
L.append(r'''\end{tabular}
\end{ruledtabular}
\end{table*}
''')
# ---------------- Table III PM
cor = E['corr']
th = {1.2: r'CLT, $\alpha\to1$', 1.3: r'CLT, $\alpha\to1$', 1.5: r'$\sqrt{m\ln m}$ CLT', 1.8: r'stable, index $1/(z-1)$', 2.0: r'infinite measure', 2.5: r'infinite measure'}
L.append(r'''\begin{table*}[t]
\caption{\label{tab:pm}Pomeau--Manneville map $x\mapsto x+x^z \bmod 1$, $g=x$, $N_{\rm traj}=10^4$, $m^{*}=800$, $m$-grid $\{50,\dots,1600\}$. ``Asymptotic law'' summarizes the rigorous results of Refs.~\onlinecite{Gouezel2004,Aaronson1997} for $\gamma=z-1$ (Sec.~\ref{sec:pmtheory}). $g_{800}=\mathrm{Var}(S_{800})/[800\,C(0)]$ is the statistical inefficiency (an integrated correlation time). Other columns as in Table~\ref{tab:logistic}.}
\footnotesize\setlength{\tabcolsep}{2pt}
\begin{ruledtabular}
\begin{tabular}{ccccccccccc}
$z$ & asymptotic law & $\lambda$ & $g_{800}$ & $\alpha$ [95\% CI] & $R^2$ & $\Delta\alpha_{\rm loc}$ & $\mathrm{NFI}_{\rm scale}$ & $\mathrm{NFI}_{\rm shape}$ ($p$) & skewness & ex.\ kurtosis\\ \hline''')
for z in [1.2, 1.3, 1.5, 1.8, 2.0, 2.5]:
    d = T[('pm', z)]; c = cor['pm%.1f' % z]
    L.append(r'%.1f & %s & %.3f & %.1f & %s%s & %.4f & %.2f & %.3f & %s (%s) & %s $\pm$ %.2f & %s $\pm$ %.2f\\' % (
        z, th[z], d['lam'], c['ineff'], ci(d), flag(d), d['R2'], d['loc_spread'], abs(d['alpha'] - 1), sgn(d['kl_exc']), pfmt(d['p']['kl']),
        sgn(d['skew'], '%.2f'), d['se_skew'], sgn(d['kurt'], '%.2f'), d['se_kurt']))
L.append(r'''\end{tabular}
\end{ruledtabular}
\end{table*}
''')
# ---------------- Table IV alternative measures
L.append(r'''\begin{table}[t]
\caption{\label{tab:alt}Complementary shape measures at $m^{*}=800$ ($N_{\rm traj}=10^4$, $g=x$): null-corrected KL divergence, Wasserstein-1 distance $W_1$ and Anderson--Darling statistic $A^2$ against $\mathcal{N}(0,1)$, each with its Gaussian-null $p$ value (99th null percentiles: $\Delta D_{\rm KL}=0.0016$, $W_1=0.0145$, $A^2=1.10$).}
\footnotesize\setlength{\tabcolsep}{2pt}
\begin{ruledtabular}
\begin{tabular}{lccc}
system & $\Delta D_{\rm KL}$ ($p$) & $W_1$ ($p$) & $A^2$ ($p$)\\ \hline''')
rows = [(('log', 3.6, 'x'), r'logistic $r=3.6$'), (('log', 3.8, 'x'), r'logistic $r=3.8$'), (('log', 4.0, 'x'), r'logistic $r=4.0$'),
        (('hen', 1.4, 0.3, 'x'), r'H\'{e}non $(1.4,0.3)$'), (('hen', 1.4, 0.1, 'x'), r'H\'{e}non $(1.4,0.1)$')] + [(('pm', z), r'PM $z=%.1f$' % z) for z in [1.2, 1.3, 1.5, 1.8, 2.0, 2.5]]
for k, nm in rows:
    d = T[k]
    L.append(r'%s & %s (%s) & %.4f (%s) & %.2f (%s)\\' % (nm, sgn(d['kl_exc']), pfmt(d['p']['kl']), d['w1'], pfmt(d['p']['w1']), d['ad'], pfmt(d['p']['ad'])))
L.append(r'''\end{tabular}
\end{ruledtabular}
\end{table}
''')
txt = '\n'.join(L)
parts = txt.split('\\begin{table')
names = ['logistic', 'henon', 'pm', 'alt']
for nm, part in zip(names, parts[1:]):
    open('tables/tables_R1_%s.tex' % nm, 'w').write('\\begin{table' + part)
print('\n'.join(L))
