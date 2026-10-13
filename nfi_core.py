"""
nfi_core.py  --  core routines for the finite-time NFI diagnostic (revision R1).

Changes with respect to the originally submitted code (sim1_core.py / bootstrap_se.py):
  * ensembles are iterated in vectorized form (identical dynamics, faster);
  * Z_{m*} is standardized with the ensemble mean and standard deviation of S_{m*};
  * the KL histogram keeps ALL samples: B equal-width bins on [-5,5] plus two
    open tail bins (-inf,-5) and (5,inf); reference probabilities are exact
    Gaussian bin masses (previously samples with |Z|>5 were silently dropped);
  * NFI_shape is reported both raw and null-corrected: the expectation of the
    same estimator for N i.i.d. Gaussian samples (same N, same bins) is
    subtracted, which removes the finite-sample/binning bias;
  * W1 is computed against the analytic N(0,1) CDF (previously against a random
    Gaussian sample);
  * Anderson-Darling statistic against the fully specified N(0,1);
  * p-values for KL, W1, AD, skewness and kurtosis from a Monte Carlo Gaussian null
    with the same N and the same standardization;
  * bootstrap over trajectories with 500 resamples (SE and percentile 95% CI).
"""
import numpy as np
from scipy import stats
from scipy.special import ndtr

# ----------------------------------------------------------------------------
# Maps (vectorized over the ensemble)
# ----------------------------------------------------------------------------

def run_logistic(r, N, M, burn=800, rng=None, g='x'):
    """Trajectories that land exactly on the fixed point x=0 (possible in double
    precision for r=4 via x=0.5 -> 1 -> 0) are re-drawn during burn-in and
    discarded if it happens during the measurement window (count in .n_dropped)."""
    rng = np.random.default_rng() if rng is None else rng
    x = rng.uniform(0.01, 0.99, N)
    for _ in range(burn):
        x = r * x * (1.0 - x)
        bad = (x <= 0.0) | (x >= 1.0)
        if bad.any():
            x[bad] = rng.uniform(0.01, 0.99, bad.sum())
    X = np.empty((N, M))
    lam = np.zeros(N); ok = np.ones(N, bool)
    for k in range(M):
        x = r * x * (1.0 - x)
        ok &= (x > 0.0) & (x < 1.0)
        X[:, k] = x
        lam += np.log(np.abs(r * (1.0 - 2.0 * x)) + 1e-300)
    G = X if g == 'x' else X ** 2
    run_logistic.n_dropped = int((~ok).sum())
    return G[ok], float(np.mean(lam[ok] / M))


def run_henon(a, b, N, M, burn=1500, rng=None, g='x'):
    rng = np.random.default_rng() if rng is None else rng
    out = []
    need = N
    while need > 0:
        n = int(need * 1.3) + 10
        x = rng.uniform(-0.5, 0.5, n); y = rng.uniform(-0.5, 0.5, n)
        ok = np.ones(n, bool)
        for _ in range(burn):
            x, y = 1.0 - a * x * x + y, b * x
            ok &= np.isfinite(x) & (np.abs(x) < 100)
            x = np.where(ok, x, 0.0); y = np.where(ok, y, 0.0)
        X = np.empty((n, M))
        for k in range(M):
            x, y = 1.0 - a * x * x + y, b * x
            ok &= np.isfinite(x) & (np.abs(x) < 100)
            x = np.where(ok, x, 0.0); y = np.where(ok, y, 0.0)
            X[:, k] = x
        out.append(X[ok][:need]); need -= out[-1].shape[0]
    X = np.vstack(out)[:N]
    return (X if g == 'x' else X ** 2)


def henon_lyap(a, b, n_ic=5, burn=2000, n_iter=50000, seed=7):
    rng = np.random.default_rng(seed); vals = []
    while len(vals) < n_ic:
        x, y = rng.uniform(-0.5, 0.5, 2)
        bad = False
        for _ in range(burn):
            x, y = 1 - a * x * x + y, b * x
            if abs(x) > 100: bad = True; break
        if bad: continue
        Q = np.eye(2); le = np.zeros(2)
        for _ in range(n_iter):
            J = np.array([[-2 * a * x, 1.0], [b, 0.0]])
            Q, R = np.linalg.qr(J @ Q); le += np.log(np.abs(np.diag(R)))
            x, y = 1 - a * x * x + y, b * x
        vals.append(np.max(le / n_iter))
    return float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def run_pm(z, N, M, burn=2000, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x = rng.uniform(0.01, 0.99, N)
    for _ in range(burn):
        x = np.mod(x + x ** z, 1.0)
    X = np.empty((N, M))
    lam = np.zeros(N)
    for k in range(M):
        lam += np.log(1.0 + z * x ** (z - 1.0))
        x = np.mod(x + x ** z, 1.0)
        X[:, k] = x
    return X, float(np.mean(lam / M))

# ----------------------------------------------------------------------------
# Partial sums
# ----------------------------------------------------------------------------

def partial_sums(G, m_list):
    """S_m for m in m_list (columns). Centering by the pooled mean (irrelevant
    for Var(S_m) and for the standardized Z)."""
    C = np.cumsum(G - G.mean(), axis=1)
    return {int(m): C[:, int(m) - 1].copy() for m in m_list}


def fit_alpha(ms, v):
    ms = np.asarray(ms, float); v = np.asarray(v, float)
    ok = np.isfinite(v) & (v > 0)
    if ok.sum() < 3:
        return np.nan, np.nan, np.nan
    lx, ly = np.log(ms[ok]), np.log(v[ok])
    res = stats.linregress(lx, ly)
    loc = np.diff(ly) / np.diff(lx)
    spread = float(loc.max() - loc.min())
    return float(res.slope), float(res.rvalue ** 2), spread


def local_slopes(ms, v):
    lx, ly = np.log(np.asarray(ms, float)), np.log(np.asarray(v, float))
    return np.exp(0.5 * (lx[1:] + lx[:-1])), np.diff(ly) / np.diff(lx)

# ----------------------------------------------------------------------------
# Shape statistics
# ----------------------------------------------------------------------------
ZR = 5.0

def standardize(S):
    s = np.std(S, ddof=1)
    if not np.isfinite(s) or s < 1e-12 * max(1.0, np.abs(S).max()):
        return None
    return (S - S.mean()) / s


def _edges(B):
    inner = np.linspace(-ZR, ZR, B + 1)
    return np.concatenate(([-np.inf], inner, [np.inf]))


def kl_hist(Z, B=80):
    """D_KL(p_hat || N(0,1)) with B bins on [-5,5] + 2 tail bins, Laplace
    pseudo-count 1 per bin, exact Gaussian bin masses as reference."""
    e = _edges(B)
    h = np.histogram(Z, bins=e)[0] + 1.0
    p = h / h.sum()
    q = np.diff(ndtr(e))
    return float(np.sum(p * np.log(p / q)))


_GRID = np.linspace(-12, 12, 24001)
_PHI = ndtr(_GRID)

def w1_gauss(Z):
    Fn = np.searchsorted(np.sort(Z), _GRID, side='right') / Z.size
    return float(np.trapezoid(np.abs(Fn - _PHI), _GRID))


def ad_gauss(Z):
    z = np.sort(Z); n = z.size
    F = np.clip(ndtr(z), 1e-300, 1 - 1e-16)
    i = np.arange(1, n + 1)
    return float(-n - np.mean((2 * i - 1) * (np.log(F) + np.log(1 - F[::-1]))))


def shape_stats(Z, B=80):
    return dict(kl=kl_hist(Z, B), w1=w1_gauss(Z), ad=ad_gauss(Z),
                skew=float(stats.skew(Z)), kurt=float(stats.kurtosis(Z)))

_NULL = {}

def gaussian_null(N, B=80, reps=2000, seed=12345):
    key = (N, B, reps)
    if key in _NULL:
        return _NULL[key]
    rng = np.random.default_rng(seed)
    rows = [shape_stats(standardize(rng.standard_normal(N)), B) for _ in range(reps)]
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    _NULL[key] = out
    return out


def null_pvalues(st, null):
    """one-sided for distances, two-sided for skewness/kurtosis."""
    p = {}
    for k in ('kl', 'w1', 'ad'):
        p[k] = float((1 + np.sum(null[k] >= st[k])) / (1 + null[k].size))
    for k in ('skew', 'kurt'):
        p[k] = float((1 + np.sum(np.abs(null[k]) >= abs(st[k]))) / (1 + null[k].size))
    return p

# ----------------------------------------------------------------------------
# Full diagnostic with bootstrap
# ----------------------------------------------------------------------------

def diagnose(G, m_grid, m_star=800, B=80, n_boot=500, seed=2026, null_reps=2000):
    m_grid = np.asarray(m_grid, int)
    allm = sorted(set(m_grid.tolist()) | {int(m_star)})
    S = partial_sums(G, allm)
    N = G.shape[0]
    v = np.array([np.var(S[m], ddof=1) for m in m_grid])
    alpha, R2, spread = fit_alpha(m_grid, v)
    Z = standardize(S[m_star])
    res = dict(N=N, m_grid=m_grid.tolist(), var=v.tolist(), alpha=alpha, R2=R2,
               loc_spread=spread, m_star=int(m_star), B=B)
    if Z is None:
        res.update(degenerate=True)
        return res
    st = shape_stats(Z, B)
    null = gaussian_null(N, B, null_reps)
    res.update(degenerate=False, **st)
    res['kl0'] = float(np.mean(null['kl']))
    res['kl_exc'] = st['kl'] - res['kl0']
    res['p'] = null_pvalues(st, null)
    res['Z'] = Z
    # bootstrap over trajectories
    rng = np.random.default_rng(seed)
    keys = ['alpha', 'kl', 'kl_exc', 'w1', 'ad', 'skew', 'kurt']
    bs = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        idx = rng.integers(0, N, N)
        vb = np.array([np.var(S[m][idx], ddof=1) for m in m_grid])
        bs['alpha'][b] = fit_alpha(m_grid, vb)[0]
        Zb = standardize(S[m_star][idx])
        sb = shape_stats(Zb, B)
        for k in ('kl', 'w1', 'ad', 'skew', 'kurt'):
            bs[k][b] = sb[k]
        bs['kl_exc'][b] = sb['kl'] - res['kl0']
    for k in keys:
        res['se_' + k] = float(np.std(bs[k], ddof=1))
        res['ci_' + k] = [float(np.percentile(bs[k], 2.5)), float(np.percentile(bs[k], 97.5))]
    return res


def tau_int(G, max_lag=2000):
    """integrated autocorrelation time 1+2*sum rho(l), summed until rho first
    drops below 0.05 (or max_lag), averaged over the ensemble."""
    c = G - G.mean()
    var = np.mean(c * c)
    tau = 1.0
    for l in range(1, min(max_lag, G.shape[1] - 1)):
        rho = np.mean(c[:, :-l] * c[:, l:]) / var
        if abs(rho) < 0.05:
            return tau, l
        tau += 2 * rho
    return tau, max_lag


# ----------------------------------------------------------------------------
# Streaming versions (store only partial sums at requested m; used for long m)
# ----------------------------------------------------------------------------

def stream_sums(step, x0, burn, m_list, obs=lambda x: x):
    """Iterate x -> step(x) for burn steps, then accumulate raw partial sums of
    obs(x) and return {m: S_m} for m in m_list. Centering is unnecessary: Var(S_m)
    and the standardized Z_m are invariant under subtraction of m*<g>."""
    x = x0
    for _ in range(burn):
        x = step(x)
    want = set(int(m) for m in m_list); M = max(want)
    S = np.zeros_like(obs(x) if not isinstance(x, tuple) else obs(x)); out = {}
    for k in range(1, M + 1):
        x = step(x)
        S = S + obs(x)
        if k in want:
            out[k] = S.copy()
    return out


def stream_logistic(r, x0, burn, m_list, rng):
    """Streaming logistic iteration with the same fixed-point guard as run_logistic."""
    x = x0.copy()
    for _ in range(burn):
        x = r * x * (1.0 - x)
        bad = (x <= 0.0) | (x >= 1.0)
        if bad.any():
            x[bad] = rng.uniform(0.01, 0.99, bad.sum())
    want = set(int(m) for m in m_list); M = max(want)
    S = np.zeros_like(x); ok = np.ones(x.size, bool); out = {}
    for k in range(1, M + 1):
        x = r * x * (1.0 - x); ok &= (x > 0.0) & (x < 1.0); S = S + x
        if k in want:
            out[k] = S.copy()
    stream_logistic.n_dropped = int((~ok).sum())
    return {m: v[ok] for m, v in out.items()}
