# NFI-Transport

Code and data for:

> İ. Düzgün, "Finite-time non-Fickian transport signatures in deterministic chaotic maps," *Chaos: An Interdisciplinary Journal of Nonlinear Science* (revised version, 2026).

## Contents (revised version, R1)

| File | Description |
|------|-------------|
| `nfi_core.py` | Maps (logistic, Hénon, Pomeau–Manneville), partial sums, scale component, null-calibrated KL divergence, Wasserstein-1, Anderson–Darling, Gaussian null, bootstrap. |
| `run_tables.py` | Tables I–IV ($N_\mathrm{traj}=10^4$, 500 bootstrap resamples). Writes `data/tables.pkl`. |
| `run_mstar.py` | Observation-time dependence up to $m=12\,800$; burn-in and ensemble-size checks. Writes `data/mstar.pkl`. |
| `run_extra.py` | Replicate ensembles, $\Delta r=0.001$ logistic scan, observational noise, binning, two-band analysis, correlation functions. Writes `data/extra.pkl`. |
| `check_z18_ensemble.py` | Ensemble-size test of the variance exponent at $z=1.8$ (Supplementary Table S1). |
| `make_tables.py` | LaTeX tables generated directly from the data. |
| `gen_figs_R1.py` | All figures of the main text and Supplementary Material. |
| `original_submission/` | Scripts of the originally submitted version (kept for reference). |

## Usage

```bash
pip install numpy scipy matplotlib
mkdir -p data figs tables
python run_tables.py     # ~3 min
python run_mstar.py      # ~3 min
python run_extra.py      # ~3 min
python check_z18_ensemble.py   # ~2 min, writes data/z18_ens.pkl
python make_tables.py
python gen_figs_R1.py
```

Random seeds are fixed in each script. The ensemble-size rows for PM $z=1.8$ in Supplementary Table S1 are the means of the exponents stored in `data/z18_ens.pkl` (5 independent ensembles for $N_\mathrm{traj}\le4\times10^4$, 2 otherwise). The scripts in `original_submission/` are kept for historical comparison only and are not used to produce any result of the revised version. Default settings: $m^*=800$, $m$-grid $\{50,100,200,400,800,1200,1600\}$, 80 bins on $[-5,5]$ plus two tail bins with Laplace smoothing, Gaussian null with 2000 replicates, 500 bootstrap resamples.

## Changes with respect to the submitted version

* KL histogram keeps all samples (two open tail bins) and is corrected by its Gaussian-null expectation.
* $W_1$ against the analytic $\mathcal{N}(0,1)$; Anderson–Darling against the fully specified $\mathcal{N}(0,1)$ with null-calibrated $p$ values.
* 500 bootstrap resamples for all systems (the submitted scripts used 60 and 30).
* Same $m$-grid (up to 1600) for all maps; ensembles of $10^4$ trajectories.
* Guard against double-precision collapse onto the fixed point $x=0$ of the logistic map at $r=4$.

## License
MIT

## Contact
İbrahim Düzgün, ibrahimduzgun@gumushane.edu.tr, Department of Physics Engineering, Gümüşhane University, Türkiye
