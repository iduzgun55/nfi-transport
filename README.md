# NFI-Transport

**Non-Fickianity Index (NFI): A Finite-Time Transport Diagnostic for Deterministic Dynamics**

This repository contains the simulation scripts, figure-generation code, and raw data accompanying the manuscript:

> İ. Düzgün, "Finite-Time Deviation from Fickian Transport in Deterministic Dynamics: A Reproducible Diagnostic Framework," submitted to *Chaos: An Interdisciplinary Journal of Nonlinear Science* (2026).

## Contents

| File | Description |
|------|-------------|
| `sim1_core.py` | Core simulations: logistic map, Hénon map, Pomeau–Manneville map. Generates Tables I–III data. |
| `sim2_extra.py` | Extended analyses: m★ dependence, noise robustness, NFI vs r scan, binning/weighting sensitivity. |
| `gen_figs.py` | Generates all 12 figures (Figs. 1–12) from simulation data. |
| `figs/` | Output directory for generated PDF figures. |

## Requirements

- Python ≥ 3.8
- NumPy, SciPy, Matplotlib

```bash
pip install numpy scipy matplotlib
```

## Usage

Run in order:

```bash
python sim1_core.py      # ~60 sec — generates core data (sim_data.pkl)
python sim2_extra.py     # ~60 sec — generates extended data (sim_extra.pkl)
python gen_figs.py       # ~10 sec — generates all figures in figs/
```

All figures are saved as PDF in the `figs/` directory.

## Reproducibility

- Random seed is fixed (`np.random.seed(42)`) for full reproducibility.
- Default parameters: m★ = 800, 80 histogram bins with Laplace smoothing, bootstrap SE with 60 (logistic) / 30 (Hénon, PM) resamples.
- All numerical settings match those reported in the manuscript.

## License

MIT

## Contact

İbrahim Düzgün — ibrahim.duzgun@gumushane.edu.tr  
Department of Physics Engineering, Gümüşhane University, Türkiye
