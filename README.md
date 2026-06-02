# Effective Potential: Dark Photon Model
### *RGE-improved effective potentials, bounce solvers, and GW predictions for a dark-photon model*

This repository contains a full numerical pipeline for studying first-order phase transitions in a dark U(1) scalar model. It combines RGE-running couplings, one-loop effective potentials, bounce-equation solvers, nucleation analysis, and gravitational-wave spectra into one coherent framework.

If you are here, you are probably either studying thermal field theory, debugging your own bounce solver at 3 am, or just curious about why the universe insists on changing phase. In any case, welcome.

---

## Repository Structure

```
FOPT_Dark_Photon_DRalgo/
│
├── src/
│   ├── constants.py              ← physical constants + model initial conditions
│   ├── RGE/
│   │   ├── RGEsolver.py          ← 1-loop RGE integration
│   │   ├── VeffFunc_RGE.py       ← effective potential (tree + CW + thermal + daisy)
│   │   ├── BounceSolFull_RGE.py  ← bounce solver (full J_B potential)
│   │   ├── BounceSolHighT_RGE.py ← bounce solver (high-T expansion)
│   │   ├── SE_interpolator.py    ← spline over pre-computed S_E grid
│   │   ├── FOPT_RGE.py           ← FOPT utilities: Tn, Tc, α, β/H (spline-backed)
│   │   ├── FOPT_RGE_real.py      ← FOPT utilities: Tn, Tc, α, β/H (solver-backed)
│   │   ├── GW_RGE_spectrum.py    ← generic GW spectrum from (Tn, α, β/H)
│   │   ├── GW_bubbles_LVV22.py   ← bubble-collision GW spectrum (LVV22)
│   │   └── GW_field_LV20.py      ← field-gradient / runaway GW spectrum (LV20)
│   └── noRGE/                    ← legacy fixed-scale code (reference only)
│
├── notebooks/
│   ├── FOPT_RGE_check.ipynb
│   ├── GW_RGE.ipynb
│   ├── S_RGE_Full_plot.ipynb
│   ├── S_RGE_HT_plot.ipynb
│   ├── Veff_RGE_plot.ipynb
│   └── plots.ipynb
│
├── utils/
│   └── plot_styles.py            ← matplotlib style registry
│
├── data/
│   ├── raw_data/                 ← unprocessed bounce-action scans
│   ├── clean_data/               ← cleaned grids used by SE_interpolator
│   ├── final_data/               ← processed outputs (Tn, α, β/H, GW)
│   └── other_data/               ← external data (PTA, LISA sensitivity)
│
├── plots/
├── VT_integralNumeric.dat        ← tabulated J_B thermal integral
├── startup.py                    ← interactive session bootstrap
├── requirements.txt
├── REPORT.md                     ← change log for the 2026-06-02 refactor
└── README.md
```

---

## Pipeline Overview

The computation flows through five layers:

```
constants.py  +  RGEsolver.py
        │
        ▼
  VeffFunc_RGE.py          ← model-specific layer
        │
        ├──► BounceSolFull_RGE.py   (full J_B)
        └──► BounceSolHighT_RGE.py  (high-T expansion)
                    │
                    ├──► SE_interpolator.py  ──► FOPT_RGE.py       (fast, spline)
                    └──────────────────────────► FOPT_RGE_real.py  (on-the-fly)
                                                        │
                                                        ▼
                                          GW_RGE_spectrum.py
                                          GW_bubbles_LVV22.py
                                          GW_field_LV20.py
```

---

## Core Components

### 1. Model Parameters — `constants.py`

Stores physical constants, cosmological parameters, detector settings (LISA), and the RGE initial conditions at `μ₀ = 1 GeV`:

```python
gD0      = 0.6     # dark gauge coupling
lambdaS0 = 1e-10   # scalar quartic
mS0      = 1e-10   # scalar mass parameter [GeV]
```

These are the only values that need to change when scanning parameter space.

---

### 2. RGE Running — `RGEsolver.py`

Evolves three couplings from `μ₀` to `μ = scale × T` using the one-loop beta functions:

| Coupling | Beta function |
|---|---|
| `gD²` | `gD⁴ / (24π²)` |
| `λS` | `[3gD⁴ − 6gD²λS + 10λS²] / (8π²)` |
| `mS` | `−mS(3gD² − 4λS) / (8π²)` |

Integration is done in `log(μ)` with RK45 for numerical stability.

---

### 3. Effective Potential — `VeffFunc_RGE.py`

The full one-loop thermal effective potential:

```
V_eff(φ, T) = V_tree + V_CW + V_T + V_daisy
```

- **V_tree**: classical `−m²φ²/2 + λφ⁴/4`
- **V_CW**: Coleman–Weinberg in MS-bar (φ, σ, A' loops)
- **V_T**: finite-temperature correction via tabulated `J_B(m²/T²)`
- **V_daisy**: Arnold-Espinosa ring resummation with Debye masses `Π_φ`, `Π_A'`

Two variants are available: `Veff` (full `J_B` integral) and `Veff_HighT` (high-T expansion).

**This is the only model-specific file.** See [Adapting to a New Model](#adapting-to-a-new-model).

---

### 4. Bounce Solvers — `BounceSolFull_RGE.py` / `BounceSolHighT_RGE.py`

Both solvers implement the **quartic tunnelling-potential ansatz** (Espinosa 1996):

1. Locate the barrier top `φ_max` and broken minimum `φ_min`.
2. Scan for the first zero of `V_eff` beyond the barrier (`φ_root`).
3. Construct a quartic `V_t(φ)` matched to `V_eff` at `φ₀` and `φ_max`.
4. Integrate the thin-wall action: `S_E ∝ ∫ (V_eff − V_t)^{3/2} / (dV_t/dφ)² dφ`.
5. Minimise `S_E(φ₀)` over `φ₀ ∈ [φ_root, φ_min]`.

`BounceSolFull_RGE` uses `Veff`; `BounceSolHighT_RGE` uses `Veff_HighT`.  
Both accept any `VeffRGE` instance so the potential can be swapped transparently.

---

### 5. FOPT Utilities — `FOPT_RGE.py` / `FOPT_RGE_real.py`

Both modules expose the same interface:

| Method | Description |
|---|---|
| `nucTemp(gD)` | Nucleation temperature `T_n` (Γ/H⁴ = 1) |
| `critTemp(gD)` | Critical temperature `T_c` (V_eff(φ_min) = 0) |
| `perTemp(gD)` | Percolation temperature (P_f = 0.71) |
| `alpha(T, gD)` | Transition strength α = −V_eff / ρ_R |
| `beta(T_star, gD)` | Inverse duration β/H ≈ T d(S/T)/dT |
| `Gamma(T, gD)` | Nucleation rate Γ(T) |
| `Pf(T, gD)` | False-vacuum probability P_f(T) |

**`FOPT_RGE`** reads `S_E` from the pre-computed spline in `SE_interpolator` — fast for parameter scans.  
**`FOPT_RGE_real`** calls the bounce solvers on demand — slower but requires no pre-computed grid.

---

### 6. Gravitational-Wave Spectra

Three modules cover different source mechanisms:

| Module | Source | Reference |
|---|---|---|
| `GW_RGE_spectrum.py` | Sound waves + turbulence + bubble collisions | Espinosa et al. (2010) |
| `GW_bubbles_LVV22.py` | Bubble collisions (supercooling, no fluid) | Lewicki, Vaskonen & von Harling (2022) |
| `GW_field_LV20.py` | Field-gradient / runaway walls | Lewicki & Vaskonen (2020) |

All three take `(Tn, α, β/H)` as inputs and return a callable `h²Ω(f)` evaluated today. LISA SNR computation is included.

---

## Installation

Python ≥ 3.9. Install dependencies:

```bash
pip install -r requirements.txt
```

Required packages: `numpy`, `scipy`, `matplotlib`, `jupyter`, `tqdm`.

---

## Quick Start

```python
# In a notebook or IPython session:
%run startup.py    # loads all modules into the global namespace

# Build the effective potential
from src.RGE.VeffFunc_RGE import VeffRGE
v = VeffRGE()      # loads VT_integralNumeric.dat

# Run the RGEs and evaluate V_eff at a point
import constants as cs
V = v.Veff(S=0.1, T=0.05, gD0=cs.gD0, scale=np.pi, lambdaS0=cs.lambdaS0)

# Compute nucleation temperature (spline mode, requires pre-computed grid)
from src.RGE.FOPT_RGE import FOPTUtilities
fopt = FOPTUtilities(veff_obj=v)
Tn = fopt.nucTemp(gD=0.6)

# GW spectrum
from src.RGE.GW_bubbles_LVV22 import spectrum_bubbles
h2 = spectrum_bubbles(Tn=Tn, alpha=10.0, beta_over_H=40.0)
f  = np.logspace(-10, -5, 300)
plt.loglog(f, h2(f))
```

---

## Adapting to a New Model

To study a different scalar model, only three files need to change:

| File | What to change |
|---|---|
| `src/constants.py` | Initial conditions `gD0`, `lambdaS0`, `mS0`; add new entries to `LV20_PRESETS` if needed |
| `src/RGE/RGEsolver.py` | Beta functions `beta_gD2`, `beta_lambdaS`, `beta_mS2` |
| `src/RGE/VeffFunc_RGE.py` | Mass spectrum (`mPhi2`, `mSigma2`, `mAp2`), Debye masses (`PiPhi`, `PiAp`), and loop structure (`Vcw`, `Vdaisy`) |

Everything else — bounce solvers, FOPT utilities, GW spectra — is model-independent.

---

## Author

Developed by **Maura Elizabeth Ramirez-Quezada** as part of ongoing research on first-order phase transitions, thermal field theory, and their cosmological signatures.
