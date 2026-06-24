# Effective Potential: Dark Photon Model
### *RGE-improved effective potentials, bounce solvers, and GW predictions for a dark-photon model*

This repository contains a full numerical pipeline for studying first-order phase transitions in a dark U(1) scalar model. It combines RGE-running couplings, one-loop effective potentials, ODE-based and quartic-tunneling bounce solvers, nucleation analysis, and gravitational-wave spectra into one coherent framework. A fermion extension (Dirac fermion with Yukawa coupling) is included as a drop-in variant.

If you are here, you are probably either studying thermal field theory, debugging your own bounce solver at 3 am, or just curious about why the universe insists on changing phase. In any case, welcome.

---

## Repository Structure

```
FOPT_Dark_Photon_DRalgo/
│
├── src/
│   ├── constants.py                   ← physical constants + model initial conditions
│   └── RGE/
│       ├── RGEsolver.py               ← 1-loop RGE integration (bosonic model)
│       ├── RGESolver_fermion.py        ← 1-loop RGE integration (+ Dirac fermion)
│       ├── VeffFunc_RGE.py            ← effective potential (tree + CW + thermal + daisy)
│       ├── VeffFunc_ferm_RGE.py       ← effective potential with fermion loop + Yukawa
│       ├── bounce_solver.py            ← ODE shooting core: BounceProfile, Find_critbubble
│       ├── BounceODE_RGE.py           ← ODE bounce solvers: BounceODEHighT / Full / HighT_fermion
│       ├── BounceSolFull_RGE.py       ← quartic-tunneling bounce solver (full J_B)
│       ├── BounceSolHighT_RGE.py      ← quartic-tunneling bounce solver (high-T expansion)
│       ├── FOPT_RGE_real.py           ← FOPT utilities: Tn, Tc, α, β/H (solver-backed)
│       ├── GW_RGE_spectrum.py         ← generic GW spectrum from (Tn, α, β/H)
│       ├── GW_bubbles_LVV22.py        ← bubble-collision GW spectrum (LVV22)
│       └── GW_field_LV20.py           ← field-gradient / runaway GW spectrum (LV20)
│
├── notebooks/
│   ├── demo_RGE.ipynb                 ← RGE running demo
│   ├── demo_Veff.ipynb                ← effective potential demo
│   ├── demo_bounce_action.ipynb       ← bounce action computation demo
│   ├── demo_FOPT_params.ipynb         ← FOPT parameter computation demo
│   ├── demo_GW_spectrum.ipynb         ← gravitational-wave spectrum demo
│   ├── analysis_EFT_power_counting.ipynb  ← EFT validity / power-counting analysis
│   └── analysis_FOPT_yukawa_impact.ipynb  ← Yukawa coupling effects on FOPT parameters
│
├── utils/
│   └── plot_styles.py                 ← matplotlib style registry
│
├── data/
│   ├── raw_data/                      ← unprocessed bounce-action scans
│   ├── clean_data/                    ← cleaned grids
│   ├── final_data/                    ← processed outputs (Tn, α, β/H, GW)
│   └── other_data/                    ← external data (PTA, LISA sensitivity)
│
├── plots/
├── VT_integralNumeric.dat             ← tabulated J_B thermal integral
├── startup.py                         ← interactive session bootstrap
├── requirements.txt
├── REPORT.md                          ← change log for the 2026-06-02 refactor
└── README.md
```

---

## Pipeline Overview

The computation flows through five layers. Two bounce-solver families are available and can be swapped transparently via dependency injection into `FOPTUtilities`. The fermion extension adds a parallel RGE + potential track.

```
constants.py  +  RGEsolver.py  [or RGESolver_fermion.py]
        │
        ▼
  VeffFunc_RGE.py  [or VeffFunc_ferm_RGE.py]       ← model-specific layer
        │
        ├──► BounceSolFull_RGE.py     (quartic-tunneling, full J_B)
        ├──► BounceSolHighT_RGE.py    (quartic-tunneling, high-T expansion)
        ├──► BounceODEFull            (ODE shooting, full J_B)    ─┐  via BounceODE_RGE.py
        └──► BounceODEHighT           (ODE shooting, high-T)      ─┘
                    │
                    ▼
             FOPT_RGE_real.py
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

### 2. RGE Running — `RGEsolver.py` / `RGESolver_fermion.py`

**`RGEsolver`** evolves three couplings from `μ₀` to `μ = scale × T` using the bosonic one-loop beta functions:

| Coupling | Beta function |
|---|---|
| `gD²` | `gD⁴ / (24π²)` |
| `λS` | `[3gD⁴ − 6gD²λS + 10λS²] / (8π²)` |
| `mS` | `−mS(3gD² − 4λS) / (8π²)` |

**`RGESolver_fermion`** adds a Yukawa coupling `y` and updates the beta functions to include the Dirac fermion loop. Interface is identical to `RGEsolver` with an extra `y0` argument and a 4-tuple return `(gD, λS, mS², y)`.

Both solvers cache results with `lru_cache` and integrate in `log(μ)` with RK45.

---

### 3. Effective Potential — `VeffFunc_RGE.py` / `VeffFunc_ferm_RGE.py`

The full one-loop thermal effective potential:

```
V_eff(φ, T) = V_tree + V_CW + V_T + V_daisy
```

- **V_tree**: classical `−m²φ²/2 + λφ⁴/4`
- **V_CW**: Coleman–Weinberg in MS-bar (φ, σ, A' loops)
- **V_T**: finite-temperature correction via tabulated `J_B(m²/T²)`
- **V_daisy**: Arnold-Espinosa ring resummation with Debye masses `Π_φ`, `Π_A'`

**`VeffFunc_ferm_RGE`** extends this with:
- A fermion loop in `V_CW`: `−4 × m_χ⁴/(64π²) × (log … − 3/2)`
- A fermionic thermal function `J_F(m_χ²/T²)` in `V_T`
- Modified Debye masses: `Π_φ` gains `+y²/12 × T²`; `Π_A' = 5/12 × gD² × T²`
- Field-dependent fermion mass `m_χ = yφ/√2`

Both classes have identical interfaces — bounce solvers and `FOPTUtilities` accept either without modification.

**These are the only model-specific files.** See [Adapting to a New Model](#adapting-to-a-new-model).

---

### 4. Bounce Solvers

Two independent families are available. Both expose the same `SE(T, gD, scale, ls0)` interface and can be injected into `FOPTUtilities` via the `solver_ht` / `solver_full` keyword arguments.

#### Quartic-tunneling (Espinosa 1996) — `BounceSolFull_RGE.py` / `BounceSolHighT_RGE.py`

Construct a quartic tunneling-potential `V_t(φ)` matched to `V_eff` and integrate the thin-wall action analytically.

#### ODE shooting — `bounce_solver.py` + `BounceODE_RGE.py`

Integrate the full O(3) bounce equation numerically (shooting method) and compute `S₃` directly from the profile via the virial formula:

```
S₃ = 4π ∫ dr r² [ ½ (dφ/dr)² + V(φ) ]
```

`bounce_solver.py` contains the core `Find_critbubble` ODE integrator. `BounceODE_RGE.py` wraps it in three drop-in classes:

| Class | Potential used |
|---|---|
| `BounceODEHighT` | `Veff_HighT` (bosonic model) |
| `BounceODEFull` | `Veff` with full J_B (bosonic model) |
| `BounceODEHighT_fermion` | `Veff_HighT` (fermion model) |

The ODE solvers pre-bake RGE parameters once per `SE` call and use analytical `dV/dφ` (with a `CubicSpline` for the full thermal derivative), avoiding numerical noise and redundant RGE solves.

---

### 5. FOPT Utilities — `FOPT_RGE_real.py`

Computes phase-transition parameters by calling the bounce solvers directly at each `(T, gD)` point.

| Method | Description |
|---|---|
| `nucTemp(gD)` | Nucleation temperature `T_n` (Γ/H⁴ = 1) |
| `critTemp(gD)` | Critical temperature `T_c` (V_eff(φ_min) = 0) |
| `perTemp(gD)` | Percolation temperature (P_f = 0.71) |
| `alpha(T, gD)` | Transition strength α = −V_eff / ρ_R |
| `beta(T_star, gD)` | Inverse duration β/H ≈ T d(S/T)/dT |
| `Gamma(T, gD)` | Nucleation rate Γ(T) |
| `Pf(T, gD)` | False-vacuum probability P_f(T) |

The bounce solver family is selected at construction time via `solver_ht` / `solver_full` keyword arguments, so switching between quartic-tunneling and ODE solvers requires no other code changes.

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

# Build the effective potential (bosonic model)
from src.RGE.VeffFunc_RGE import VeffRGE
v = VeffRGE()      # loads VT_integralNumeric.dat

# Run the RGEs and evaluate V_eff at a point
import constants as cs
V = v.Veff(S=0.1, T=0.05, gD0=cs.gD0, scale=np.pi, lambdaS0=cs.lambdaS0)

# Compute FOPT parameters using the ODE bounce solver
from src.RGE.BounceODE_RGE import BounceODEHighT, BounceODEFull
from src.RGE.FOPT_RGE_real import FOPTUtilities
fopt = FOPTUtilities(veff_obj=v, solver_ht=BounceODEHighT(v), solver_full=BounceODEFull(v))
Tn = fopt.nucTemp(gD=0.6)

# GW spectrum
from src.RGE.GW_bubbles_LVV22 import spectrum_bubbles
h2 = spectrum_bubbles(Tn=Tn, alpha=10.0, beta_over_H=40.0)
f  = np.logspace(-10, -5, 300)
plt.loglog(f, h2(f))
```

#### Fermion model variant

```python
from src.RGE.VeffFunc_ferm_RGE import VeffRGE_fermion
from src.RGE.BounceODE_RGE import BounceODEHighT_fermion

vf   = VeffRGE_fermion(y0=0.3)
bsf  = BounceODEHighT_fermion(vf)
fopt = FOPTUtilities(veff_obj=vf, solver_ht=bsf)
Tn   = fopt.nucTemp(gD=0.6)
```

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `demo_RGE.ipynb` | Walk through the one-loop RGE running |
| `demo_Veff.ipynb` | Visualise the effective potential at various T |
| `demo_bounce_action.ipynb` | Compute and compare bounce actions from both solver families |
| `demo_FOPT_params.ipynb` | Compute Tn, Tc, α, β/H for a parameter point |
| `demo_GW_spectrum.ipynb` | Generate a GW spectrum and overlay LISA sensitivity |
| `analysis_EFT_power_counting.ipynb` | EFT validity and perturbativity checks |
| `analysis_FOPT_yukawa_impact.ipynb` | Yukawa coupling effects on FOPT parameters |

---

## Adapting to a New Model

### Bosonic model (no fermion)

Only three files need to change:

| File | What to change |
|---|---|
| `src/constants.py` | Initial conditions `gD0`, `lambdaS0`, `mS0`; add new entries to `LV20_PRESETS` if needed |
| `src/RGE/RGEsolver.py` | Beta functions `beta_gD2`, `beta_lambdaS`, `beta_mS2` |
| `src/RGE/VeffFunc_RGE.py` | Mass spectrum (`mPhi2`, `mSigma2`, `mAp2`), Debye masses (`PiPhi`, `PiAp`), and loop structure (`Vcw`, `Vdaisy`) |

### Fermion extension

To add a new fermion species, mirror the pattern of the existing extension:

| File | What to change |
|---|---|
| `src/RGE/RGESolver_fermion.py` | Beta functions updated for your fermion representation and charges |
| `src/RGE/VeffFunc_ferm_RGE.py` | Fermion mass `mchi2`, loop degeneracy, Debye mass corrections |

Everything else — bounce solvers, FOPT utilities, GW spectra — is model-independent.

---

## Author

Developed by **Maura Elizabeth Ramirez-Quezada** as part of ongoing research on first-order phase transitions, thermal field theory, and their cosmological signatures.
