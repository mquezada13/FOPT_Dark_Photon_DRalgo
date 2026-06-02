# Pipeline Refactoring Report

**Project:** FOPT Dark Photon — DRalgo  
**Date:** 2026-06-02  
**Scope:** Cleanup, bug fixes, documentation, and modularity improvements.  
Physics (formulas, numerical values, algorithm logic) is unchanged.

---

## Summary

The pipeline was audited module-by-module.  Changes fall into four categories:

| Category | Count |
|---|---|
| Bug fixes (would cause runtime errors) | 4 |
| Cleanup (unused code, Spanish comments, style) | 13 files |
| Modularity improvements | 3 |
| Documentation (docstrings, module headers) | all files |

---

## Module-by-module changes

### `src/constants.py`

| Change | Reason |
|---|---|
| Added `Hhund = H_HUND` alias | `GW_field_LV20` imports `cs.Hhund`; the constant existed as `H_HUND` only — runtime `AttributeError` |
| Added `SNRthr = SNR_THR` alias | Same issue; `GW_field_LV20.h2Omega_thr` used `cs.SNRthr` |
| Added `LV20_PRESETS` dict with `"U1"` entry | `GW_field_LV20` expects `cs.LV20_PRESETS[lv20_key]`; missing previously caused `AttributeError`; entry is flagged TODO for users to verify values against the paper |
| Fixed spacing `lambdaS0= 1e-10` → `lambdaS0 = 1e-10` | PEP 8 |
| Added full module docstring | Documentation |

---

### `src/RGE/VeffFunc_RGE.py`

| Change | Reason |
|---|---|
| Removed `import cmath` | Unused — complex arithmetic is handled via `complex(1,0)` already |
| Removed `Imi = complex(0., 1.)` class variable | Never referenced anywhere in the codebase |
| Renamed `Re1` to `_Re1` (private) | Signals internal use |
| Parametrized `Veff0(S, gD0, ls0=cs.lambdaS0, mS0=cs.mS0)` | **Modularity**: `Veff0` previously hardcoded `cs.lambdaS0` and `cs.mS0` directly, making it impossible to study a different model without editing the source.  Now consistent with `Veff0_RGE`, `Veff`, and `Veff_HighT` |
| Updated `Veff` and `Veff_HighT` to pass `lambdaS0` to `Veff0` | Follows from the above; keeps the T=0 fallback branch consistent |
| Restructured all docstrings (NumPy style) | Documentation |
| Added module-level docstring explaining the pipeline position and how to extend to a new model | Documentation |

---

### `src/RGE/RGEsolver.py`

| Change | Reason |
|---|---|
| Renamed `RGEs_logMu` to `_rhs` (private) | The public name was only used internally; `run_params` is the public API |
| Added `rtol=1e-8, atol=1e-10` to `solve_ivp` | Tighter tolerances than the default `1e-3` / `1e-6`, appropriate for physics accuracy |
| Added NumPy-style docstrings to all methods | Documentation |
| Added module docstring | Documentation |

---

### `src/RGE/BounceSolFull_RGE.py`

| Change | Reason |
|---|---|
| Removed dated author/version header block | Superseded by git history |
| Renamed `get_intervals` to `_get_search_interval` (private, simplified) | Returns a single interval tuple instead of three; callers only used the first element |
| Replaced `phi_min` / `phi_max` / `phi_root` to use `_get_search_interval` consistently | Reduces duplication |
| Added `verbose` flag docs, `debug` flag docs | Documentation |
| Improved all docstrings (NumPy style, parameter descriptions) | Documentation |
| Added module docstring | Documentation |

---

### `src/RGE/BounceSolHighT_RGE.py`

| Change | Reason |
|---|---|
| Removed dated author/version header block | Superseded by git history |
| Renamed `get_intervals` to `_get_search_interval` (private, simplified) | Same as BounceSolFull |
| Improved all docstrings | Documentation |
| Added module docstring | Documentation |

---

### `src/RGE/FOPT_RGE.py`  *(spline-backed utilities)*

| Change | Reason |
|---|---|
| **BUG FIX**: `get_phi_min` for `is_HT=False` called `self.solver_full.phi_min(T, gD)` (2 args) | `BounceSolver.phi_min` requires `(T, gD, scale, ls0)` — would raise `TypeError` at runtime.  Fixed to `self.solver_full.phi_min(T, gD, mu, ls0)` |
| Removed `import src.RGE.VeffFunc_RGE as veff` | Unused |
| Removed `import src.RGE.BounceSolFull_RGE as bs_full` | Now correctly used as `import src.RGE.BounceSolFull_RGE as bs_full` within the file |
| Removed `import src.RGE.BounceSolHighT_RGE as bs_ht` | Same |
| Improved all docstrings | Documentation |
| Added module docstring with pipeline position | Documentation |

---

### `src/RGE/FOPT_RGE_real.py`  *(solver-backed utilities)*

| Change | Reason |
|---|---|
| **BUG FIX**: `alpha` method was at module scope (indentation 0) | Missing 4-space indent made it a module-level function instead of a class method — calls via `self.alpha(...)` would fail with `TypeError` |
| **BUG FIX**: `_SE_from_solver` for `is_HT=False` called `self.solver_full.SE(T, gD)` (2 args) | `BounceSolver.SE` requires `(T, gD, scale, ls0)` — `TypeError` at runtime.  Fixed to pass `mu` and `ls0` |
| **BUG FIX**: `get_phi_min` for `is_HT=False` called `self.solver_full.phi_min(T, gD)` (2 args) | Same as above |
| Removed Spanish-language docstrings and inline comments | All documentation is now in English |
| Removed typo `"""F` in constructor docstring | Minor |
| Removed informal comment "If you insist on the spline, too bad" | Replaced with a clear description of the finite-difference approach |
| Renamed `use_num_derivative` parameter in `beta` to `use_centered` | More descriptive; semantics unchanged |
| Removed `import src.RGE.VeffFunc_RGE as veff` | Unused |
| Added module docstring | Documentation |

---

### `src/RGE/SE_interpolator.py`

| Change | Reason |
|---|---|
| **BUG FIX**: `spline_ST` reloaded the `.npz` file from disk on every call | The data was already loaded in `__init__` and stored.  Now uses `self._raw_logS` instead — eliminates redundant I/O |
| Removed Spanish inline comment `# rellenar NaNs` | Documentation |
| Added `_GRID_FILENAME` class constant | Avoids repeating the filename string in two places |
| Improved all docstrings | Documentation |
| Added module docstring | Documentation |

---

### `src/RGE/GW_bubbles_LVV22.py`

| Change | Reason |
|---|---|
| **BUG FIX**: `spectrum_bubbles` ignored its `g_star` parameter and always used `cs.g_dof` via an inner `import constants as cs` | Moved the import to module level and honoured the `g_star` argument |
| Removed Spanish-language docstrings | Documentation |
| Renamed `_S_shape_f_over_beta` → `_S_shape` | Shorter; the argument name conveys the meaning |
| Moved `import constants as cs` from inside function to module level | Standard practice |
| Added module docstring | Documentation |
| Improved all docstrings (NumPy style) | Documentation |

---

### `src/RGE/GW_field_LV20.py`

| Change | Reason |
|---|---|
| **BUG FIX**: `lisa_noise_h2Omega` used `cs.Hhund` which didn't exist | Added `Hhund = H_HUND` alias in `constants.py` |
| **BUG FIX**: `h2Omega_thr` used `cs.SNRthr` which didn't exist | Added `SNRthr = SNR_THR` alias in `constants.py` |
| **BUG FIX**: `GWFieldLV20.__init__` expected `cs.LV20_PRESETS` which didn't exist | Added `LV20_PRESETS` to `constants.py` |
| Removed Spanish-language docstrings and comments | Documentation |
| Improved error message when `LV20_PRESETS` key is missing | Now catches both `AttributeError` and `KeyError` |
| Added module docstring | Documentation |
| Improved all docstrings (NumPy style) | Documentation |

---

### `src/RGE/GW_RGE_spectrum.py`

| Change | Reason |
|---|---|
| Removed unused `from math import pi` | `np.pi` is used consistently throughout |
| Removed unused `from scipy.integrate import quad` | Not called in this module |
| Added module docstring with references | Documentation |
| Improved all docstrings (NumPy style) | Documentation |

---

### `startup.py`

| Change | Reason |
|---|---|
| Removed Spanish-language comments | Documentation |
| Added imports for `GW_RGE_spectrum`, `GW_bubbles_LVV22`, `GW_field_LV20` | These modules existed but were not loaded in the interactive session |
| Replaced `sys.path.insert(0, str(project_root / "src"))` with `sys.path.insert(0, str(project_root))` as the primary entry + src fallback | Cleaner; avoids ambiguity between `import constants` and `import src.constants` |
| Replaced emoji in print statement | Neutral print, no emoji |
| Added module docstring | Documentation |

---

### `utils/plot_styles.py`

| Change | Reason |
|---|---|
| Removed large commented-out `apply_ordered_legend` block | Dead code; the live implementation below it is used |
| Extracted all rcParams into a single `matplotlib.rcParams.update(...)` call at module load | Avoids scattered mutations; idempotent |
| Added `_cmfont` as a private variable (underscored) | Signals it is an implementation detail |
| Added module docstring | Documentation |
| Fixed label `"HighT": r"High - T"` → `r"High-$T$"` | Formatting improvement (math mode) |

---

### `src/__init__.py`, `src/RGE/__init__.py`, `src/noRGE/__init__.py`

| Change | Reason |
|---|---|
| Created proper `__init__.py` files | The existing files were misspelled `__int__.py` (missing the `i`). Python does not recognise them as package markers, so IDE tooling and any code using `from src.RGE import ...` may have worked only due to `sys.path` injection. The correct files are now in place with descriptive module docstrings. |

---

## Modularity architecture

The pipeline is now structured so that studying a **different scalar model** requires changes in **at most two files**:

```
constants.py       ← initial conditions (gD0, lambdaS0, mS0)
                     and LV20_PRESETS if GW spectra are needed

src/RGE/VeffFunc_RGE.py  ← all model-specific potential formulas
  ├── Vtree, mPhi2, mSigma2, mAp2   (mass spectrum)
  ├── PiPhi, PiAp                    (Debye masses)
  └── Vcw, Vdaisy                    (loop corrections)

src/RGE/RGEsolver.py     ← beta functions for the new couplings
```

Everything else (bounce solvers, FOPT utilities, GW spectra) is
model-independent and requires no changes.

To extend:
1. Subclass `VeffRGE` and override the physics methods, **or**
2. Edit `VeffFunc_RGE.py` directly (it is a leaf in the dependency tree).

---

## Files not modified

| File | Reason |
|---|---|
| `src/noRGE/NoRGEVeff.py` | Legacy fixed-scale code; uses old import paths (`Miselanie.constants`). Kept for reference only — do not import in new code. |
| `src/noRGE/FOPT_params.py` | Same. Contains heavy commented-out blocks from early development. |
| `src/noRGE/BounceSolFull.py` | Same. |
| `src/noRGE/BounceSolHighT.py` | Same. |
| `src/noRGE/CW_spectrum.py` | Same. |
| Data files (`data/`) | Read-only. |
| Notebooks (`notebooks/`) | Execution state; no source changes were made. |
| `VT_integralNumeric.dat` | Numerical table used by `VeffRGE`. |
