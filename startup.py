# startup.py
"""
Interactive session bootstrap for the FOPT Dark Photon project.

Run this file at the start of a Jupyter notebook or IPython session:

    %run startup.py          # in a notebook
    exec(open("startup.py").read())   # in plain Python

After loading, the following objects are available in the global namespace:
  - numpy  (np), scipy sub-packages, matplotlib (plt)
  - cs     : physical constants module
  - veff   : VeffFunc_RGE module  (dark photon, no fermion)
  - bs_full / bs_ht : bounce-solver modules (compatible with both Veff classes)
  - fopt_real : phase-transition utility module (direct bounce solvers)
  - rge    : RGE solver module  (no fermion)
  - gw_spec / gw_bub / gw_lv20 : GW spectrum modules
  - ps     : plot style utilities
  - new_figure() : creates a pre-formatted (fig, ax) pair

Fermion-model extensions (U(1)_D + complex scalar + Dirac fermion):
  - veff_ferm : VeffFunc_ferm_RGE module  (VeffRGE_fermion class)
  - rge_ferm  : RGESolver_fermion module
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from tqdm import tqdm

from scipy import optimize, integrate, interpolate, special
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, root_scalar
from scipy.integrate import solve_ivp

# ------------------------------------------------------------------ #
# Path setup                                                           #
# ------------------------------------------------------------------ #
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))        # for constants.py
sys.path.insert(0, str(project_root / "src"))  # legacy fallback for bare imports

# ------------------------------------------------------------------ #
# Project imports                                                      #
# ------------------------------------------------------------------ #
import constants as cs

from src.RGE import VeffFunc_RGE   as veff
from src.RGE import BounceSolFull_RGE  as bs_full
from src.RGE import BounceSolHighT_RGE as bs_ht
from src.RGE import FOPT_RGE_real  as fopt_real
from src.RGE import RGEsolver      as rge
from src.RGE import GW_RGE_spectrum as gw_spec
from src.RGE import GW_bubbles_LVV22 as gw_bub
from src.RGE import GW_field_LV20  as gw_lv20

from src.RGE import VeffFunc_ferm_RGE  as veff_ferm
from src.RGE import RGESolver_fermion  as rge_ferm

from utils import plot_styles as ps

# ------------------------------------------------------------------ #
# Matplotlib configuration                                             #
# ------------------------------------------------------------------ #
plt.style.use("classic")
plt.rcParams.update({
    "figure.dpi":                   100,
    "font.size":                    12,
    "legend.fontsize":              10,
    "xtick.direction":              "in",
    "ytick.direction":              "in",
    "axes.linewidth":               1.2,
    "axes.formatter.use_mathtext":  True,
})

if hasattr(ps, "apply_standard_formatting"):
    def new_figure(*args, **kwargs):
        """Create a (fig, ax) pair with standard project formatting applied."""
        fig, ax = plt.subplots(*args, **kwargs)
        ps.apply_standard_formatting(ax)
        return fig, ax
else:
    def new_figure(*args, **kwargs):
        """Create a plain (fig, ax) pair."""
        return plt.subplots(*args, **kwargs)

print("startup.py loaded: numpy, scipy, matplotlib, and all project modules are ready.")
print("  Use new_figure() to create pre-formatted plots.")
