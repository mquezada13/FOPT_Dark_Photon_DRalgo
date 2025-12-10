# dev.py - convenience imports for interactive work

#startup.py
import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


from scipy import optimize, integrate, interpolate, special
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, root_scalar
from scipy.integrate import solve_ivp

# Local project imports
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "utils"))


import constants as cs
from RGE import VeffFunc_RGE as veff
from RGE import BounceSolFull_RGE as bs_full
from RGE import BounceSolHighT_RGE as bs_ht
from RGE import FOPT_RGE as fopt
from RGE import RGEsolver as rge
from RGE import FOPT_RGE_real as fopt_real
from utils import plot_styles as ps

# =========================
# Matplotlib configuration
# =========================
plt.style.use("classic")  # estilo limpio y estable
plt.rcParams["figure.dpi"] = 100
plt.rcParams["axes.formatter.use_mathtext"] = True  # arregla warning cmr10
plt.rcParams["font.size"] = 12
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["axes.linewidth"] = 1.2

# aplicar estilo propio si existe
if hasattr(ps, "apply_standard_formatting"):
    def new_figure(*args, **kwargs):
        fig, ax = plt.subplots(*args, **kwargs)
        ps.apply_standard_formatting(ax)
        return fig, ax
    print("📊 Plot style: using ps.apply_standard_formatting")
else:
    def new_figure(*args, **kwargs):
        return plt.subplots(*args, **kwargs)

print("✅ startup.py loaded: numpy, scipy, matplotlib, and project modules are ready.")
print("   Use new_figure() to create pre-formatted plots.")
