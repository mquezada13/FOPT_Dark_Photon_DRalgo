# SE_interpolator.py
"""
Spline interpolator for the pre-computed Euclidean action grid.

Loads log10(S_E(T, gD)) from a cleaned .npz file and provides smooth
UnivariateSpline objects for each tabulated gD value.  Also exposes
a spline of S_E/T (used by FOPT_RGE.FOPTUtilities.beta).

The grid file is expected at:
    <project_root>/data/clean_data/SE_RGE_log_grid_HT_piT_cleaned.npz

with arrays:
    S_log_grid : shape (n_T, n_gD),  log10(S_E) values
    T_vals     : shape (n_T,)
    gD_vals    : shape (n_gD,)
"""

import os
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import generic_filter


class SEInterpolator:
    """
    Spline interface for log10(S_E(T, gD)) from a pre-computed grid.

    Only gD values present in the input grid are supported; querying
    an unlisted gD raises ValueError.

    Parameters
    ----------
    smooth : float, optional
        Smoothing factor passed to UnivariateSpline (default 1e-2).
        Increase if the spline overshoots; decrease for tighter fits.
    """

    _GRID_FILENAME = "SE_RGE_log_grid_HT_piT_cleaned.npz"

    def __init__(self, smooth: float = 1e-2):
        here     = os.path.dirname(__file__)
        grid_dir = os.path.normpath(os.path.join(here, "..", "..", "data", "clean_data"))
        path     = os.path.join(grid_dir, self._GRID_FILENAME)

        if not os.path.exists(path):
            raise FileNotFoundError(f"S_E grid not found: {path}")

        data  = np.load(path)
        logS  = data["S_log_grid"]   # already in log10
        T     = data["T_vals"]
        gD    = data["gD_vals"]

        # Fill isolated NaN cells with local mean to avoid spline artefacts.
        if np.isnan(logS).any():
            logS = generic_filter(logS, function=np.nanmean, size=3, mode="nearest")

        self.T_vals     = T
        self.gD_vals    = gD
        self._raw_logS  = logS   # kept so spline_ST can re-use it
        self.spline_dict: dict = {}

        for j, g in enumerate(gD):
            col  = logS[:, j]
            mask = np.isfinite(col)
            if np.count_nonzero(mask) < 4:
                continue
            self.spline_dict[float(g)] = UnivariateSpline(T[mask], col[mask], s=smooth)

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def log10SE(self, T: float, gD: float) -> float:
        """
        Return log10(S_E(T, gD)) evaluated on the spline.

        Parameters
        ----------
        T  : temperature [GeV]
        gD : dark gauge coupling (must be one of the tabulated values)
        """
        key = float(gD)
        if key not in self.spline_dict:
            raise ValueError(
                f"gD={gD} is not in the grid.  Available values: {list(self.spline_dict.keys())}"
            )
        return float(self.spline_dict[key](T))

    def SE(self, T: float, gD: float) -> float:
        """Return S_E(T, gD) in linear scale."""
        return 10.0 ** self.log10SE(T, gD)

    def spline_ST(self, gD: float, smooth: float = 1e1) -> UnivariateSpline:
        """
        Return a UnivariateSpline of S_E / T (linear) for a given gD.

        This spline is used by FOPT_RGE.FOPTUtilities.beta to compute
        d(S/T)/dT analytically via spline differentiation.

        Parameters
        ----------
        gD     : dark gauge coupling (must be one of the tabulated values)
        smooth : smoothing factor for this secondary spline (default 10,
                 intentionally larger than for the log spline).
        """
        j = np.where(np.isclose(self.gD_vals, float(gD)))[0]
        if len(j) == 0:
            raise ValueError(f"gD={gD} not found in grid.")
        j = j[0]

        col   = self._raw_logS[:, j]
        mask  = np.isfinite(col)
        SE_ln = 10.0 ** col[mask]
        ST    = SE_ln / self.T_vals[mask]

        return UnivariateSpline(self.T_vals[mask], ST, s=smooth)
