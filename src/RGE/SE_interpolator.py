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

import bisect
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

    def _interp_gD(self, T: float, gD: float) -> float:
        """Linearly interpolate log10(S_E) in gD between the two nearest grid values."""
        gD_grid = sorted(self.spline_dict.keys())
        key = float(gD)
        if key < gD_grid[0] or key > gD_grid[-1]:
            raise ValueError(
                f"gD={gD} is outside the grid range [{gD_grid[0]}, {gD_grid[-1]}]."
            )
        idx  = bisect.bisect_left(gD_grid, key)
        if idx == 0:
            return float(self.spline_dict[gD_grid[0]](T))
        g_lo = gD_grid[idx - 1]
        g_hi = gD_grid[idx]
        w    = (key - g_lo) / (g_hi - g_lo)
        return (1.0 - w) * float(self.spline_dict[g_lo](T)) + w * float(self.spline_dict[g_hi](T))

    def log10SE(self, T: float, gD: float) -> float:
        """
        Return log10(S_E(T, gD)) evaluated on the spline.

        If gD is not exactly on the grid, linearly interpolates between the
        two nearest tabulated gD values.

        Parameters
        ----------
        T  : temperature [GeV]
        gD : dark gauge coupling (within the tabulated range)
        """
        key = float(gD)
        if key in self.spline_dict:
            return float(self.spline_dict[key](T))
        return self._interp_gD(T, gD)

    def SE(self, T: float, gD: float) -> float:
        """Return S_E(T, gD) in linear scale."""
        return 10.0 ** self.log10SE(T, gD)

    def spline_ST(self, gD: float, smooth: float = 1e1) -> UnivariateSpline:
        """
        Return a UnivariateSpline of S_E / T (linear) for a given gD.

        If gD is not exactly on the grid, S_E/T is linearly interpolated in
        gD between the two nearest tabulated columns before fitting the spline.

        Parameters
        ----------
        gD     : dark gauge coupling (within the tabulated range)
        smooth : smoothing factor for this secondary spline (default 10).
        """
        gD_arr = self.gD_vals
        key    = float(gD)
        j_exact = np.where(np.isclose(gD_arr, key))[0]

        if len(j_exact) > 0:
            col  = self._raw_logS[:, j_exact[0]]
            mask = np.isfinite(col)
            ST   = 10.0 ** col[mask] / self.T_vals[mask]
            return UnivariateSpline(self.T_vals[mask], ST, s=smooth)

        # Interpolate between the two nearest columns
        idx  = bisect.bisect_left(list(gD_arr), key)
        idx  = min(max(idx, 1), len(gD_arr) - 1)
        g_lo, g_hi = gD_arr[idx - 1], gD_arr[idx]
        w    = (key - g_lo) / (g_hi - g_lo)

        col_lo = self._raw_logS[:, idx - 1]
        col_hi = self._raw_logS[:, idx]
        mask   = np.isfinite(col_lo) & np.isfinite(col_hi)
        col_ip = (1.0 - w) * col_lo[mask] + w * col_hi[mask]
        ST     = 10.0 ** col_ip / self.T_vals[mask]

        return UnivariateSpline(self.T_vals[mask], ST, s=smooth)
