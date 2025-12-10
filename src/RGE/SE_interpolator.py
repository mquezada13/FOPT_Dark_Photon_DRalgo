import os
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import generic_filter


class SEInterpolator:
    """
    Interface to evaluate log10(S_E(T,gD)) using splines.
    Only works for gD values tabulated in the input grid.
    """

    def __init__(self, smooth=1e-2):
        _here = os.path.dirname(__file__)
        base_dir = os.path.normpath(os.path.join(_here, "..", "..", "data", "clean_data"))
        path_ht = os.path.join(base_dir, "SE_RGE_log_grid_HT_piT_cleaned.npz")

        if not os.path.exists(path_ht):
            raise FileNotFoundError(f"Grid file not found: {path_ht}")

        d = np.load(path_ht)
        logS = d["S_log_grid"]   # ya en log10
        T    = d["T_vals"]
        gD   = d["gD_vals"]

        # rellenar NaNs
        if np.isnan(logS).any():
            logS = generic_filter(logS, function=np.nanmean, size=3, mode="nearest")

        self.T_vals  = T
        self.gD_vals = gD
        self.spline_dict = {}

        for j, g in enumerate(gD):
            col = logS[:, j]
            mask = np.isfinite(col)
            if np.count_nonzero(mask) < 4:
                continue
            spline = UnivariateSpline(T[mask], col[mask], s=smooth)  # spline de log10(S)
            self.spline_dict[float(g)] = spline

    def log10SE(self, T, gD):
        """Return log10(S_E(T,gD)) from spline."""
        if float(gD) not in self.spline_dict:
            raise ValueError(f"gD={gD} not in grid. Available: {self.gD_vals}")
        return self.spline_dict[float(gD)](T)

    def SE(self, T, gD):
        """Return S_E(T,gD) in linear scale."""
        return 10**self.log10SE(T, gD)
        
    def spline_ST(self, gD, smooth=1e1):
        """
        Return a spline of S/T (linear) for a given gD.
        This matches the behavior of the original snippet.
        """
        j = np.where(np.isclose(self.gD_vals, float(gD)))[0][0]
        # columna original
        data = np.load(os.path.join(
            os.path.dirname(__file__),
            "..", "..", "data", "clean_data", "SE_RGE_log_grid_HT_piT_cleaned.npz"
        ))
        col = data["S_log_grid"][:, j]
        T_vals = data["T_vals"]
        mask = np.isfinite(col)

        SE_clean = 10**col[mask]
        ST_clean = SE_clean / T_vals[mask]

        return UnivariateSpline(T_vals[mask], ST_clean, s=smooth)
