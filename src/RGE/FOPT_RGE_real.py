# FOPT_RGE_real.py
"""
Phase-transition utilities using the real bounce solvers (no pre-computed grid).

This module provides the FOPTUtilities class that computes nucleation
temperatures, critical temperatures, GW parameters (alpha, beta/H),
and related quantities by calling the bounce solvers directly at each
(T, gD) point.

Use FOPT_RGE.py instead when a pre-computed S_E grid is available; it is
orders of magnitude faster for scans.

Pipeline position
-----------------
VeffRGE  -->  BounceSolFull / BounceSolHighT  -->  FOPTUtilities  -->  notebooks / GW
"""

import numpy as np
import scipy.optimize as optimize
from scipy.integrate import quad
from functools import lru_cache

import constants as cs
import src.RGE.BounceSolFull_RGE as bs_full
import src.RGE.BounceSolHighT_RGE as bs_ht


class FOPTUtilities:
    """
    Phase-transition utilities backed by direct bounce-solver calls.

    Euclidean actions are computed on demand by BounceSolver / BounceSolverHighT.
    Results are cached with lru_cache to avoid redundant solves within a single
    Python session.

    Parameters
    ----------
    veff_obj : VeffRGE
        Pre-constructed effective potential instance.
    assume_solver_returns_S3 : bool, optional
        If True (default), the solvers return S3(T).  Set to False if your
        custom solver returns S3/T instead; the code multiplies by T internally.
    smooth_unused : any
        Accepted for API compatibility with FOPT_RGE.FOPTUtilities; ignored.
    """

    def __init__(self, veff_obj, assume_solver_returns_S3: bool = True,
                 smooth_unused=None, solver_ht=None, solver_full=None):
        self.G      = cs.GF
        self.gstar  = cs.g_dof
        self.veff_obj    = veff_obj
        self.solver_ht   = solver_ht   if solver_ht   is not None else bs_ht.BounceSolverHighT(veff=self.veff_obj)
        self.solver_full = solver_full if solver_full is not None else bs_full.BounceSolver(veff=self.veff_obj)
        self.assume_solver_returns_S3 = assume_solver_returns_S3
        self._phi_min_cache: dict = {}

    # ------------------------------------------------------------------ #
    # Internal: S_E from solvers                                           #
    # ------------------------------------------------------------------ #

    def _SE_from_solver(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0) -> float:
        """
        Call the appropriate bounce solver and return S3(T).

        The solver output may be a scalar or a (S3, phi0_opt) tuple;
        only the first element is used.
        """
        mu = scale or np.pi

        if is_HT:
            if not hasattr(self.solver_ht, "SE"):
                raise AttributeError("High-T solver is missing the SE(T, gD, scale, ls0) method.")
            out   = self.solver_ht.SE(T, gD, mu, ls0)
            S_val = out[0] if isinstance(out, (tuple, list)) else float(out)
        else:
            if hasattr(self.solver_full, "SE"):
                out   = self.solver_full.SE(T, gD, mu, ls0)
                S_val = out[0] if isinstance(out, (tuple, list)) else float(out)
            elif hasattr(self.solver_full, "S3"):
                S_val = float(self.solver_full.S3(T, gD, mu, ls0))
            else:
                raise AttributeError(
                    "Full-T solver is missing SE(T, gD, scale, ls0) or S3(T, gD, scale, ls0)."
                )

        if not self.assume_solver_returns_S3:
            S_val = S_val * T
        return float(S_val)

    # ------------------------------------------------------------------ #
    # Cached Euclidean action                                              #
    # ------------------------------------------------------------------ #

    @lru_cache(maxsize=2000)
    def log10SE_cached(self, T: float, gD: float, is_HT: bool = False,
                       scale=None, ls0: float = cs.lambdaS0) -> float:
        """Return log10(S3(T, gD)) from the real bounce solver (cached)."""
        S3 = self._SE_from_solver(T, gD, is_HT=is_HT, scale=scale, ls0=ls0)
        return float(np.log10(S3))

    @lru_cache(maxsize=2000)
    def SE_cached(self, T: float, gD: float, is_HT: bool = False,
                  scale=None, ls0: float = cs.lambdaS0) -> float:
        """Return S3(T, gD) in linear scale from the real bounce solver (cached)."""
        return float(10.0 ** self.log10SE_cached(T, gD, is_HT=is_HT, scale=scale, ls0=ls0))

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def get_phi_min(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0) -> float:
        """
        Return the broken-phase field minimum at temperature T.

        Results are cached so repeated calls within a session (e.g. nucTemp
        followed by alpha at the same T_n) hit the cache instead of
        re-running the minimiser.

        Parameters
        ----------
        is_HT : bool
            If True use the high-T solver; otherwise use the full solver.
        scale : float or None
            RGE scale factor (mu = scale * T).  Defaults to pi.
        ls0   : float
            Scalar quartic at mu0.
        """
        key = (float(T), float(gD), bool(is_HT), float(scale or np.pi), float(ls0))
        if key not in self._phi_min_cache:
            mu = scale or np.pi
            if is_HT:
                self._phi_min_cache[key] = float(self.solver_ht.phi_min_Veff0(T, gD, mu, ls0))
            else:
                self._phi_min_cache[key] = float(self.solver_full.phi_min(T, gD, mu, ls0))
        return self._phi_min_cache[key]

    def hubble(self, phi_min, T, gD, scale=None, ls0=cs.lambdaS0) -> float:
        """
        Hubble parameter H(T) assuming radiation + vacuum energy.

        rho_H = -V_eff^0_RGE(phi_min) + (pi^2/30) g_* T^4
        """
        mu = scale or np.pi
        rhoH = (
            -self.veff_obj.Veff0_RGE(phi_min, T, gD, mu, ls0)
            + (np.pi**2 / 30.0) * self.gstar * T**4
        )
        return float(np.sqrt((8.0 * np.pi * self.G / 3.0) * rhoH))

    # ------------------------------------------------------------------ #
    # Nucleation rate                                                      #
    # ------------------------------------------------------------------ #

    def log10Gamma(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0) -> float:
        """
        log10(Gamma(T)) where Gamma ~ T^4 (S/2piT)^{3/2} exp(-S/T).

        S3 is computed from the real bounce solver.
        """
        S3 = self.SE_cached(T, gD, is_HT=is_HT, scale=scale, ls0=ls0)
        return (
            4.0 * np.log10(T)
            + 1.5 * np.log10(S3 / (2.0 * np.pi * T))
            - (S3 / T) / np.log(10.0)
        )

    def Gamma(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0) -> float:
        """Nucleation rate Gamma(T) in linear scale."""
        return float(10.0 ** self.log10Gamma(T, gD, is_HT=is_HT, scale=scale, ls0=ls0))

    # ------------------------------------------------------------------ #
    # Gamma / H^4                                                         #
    # ------------------------------------------------------------------ #

    def log10Next(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0) -> float:
        """log10(Gamma / H^4) — nucleation efficiency per Hubble volume."""
        phi_min = self.get_phi_min(T, gD, is_HT, scale, ls0)
        H       = self.hubble(phi_min, T, gD, scale=scale, ls0=ls0)
        return self.log10Gamma(T, gD, is_HT=is_HT, scale=scale, ls0=ls0) - 4.0 * np.log10(H)

    def Next(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0) -> float:
        """Gamma / H^4 in linear scale."""
        return float(10.0 ** self.log10Next(T, gD, is_HT=is_HT, scale=scale, ls0=ls0))

    # ------------------------------------------------------------------ #
    # Nucleation temperature                                               #
    # ------------------------------------------------------------------ #

    def nucTemp(self, gD, is_HT=False, scale=None, ls0=cs.lambdaS0,
                tol_log10: float = 0.5) -> float:
        """
        Nucleation temperature T_n defined by Gamma/H^4 = 1.

        Coarse log-spaced scan over (1e-4, 0.35) GeV to bracket the root, then
        brentq for precise root finding.  Falls back to minimize_scalar if no
        sign change is found.  Returns NaN if no reliable solution exists.

        tol_log10 : max accepted |log10(Gamma/H^4)| at the fallback solution.
        """
        T_lo, T_hi = 1e-4, 0.35

        def f(T):
            return self.log10Next(T, gD, is_HT=is_HT, scale=scale, ls0=ls0)

        # Coarse scan from HIGH to LOW T (cooling direction).
        # f = log10(Gamma/H^4) starts negative at high T (large S3, slow nucleation)
        # and crosses zero at T_n.  Scanning high→low finds the FIRST crossing
        # during cooling, which is the physical nucleation temperature.
        T_scan = np.geomspace(T_hi, T_lo, 50)
        f_scan = np.full(len(T_scan), np.nan)
        for i, T in enumerate(T_scan):
            try:
                f_scan[i] = f(T)
            except Exception:
                pass

        valid = np.isfinite(f_scan)
        for i in range(len(T_scan) - 1):
            if valid[i] and valid[i + 1] and f_scan[i] * f_scan[i + 1] < 0:
                try:
                    T_a = min(T_scan[i], T_scan[i + 1])
                    T_b = max(T_scan[i], T_scan[i + 1])
                    root = optimize.brentq(f, T_a, T_b, xtol=1e-8, rtol=1e-7)
                    return float(root)
                except Exception:
                    pass

        # Fallback: minimise |f|; reject boundary hits and large residuals
        result = optimize.minimize_scalar(
            lambda T: abs(f(T)), bounds=(T_lo, T_hi), method="bounded"
        )
        if not result.success:
            return np.nan
        if abs(result.x - T_lo) < 1e-6 * T_lo or abs(result.x - T_hi) < 1e-4 * T_hi:
            return np.nan
        if abs(f(result.x)) > tol_log10:
            return np.nan
        return float(result.x)

    # ------------------------------------------------------------------ #
    # Critical temperature                                                 #
    # ------------------------------------------------------------------ #

    def critTemp(self, gD, is_HT=False, scale=None, ls0=cs.lambdaS0) -> float:
        """
        Critical temperature T_c defined by V_eff(phi_min, T_c) = 0.

        Solved by minimising |V_eff(phi_min)| over T in (1e-4, 0.35) GeV.
        Returns NaN if no solution is found.
        """
        def obj(T):
            phi_min = self.get_phi_min(T, gD, is_HT, scale, ls0)
            if is_HT:
                return abs(self.veff_obj.Veff_HighT(phi_min, T, gD, scale or np.pi, ls0))
            return abs(self.veff_obj.Veff(phi_min, T, gD, scale or np.pi, ls0))

        result = optimize.minimize_scalar(obj, bounds=(1e-4, 0.35), method="bounded")
        return result.x if result.success else np.nan

    # ------------------------------------------------------------------ #
    # GW parameters                                                        #
    # ------------------------------------------------------------------ #

    def alpha(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0) -> float:
        """
        Transition strength parameter alpha = -V_eff(phi_min) / rho_R.

        rho_R = (pi^2/30) g_* T^4.
        """
        rho_R   = (np.pi**2 / 30.0) * self.gstar * T**4
        phi_min = self.get_phi_min(T, gD, is_HT, scale, ls0)
        if is_HT:
            veff_val = self.veff_obj.Veff0_RGE(phi_min, T, gD, scale or np.pi, ls0)
        else:
            veff_val = self.veff_obj.Veff(phi_min, T, gD, scale or np.pi, ls0)
        return float(-veff_val / rho_R)

    def beta(self, T_star: float, gD: float, is_HT: bool = False,
             scale=None, ls0: float = cs.lambdaS0,
             use_centered: bool = True) -> float:
        """
        Inverse transition duration beta/H* ~ T d(S3/T)/dT at T_star.

        A numerical finite-difference derivative is used.  The step size
        scales with T_star to remain well-conditioned across the parameter space.

        Parameters
        ----------
        use_centered : bool
            If True (default), use a centred finite difference (order 2).
            If False, use a forward difference (order 1).
        """
        # Evaluate at three relative step sizes and take the median.
        # This is robust against both (a) steps too large relative to T_star
        # (which caused the old downward spikes) and (b) steps too small that
        # amplify solver noise (which caused the upward spike at ~0.65).
        estimates = []
        for frac in (0.005, 0.02, 0.08):
            h = frac * T_star
            try:
                if use_centered:
                    T1  = T_star - h
                    T2  = T_star + h
                    ST1 = self.SE_cached(T1, gD, is_HT=is_HT, scale=scale, ls0=ls0) / T1
                    ST2 = self.SE_cached(T2, gD, is_HT=is_HT, scale=scale, ls0=ls0) / T2
                    estimates.append(T_star * (ST2 - ST1) / (T2 - T1))
                else:
                    T2  = T_star + h
                    ST0 = self.SE_cached(T_star, gD, is_HT=is_HT, scale=scale, ls0=ls0) / T_star
                    ST2 = self.SE_cached(T2,     gD, is_HT=is_HT, scale=scale, ls0=ls0) / T2
                    estimates.append(T_star * (ST2 - ST0) / (T2 - T_star))
            except Exception:
                pass

        if not estimates:
            return np.nan
        return float(np.median(estimates))
