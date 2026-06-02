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
                 smooth_unused=None):
        self.G      = cs.GF
        self.gstar  = cs.g_dof
        self.veff_obj    = veff_obj
        self.solver_ht   = bs_ht.BounceSolverHighT(veff=self.veff_obj)
        self.solver_full = bs_full.BounceSolver(veff=self.veff_obj)
        self.assume_solver_returns_S3 = assume_solver_returns_S3

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

        Parameters
        ----------
        is_HT : bool
            If True use the high-T solver; otherwise use the full solver.
        scale : float or None
            RGE scale factor (mu = scale * T).  Defaults to pi.
        ls0   : float
            Scalar quartic at mu0.
        """
        mu = scale or np.pi
        if is_HT:
            return float(self.solver_ht.phi_min_Veff0(T, gD, mu, ls0))
        else:
            return float(self.solver_full.phi_min(T, gD, mu, ls0))

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

    def nucTemp(self, gD, is_HT=False, scale=None, ls0=cs.lambdaS0) -> float:
        """
        Nucleation temperature T_n defined by Gamma/H^4 = 1.

        Solved by minimising |log10(Next)| over T in (1e-4, 1e-1) GeV.
        Returns NaN if no solution is found.
        """
        result = optimize.minimize_scalar(
            lambda T: abs(self.log10Next(T, gD, is_HT=is_HT, scale=scale, ls0=ls0)),
            bounds=(1e-4, 1e-1),
            method="bounded",
        )
        return result.x if result.success else np.nan

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
        h  = 1e-4 * max(T_star, 1.0)

        if use_centered:
            T1    = max(T_star - h, 1e-8)
            T2    = T_star + h
            ST1   = self._SE_from_solver(T1, gD, is_HT=is_HT, scale=scale, ls0=ls0) / T1
            ST2   = self._SE_from_solver(T2, gD, is_HT=is_HT, scale=scale, ls0=ls0) / T2
            dST   = (ST2 - ST1) / (T2 - T1)
        else:
            T2    = T_star + h
            ST0   = self._SE_from_solver(T_star, gD, is_HT=is_HT, scale=scale, ls0=ls0) / T_star
            ST2   = self._SE_from_solver(T2, gD, is_HT=is_HT, scale=scale, ls0=ls0) / T2
            dST   = (ST2 - ST0) / (T2 - T_star)

        return float(T_star * dST)
