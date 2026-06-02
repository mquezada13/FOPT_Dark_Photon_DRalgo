# FOPT_RGE.py
"""
Phase-transition utilities using a spline interpolator for log10(S_E).

This module provides the FOPTUtilities class that computes nucleation
temperatures, critical temperatures, GW parameters (alpha, beta/H),
and related quantities by reading S_E(T, gD) from a pre-computed grid
stored in data/clean_data/.

Use FOPT_RGE_real.py instead when you need to evaluate S_E directly from
the bounce solvers (no pre-computed grid required).

Pipeline position
-----------------
VeffRGE  -->  SEInterpolator (grid)  -->  FOPTUtilities  -->  notebooks / GW
"""

import numpy as np
import scipy.optimize as optimize
from scipy.integrate import quad
from functools import lru_cache

import constants as cs
from src.RGE.SE_interpolator import SEInterpolator
import src.RGE.BounceSolFull_RGE as bs_full
import src.RGE.BounceSolHighT_RGE as bs_ht


class FOPTUtilities:
    """
    Phase-transition utilities backed by a spline of log10(S_E).

    The Euclidean action S_E is read from a pre-computed (T, gD) grid via
    SEInterpolator, avoiding expensive on-the-fly bounce solves.
    All thermodynamic quantities (Tc, Tn, alpha, beta/H, Pf) are derived
    from this spline and the VeffRGE object that was used to generate the grid.

    Parameters
    ----------
    veff_obj : VeffRGE
        Pre-constructed effective potential instance.
    smooth : float, optional
        Smoothing parameter forwarded to the UnivariateSpline inside
        SEInterpolator (default 1e-2).
    """

    def __init__(self, veff_obj, smooth: float = 1e-2):
        self.G      = cs.GF
        self.gstar  = cs.g_dof
        self.interp = SEInterpolator(smooth=smooth)
        self.veff_obj    = veff_obj
        self.solver_ht   = bs_ht.BounceSolverHighT(veff=self.veff_obj)
        self.solver_full = bs_full.BounceSolver(veff=self.veff_obj)

    # ------------------------------------------------------------------ #
    # Cached Euclidean action from spline                                  #
    # ------------------------------------------------------------------ #

    @lru_cache(maxsize=2000)
    def log10SE_cached(self, T: float, gD: float) -> float:
        """Return log10(S_E(T, gD)) from the pre-computed spline."""
        return float(self.interp.log10SE(T, gD))

    @lru_cache(maxsize=2000)
    def SE_cached(self, T: float, gD: float) -> float:
        """Return S_E(T, gD) in linear scale from the pre-computed spline."""
        return 10.0 ** self.log10SE_cached(T, gD)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def get_phi_min(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0):
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

    def hubble(self, phi_min, T, gD, scale=None, ls0=cs.lambdaS0):
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

    def log10Gamma(self, T: float, gD: float) -> float:
        """
        log10(Gamma(T)) where Gamma ~ T^4 (S/2piT)^{3/2} exp(-S/T).

        S_E is read from the spline.
        """
        S = self.SE_cached(T, gD)
        return (
            4.0 * np.log10(T)
            + 1.5 * np.log10(S / (2.0 * np.pi * T))
            - (S / T) / np.log(10.0)
        )

    def Gamma(self, T: float, gD: float) -> float:
        """Nucleation rate Gamma(T) in linear scale."""
        return 10.0 ** self.log10Gamma(T, gD)

    # ------------------------------------------------------------------ #
    # Gamma / H^4                                                         #
    # ------------------------------------------------------------------ #

    def log10Next(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0) -> float:
        """log10(Gamma / H^4) — nucleation efficiency per Hubble volume."""
        phi_min = self.get_phi_min(T, gD, is_HT, scale, ls0)
        H       = self.hubble(phi_min, T, gD, scale, ls0)
        return self.log10Gamma(T, gD) - 4.0 * np.log10(H)

    def Next(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0) -> float:
        """Gamma / H^4 in linear scale."""
        return 10.0 ** self.log10Next(T, gD, is_HT, scale, ls0)

    # ------------------------------------------------------------------ #
    # Nucleation temperature                                               #
    # ------------------------------------------------------------------ #

    def nucTemp(self, gD, is_HT=False, scale=None, ls0=cs.lambdaS0) -> float:
        """
        Nucleation temperature T_n defined by Gamma/H^4 = 1.

        Solved by minimising |log10(Next)| over T in (1e-5, 0.35) GeV.
        Returns NaN if no solution is found.
        """
        result = optimize.minimize_scalar(
            lambda T: abs(self.log10Next(T, gD, is_HT, scale, ls0)),
            bounds=(1e-5, 0.35),
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
    # Percolation probability                                              #
    # ------------------------------------------------------------------ #

    def outer_integrand(self, T, TT, gD, is_HT=False, scale=None, ls0=cs.lambdaS0) -> float:
        """Integrand of the percolation exponent I(T) at a single outer temperature TT."""
        inner, _ = quad(
            lambda Tp: 1.0 / self.hubble(self.get_phi_min(Tp, gD, is_HT, scale, ls0), Tp, gD, scale, ls0),
            T, TT, limit=500, epsrel=1e-6, epsabs=1e-6,
        )
        gamma_val = self.Gamma(TT, gD)
        phi_TT    = self.get_phi_min(TT, gD, is_HT, scale, ls0)
        H_TT      = self.hubble(phi_TT, TT, gD, scale, ls0)
        return float((gamma_val / (TT**4 * H_TT)) * inner**3)

    def Pf(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0) -> float:
        """
        Probability that a point remains in the false vacuum at temperature T.

        P_f(T) = exp[-(4pi/3) integral_{T}^{T_c} dTT * outer_integrand(T, TT)]
        """
        Tc = self.critTemp(gD, is_HT, scale, ls0)
        integral, _ = quad(
            lambda TT: self.outer_integrand(T, TT, gD, is_HT, scale, ls0),
            T, Tc, limit=500, epsrel=1e-6, epsabs=1e-6,
        )
        return float(np.exp((-4.0 * np.pi / 3.0) * integral))

    def perTemp(self, gD, is_HT=False, scale=None, ls0=cs.lambdaS0) -> float:
        """
        Percolation temperature T_p defined by P_f(T_p) = 0.71.

        Returns NaN if no solution is found.
        """
        result = optimize.minimize_scalar(
            lambda T: abs(self.Pf(T, gD, is_HT, scale, ls0) - 0.71),
            bounds=(1e-4, 0.3),
            method="bounded",
        )
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

    def beta(self, T_star: float, gD: float) -> float:
        """
        Inverse transition duration beta/H* ~ T d(S/T)/dT evaluated at T_star.

        Uses the analytic derivative of the S_E / T spline.
        Returns NaN if the spline is not available for this gD value.
        """
        try:
            spl       = self.interp.spline_ST(gD)
            dST_dT    = spl.derivative()(T_star)
            return float(T_star * dST_dT)
        except Exception:
            return np.nan
