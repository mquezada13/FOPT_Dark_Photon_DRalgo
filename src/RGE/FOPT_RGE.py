# ============================================================
# Phase transition utilities (class-based, with SE spline in log10)
# ============================================================

import numpy as np
import scipy.optimize as optimize
from scipy.integrate import quad
from functools import lru_cache

import constants as cs
import VeffFunc_RGE as veff
import BounceSolFull_RGE as bs_full
import BounceSolHighT_RGE as bs_ht
from SE_interpolator import SEInterpolator


class FOPTUtilities:
    def __init__(self, veff_obj, smooth=1e-2):
        """
        Phase transition utilities with spline-based log10(SE) interpolation.

        Parameters
        ----------
        veff_obj : VeffRGE
            Preconstructed effective potential object with correct path to data.
        smooth : float, optional
            Smoothing parameter for the spline interpolator.
        """
        # --- Constants ---
        self.G = cs.GF
        self.gstar = cs.g_dof

        # --- Global interpolator (spline of log10(SE)) ---
        self.interp = SEInterpolator(smooth=smooth)

        # --- Effective potential (passed explicitly) ---
        self.veff_obj = veff_obj

        # --- Bounce solvers using this veff ---
        self.solver_ht = bs_ht.BounceSolverHighT(veff=self.veff_obj)
        self.solver_full = bs_full.BounceSolver(veff=self.veff_obj)

    # ================== Cached SE ==================
    @lru_cache(maxsize=2000)
    def log10SE_cached(self, T, gD):
        """Return log10(S_E(T,gD)) from spline."""
        return float(self.interp.log10SE(T, gD))

    @lru_cache(maxsize=2000)
    def SE_cached(self, T, gD):
        """Return S_E(T,gD) in linear scale."""
        return 10**self.log10SE_cached(T, gD)

    # ================== Helpers ==================
    def get_phi_min(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0):
        if is_HT:
            return float(self.solver_ht.phi_min_Veff0(T, gD, scale or np.pi, ls0))
        else:
            return float(self.solver_full.phi_min(T, gD))

    def hubble(self, phi_min, T, gD, scale=None, ls0=cs.lambdaS0):
        mu_scale = scale or np.pi
        rhoH = -self.veff_obj.Veff0_RGE(phi_min, T, gD, mu_scale, ls0) \
             + (np.pi**2 / 30) * self.gstar * T**4
        return float(np.sqrt((8 * np.pi * self.G / 3) * rhoH))

    # ================== Gamma ==================
    def log10Gamma(self, T, gD):
        """Return log10(Gamma)."""
        logS10 = self.log10SE_cached(T, gD)
        S = 10**logS10
        return (4*np.log10(T)
                + 1.5*np.log10(S / (2*np.pi*T))
                - (S/T)/np.log(10))

    def Gamma(self, T, gD):
        """Return Gamma in linear scale."""
        return 10**self.log10Gamma(T, gD)

    # ================== Next (Γ/H^4) ==================
    def log10Next(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0):
        phi_min = self.get_phi_min(T, gD, is_HT, scale, ls0)
        H = self.hubble(phi_min, T, gD)
        return self.log10Gamma(T, gD) - 4*np.log10(H)

    def Next(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0):
        return 10**self.log10Next(T, gD, is_HT, scale, ls0)

    # ================== Nucleation temperature ==================
    def nucTemp(self, gD, is_HT=False, scale=None, ls0=cs.lambdaS0):
        def func(T):
            return abs(self.log10Next(T, gD, is_HT, scale, ls0))
        result = optimize.minimize_scalar(func, bounds=(1e-5, 0.35), method="bounded")
        return result.x if result.success else np.nan

    # ================== Critical temperature ==================
    def critTemp(self, gD, is_HT=False, scale=None, ls0=cs.lambdaS0):
        def func(T):
            phi_min = self.get_phi_min(T, gD, is_HT, scale, ls0)
            if is_HT:
                Veff_val = self.veff_obj.Veff_HighT(phi_min, T, gD, scale or np.pi, ls0)
            else:
                Veff_val = self.veff_obj.Veff(phi_min, T, gD, scale or np.pi, ls0)
            return abs(Veff_val)

        result = optimize.minimize_scalar(func, bounds=(1e-4, 0.35), method="bounded")
        return result.x if result.success else np.nan

    # ================== Percolation probability ==================
    def outer_integrand(self, T, TT, gD, is_HT=False, scale=None, ls0=cs.lambdaS0):
        integ, _ = quad(lambda Tp: 1 / self.hubble(self.get_phi_min(Tp, gD, is_HT, scale, ls0), Tp, gD),
                        T, TT, limit=500, epsrel=1e-6, epsabs=1e-6)
        gamma_val = self.Gamma(TT, gD)
        phi_min = self.get_phi_min(TT, gD, is_HT, scale, ls0)
        H_TT = self.hubble(phi_min, TT, gD)
        return float((gamma_val / (TT**4 * H_TT)) * integ**3)

    def Pf(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0):
        Tc = self.critTemp(gD, is_HT, scale, ls0)
        integral, _ = quad(lambda TT: self.outer_integrand(T, TT, gD, is_HT, scale, ls0),
                           T, Tc, limit=500, epsrel=1e-6, epsabs=1e-6)
        return float(np.exp((-4*np.pi/3) * integral))

    def perTemp(self, gD, is_HT=False, scale=None, ls0=cs.lambdaS0):
        def func(T):
            return abs(self.Pf(T, gD, is_HT, scale, ls0) - 0.71)
        result = optimize.minimize_scalar(func, bounds=(1e-4, 0.3), method="bounded")
        return result.x if result.success else np.nan

    # ================== Alpha ==================
    def alpha(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0):
        rho_R = (np.pi**2 / 30) * self.gstar * T**4
        phi_min = self.get_phi_min(T, gD, is_HT, scale, ls0)
        if is_HT:
            veff_val = self.veff_obj.Veff0_RGE(phi_min, T, gD, scale or np.pi, ls0)
        else:
            veff_val = self.veff_obj.Veff(phi_min, T, gD, scale or np.pi, ls0)
        return float(-veff_val / rho_R)

    # ================== Beta ==================
    def beta(self, T_star, gD):
        try:
            spline_ST = self.interp.spline_ST(gD)
            dST_dT = spline_ST.derivative()(T_star)
            return float(T_star * dST_dT)
        except Exception:
            return np.nan

