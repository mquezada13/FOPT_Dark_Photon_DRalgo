# ============================================================
# Phase transition utilities (class-based, using REAL SE from solvers)
# ============================================================

import numpy as np
import scipy.optimize as optimize
from scipy.integrate import quad
from functools import lru_cache

import constants as cs
import VeffFunc_RGE as veff
import BounceSolFull_RGE as bs_full
import BounceSolHighT_RGE as bs_ht


class FOPTUtilities:
    def __init__(self, veff_obj, assume_solver_returns_S3=True, smooth_unused=None):
        """
        Phase transition utilities using the actual bounce solvers (no spline).

        Parameters
        ----------F
        veff_obj : VeffRGE
            Preconstructed effective potential object with correct path to data.
        assume_solver_returns_S3 : bool, optional
            If True, solvers return S3(T). If a solver returns (S3/T),
            set this to False and the code will multiply by T internally.
        smooth_unused : any
            Kept only to preserve old constructor signature. Ignored.
        """
        # --- Constants ---
        self.G = cs.GF
        self.gstar = cs.g_dof

        # --- Effective potential (passed explicitly) ---
        self.veff_obj = veff_obj

        # --- Bounce solvers using this veff ---
        self.solver_ht = bs_ht.BounceSolverHighT(veff=self.veff_obj)
        self.solver_full = bs_full.BounceSolver(veff=self.veff_obj)

        # --- Convention about what solvers return ---
        self.assume_solver_returns_S3 = assume_solver_returns_S3

    # ================== Internal: SE from solvers ==================
    def _SE_from_solver(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0):
        if is_HT:
            mu_scale = scale or np.pi
            # Usa SIEMPRE SE(T, gD, scale, ls0) y extrae S3
            if hasattr(self.solver_ht, "SE"):
                out = self.solver_ht.SE(T, gD, mu_scale, ls0)
                Sval = out[0] if isinstance(out, (tuple, list)) else float(out)
            else:
                raise AttributeError("High-T solver: falta método SE(T,gD,scale,ls0).")
        else:
            # Full-T: acepta SE(T,gD) y podría devolver S3 o (S3, algo)
            if hasattr(self.solver_full, "SE"):
                out = self.solver_full.SE(T, gD)
                Sval = out[0] if isinstance(out, (tuple, list)) else float(out)
            elif hasattr(self.solver_full, "S3"):
                Sval = float(self.solver_full.S3(T, gD))
            else:
                raise AttributeError("Full-T solver: falta SE(T,gD) o S3(T,gD).")

        # Si algún solver entrega S3/T en lugar de S3
        if not self.assume_solver_returns_S3:
            Sval = Sval * T
        return float(Sval)


    # ================== Cached SE ==================
    @lru_cache(maxsize=2000)
    def log10SE_cached(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0):
        """Return log10(S3(T,gD)) from the *real* solver."""
        S3 = self._SE_from_solver(T, gD, is_HT=is_HT, scale=scale, ls0=ls0)
        return float(np.log10(S3))

    @lru_cache(maxsize=2000)
    def SE_cached(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0):
        """Return S3(T,gD) in linear scale, from the *real* solver."""
        return float(10 ** self.log10SE_cached(T, gD, is_HT=is_HT, scale=scale, ls0=ls0))

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
    def log10Gamma(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0):
        """
        log10 Γ(T) with Γ ≈ T^4 (S3/2πT)^(3/2) e^{-S3/T}.
        Uses S3 from the real solver.
        """
        S3 = self.SE_cached(T, gD, is_HT=is_HT, scale=scale, ls0=ls0)
        return (4*np.log10(T)
                + 1.5*np.log10(S3 / (2*np.pi*T))
                - (S3/T)/np.log(10))

    def Gamma(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0):
        return float(10 ** self.log10Gamma(T, gD, is_HT=is_HT, scale=scale, ls0=ls0))

    # ================== Next (Γ/H^4) ==================
    def log10Next(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0):
        phi_min = self.get_phi_min(T, gD, is_HT, scale, ls0)
        H = self.hubble(phi_min, T, gD, scale=scale, ls0=ls0)
        return self.log10Gamma(T, gD, is_HT=is_HT, scale=scale, ls0=ls0) - 4*np.log10(H)

    def Next(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0):
        return float(10 ** self.log10Next(T, gD, is_HT=is_HT, scale=scale, ls0=ls0))

    # ================== Nucleation temperature ==================
    def nucTemp(self, gD, is_HT=False, scale=None, ls0=cs.lambdaS0):
        def obj(T):
            return abs(self.log10Next(T, gD, is_HT=is_HT, scale=scale, ls0=ls0))
        res = optimize.minimize_scalar(obj, bounds=(1e-4, 1e-1), method="bounded")
        return res.x if res.success else np.nan

    # ================== Critical temperature ==================
    def critTemp(self, gD, is_HT=False, scale=None, ls0=cs.lambdaS0):
        def obj(T):
            phi_min = self.get_phi_min(T, gD, is_HT, scale, ls0)
            if is_HT:
                Veff_val = self.veff_obj.Veff_HighT(phi_min, T, gD, scale or np.pi, ls0)
            else:
                Veff_val = self.veff_obj.Veff(phi_min, T, gD, scale or np.pi, ls0)
            return abs(Veff_val)
        res = optimize.minimize_scalar(obj, bounds=(1e-4, 0.35), method="bounded")
        return res.x if res.success else np.nan

# ================== Alpha ==================
    def alpha(self, T, gD, is_HT=False, scale=None, ls0=cs.lambdaS0):
        rho_R = (np.pi**2 / 30.0) * self.gstar * T**4
        phi_min = self.get_phi_min(T, gD, is_HT, scale, ls0)
        if is_HT:
            veff_val = self.veff_obj.Veff0_RGE(phi_min, T, gD, scale or np.pi, ls0)
        else:
            veff_val = self.veff_obj.Veff(phi_min, T, gD, scale or np.pi, ls0)
        return float(-veff_val / rho_R)

    # ================== Beta ==================
    def beta(self, T_star, gD, is_HT=False, scale=None, ls0=cs.lambdaS0, use_num_derivative=False):
        """
        beta/H* ≈ T d(S3/T)/dT at T_star.
        If you insist on the spline, too bad; we do a robust numeric derivative instead.
        """
        if use_num_derivative:
            # Centered finite difference in log-safe way
            eps = 1e-4 * max(T_star, 1.0)
            T1, T2 = max(T_star - eps, 1e-8), T_star + eps
            S3_T1 = self._SE_from_solver(T1, gD, is_HT=is_HT, scale=scale, ls0=ls0) / T1
            S3_T2 = self._SE_from_solver(T2, gD, is_HT=is_HT, scale=scale, ls0=ls0) / T2
            dST_dT = (S3_T2 - S3_T1) / (T2 - T1)
            return float(T_star * dST_dT)
        else:
            # One-sided derivative if you're feeling reckless
            h = 1e-4 * max(T_star, 1.0)
            T2 = T_star + h
            S3T_star = self._SE_from_solver(T_star, gD, is_HT=is_HT, scale=scale, ls0=ls0) / T_star
            S3T_2 = self._SE_from_solver(T2, gD, is_HT=is_HT, scale=scale, ls0=ls0) / T2
            dST_dT = (S3T_2 - S3T_star) / (T2 - T_star)
            return float(T_star * dST_dT)