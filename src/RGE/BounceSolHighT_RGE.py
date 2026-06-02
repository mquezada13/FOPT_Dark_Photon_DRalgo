# BounceSolHighT_RGE.py
"""
Bounce-solution calculator using the **high-T expansion** effective potential.

Identical algorithmic structure to BounceSolFull_RGE but calls Veff_HighT
instead of Veff.  Use this when m/T << 1 holds throughout the scan,
or to compare against the full calculation.
"""

import numpy as np
import warnings
from scipy import optimize
from scipy.integrate import quad

import constants as cs
from src.RGE.VeffFunc_RGE import VeffRGE


class BounceSolverHighT:
    """
    Bounce-solution calculator for the high-T expanded effective potential.

    Accepts any VeffRGE instance (or subclass) so the potential can be swapped
    without modifying this solver.

    Parameters
    ----------
    veff : VeffRGE or None
        Pre-constructed effective potential object.  If None, a default
        VeffRGE() is created with the standard table path.

    Public Methods
    --------------
    phi_min_Veff0(T, gD, scale, ls0) -> float
    phi_max(T, gD, scale, ls0) -> float
    phi_root_HT(T, gD, scale, ls0, ...) -> float
    ConstructVt(phi0, T, gD, scale, ls0) -> (callable, (a1,a2,a3,a4))
    dVt(phi, phi0, a1, a2, a3, a4) -> float
    SE0(phi0, T, gD, scale, ls0) -> float
    SE(T, gD, scale, ls0) -> (S_E, phi0_opt)
    """

    def __init__(self, veff: VeffRGE = None):
        self.veff = veff if veff is not None else VeffRGE()

    # ------------------------------------------------------------------ #
    # Vacuum structure                                                     #
    # ------------------------------------------------------------------ #

    def phi_min_Veff0(self, T, gD, scale, ls0, bound_min=(0.0, 1e7)):
        """
        Locate the minimum of the zero-T RGE potential.

        Used as an upper bound for all high-T broken-phase searches.
        """
        return optimize.minimize_scalar(
            lambda S: self.veff.Veff0_RGE(S, T, gD, scale, ls0),
            bounds=bound_min,
            method="bounded",
        ).x

    def _get_search_interval(self, T, gD, scale, ls0):
        upper = self.phi_min_Veff0(T, gD, scale, ls0)
        return (1e-7, upper)

    def phi_max(self, T, gD, scale, ls0):
        """Location of the potential barrier in the high-T potential."""
        interval = self._get_search_interval(T, gD, scale, ls0)
        return optimize.minimize_scalar(
            lambda S: -np.real(self.veff.Veff_HighT(S, T, gD, scale, ls0)),
            bounds=interval,
            method="bounded",
        ).x

    def phi_root_HT(self, T, gD, scale, ls0, verbose=False):
        """
        First zero of V_eff_HighT beyond the barrier top.

        Uses a dense scan to bracket the sign change robustly even when
        the barrier is shallow or narrow.

        Parameters
        ----------
        verbose : bool
            If True, prints the bracketing interval for debugging.
        """
        phi_top = self.phi_max(T, gD, scale, ls0)
        phi_brk = self.phi_min_Veff0(T, gD, scale, ls0)
        Vscaled = lambda S: np.real(self.veff.Veff_HighT(S, T, gD, scale, ls0))

        S_arr = np.linspace(phi_top, phi_brk, 50_000)
        V_arr = np.array([Vscaled(s) for s in S_arr])

        changes = np.where(np.sign(V_arr[:-1]) != np.sign(V_arr[1:]))[0]
        if len(changes) == 0:
            raise ValueError("No root found between barrier top and broken minimum.")

        idx = changes[0]
        lo, hi = S_arr[idx], S_arr[idx + 1]

        if verbose:
            print(f"[phi_root_HT] bracket: [{lo:.4e}, {hi:.4e}]  "
                  f"V={Vscaled(lo):.4e} / {Vscaled(hi):.4e}")

        return optimize.root_scalar(Vscaled, bracket=(lo, hi), xtol=1e-14).root

    # ------------------------------------------------------------------ #
    # Tunnelling potential                                                 #
    # ------------------------------------------------------------------ #

    def ConstructVt(self, phi0, T, gD, scale, ls0):
        """
        Quartic tunnelling-potential ansatz V_t(phi) matched at phi0 and phi_max.

        Returns
        -------
        vt : callable     V_t(phi)
        coeffs : tuple    (a1, a2, a3, a4) — passed to dVt for efficiency.
        """
        phi_top = self.phi_max(T, gD, scale, ls0)
        V0      = self.veff.Veff_HighT(phi0, T, gD, scale, ls0)
        eps     = 1e-9

        def dVeff(s):
            return (self.veff.Veff_HighT(s + eps, T, gD, scale, ls0)
                    - self.veff.Veff_HighT(s - eps, T, gD, scale, ls0)) / (2.0 * eps)

        dV0 = dVeff(phi0)
        VT  = self.veff.Veff_HighT(phi_top, T, gD, scale, ls0)

        d  = 3
        a1 = V0 / phi0
        a2 = ((d - 1) * phi0 * dV0 - d * V0) / (d * phi0**2)
        a3 = ((d - 1) * phi0 * dV0 - 2 * d * V0) / (d * phi0**3)

        dVt3T  = a1 + a2 * (2 * phi_top - phi0) + a3 * (3 * phi_top - phi0) * (phi_top - phi0)
        d2Vt3T = 2 * a2 + 2 * a3 * (3 * phi_top - 2 * phi0)

        phi0T = phi0 - phi_top
        c     = 4 * phi_top**2 * phi0T**2 * phi0**2

        Vt3T = (
            V0 * phi_top / phi0
            + (2 * phi0 * dV0 - 3 * V0) / (3 * phi0**2) * phi_top * (phi_top - phi0)
            + (2 * phi0 * dV0 - 6 * V0) / (3 * phi0**3) * phi_top * (phi_top - phi0)**2
        )

        a0T  = (
            -4 * (VT - Vt3T) * (phi0**2 - 6 * phi_top * phi0T)
            - 6 * phi_top * (phi0T - phi_top) * phi0T * dVt3T
            + 2 * phi_top**2 * phi0T**2 * d2Vt3T
        )
        Ut3T = 3 * dVt3T**2 + 4 * (VT - Vt3T) * d2Vt3T
        disc = a0T**2 - c * Ut3T
        a4   = (a0T - np.sqrt(disc)) / c if disc > 0 else 0.0

        def vt(phi):
            return (
                a1 * phi
                + a2 * phi * (phi - phi0)
                + a3 * phi * (phi - phi0)**2
                + a4 * phi**2 * (phi - phi0)**2
            )

        return vt, (a1, a2, a3, a4)

    def dVt(self, phi, phi0, a1, a2, a3, a4):
        """Analytic derivative of V_t with respect to phi."""
        return (
            a1
            + a2 * (2 * phi - phi0)
            + a3 * (3 * phi**2 - 4 * phi * phi0 + phi0**2)
            + a4 * (4 * phi**3 - 6 * phi**2 * phi0 + 2 * phi * phi0**2)
        )

    # ------------------------------------------------------------------ #
    # Euclidean action                                                     #
    # ------------------------------------------------------------------ #

    def SE0(self, phi0, T, gD, scale, ls0):
        """
        Euclidean action S_E for a fixed tunnelling endpoint phi0,
        evaluated with the high-T potential.
        """
        vt, (a1, a2, a3, a4) = self.ConstructVt(phi0, T, gD, scale, ls0)

        def integrand(S):
            vt_val   = vt(S)
            veff_val = self.veff.Veff_HighT(S, T, gD, scale, ls0)
            dvt_val  = self.dVt(S, phi0, a1, a2, a3, a4)

            dvt_val = max(np.abs(dvt_val), 1e-65)
            diff    = max(veff_val - vt_val, 0.0)
            return diff**1.5 / dvt_val**2

        result = quad(integrand, 0, phi0, limit=5_000, epsrel=1e-6, epsabs=1e-6)[0]
        return (32.0 * np.pi * np.sqrt(2.0) / 3.0) * result

    def SE(self, T, gD, scale, ls0):
        """
        Minimised Euclidean action S_E(T) and the optimal tunnelling endpoint.

        Returns
        -------
        (S_E_min, phi0_opt) : tuple of floats
        """
        phi_lo = self.phi_root_HT(T, gD, scale, ls0)
        phi_hi = self.phi_min_Veff0(T, gD, scale, ls0)
        result = optimize.minimize_scalar(
            lambda S: self.SE0(S, T, gD, scale, ls0),
            bounds=(phi_lo, phi_hi),
            method="bounded",
        )
        return result.fun, result.x
