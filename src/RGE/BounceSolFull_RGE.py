# BounceSolFull_RGE.py
"""
Bounce-solution calculator using the **full** (exact J_B integral) effective potential.

The Euclidean action S_E(T, gD, scale, ls0) is minimised over the thin-wall
ansatz phi_0, bracketed between the first zero of V_eff beyond the barrier
and the symmetry-broken minimum.

The quartic tunnelling-potential ansatz follows the approach of Espinosa (1996).
"""

import numpy as np
import warnings
from scipy import optimize
from scipy.integrate import quad

import constants as cs
from src.RGE.VeffFunc_RGE import VeffRGE


class BounceSolver:
    """
    Bounce-solution calculator for the full finite-temperature effective potential.

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
        Minimum of the zero-T RGE potential (used as an upper bound).
    phi_min(T, gD, scale, ls0) -> float
        Symmetry-broken minimum of the full finite-T potential.
    phi_max(T, gD, scale, ls0) -> float
        Location of the potential barrier maximum.
    phi_root(T, gD, scale, ls0, ...) -> float
        First zero of V_eff beyond the barrier.
    ConstructVt(phi0, T, gD, scale, ls0) -> callable
        Quartic tunnelling-potential matching V and dV at phi0 and phi_max.
    SE0(phi0, T, gD, scale, ls0) -> float
        Euclidean action for a fixed phi0.
    SE(T, gD, scale, ls0) -> (S_E, phi0_opt)
        Minimised Euclidean action and optimal phi0.
    """

    def __init__(self, veff: VeffRGE = None):
        self.veff = veff if veff is not None else VeffRGE()

    # ------------------------------------------------------------------ #
    # Vacuum structure                                                     #
    # ------------------------------------------------------------------ #

    def phi_min_Veff0(self, T, gD, scale, ls0, bound_min=(1e-3, 1e7)):
        """
        Locate the minimum of the zero-T RGE potential.

        Used as an upper bound when scanning the broken phase.
        """
        return optimize.minimize_scalar(
            lambda S: self.veff.Veff0_RGE(S, T, gD, scale, ls0),
            bounds=bound_min,
            method="bounded",
        ).x

    def _get_search_interval(self, T, gD, scale, ls0):
        """Return the (lower, upper) field interval for the broken-phase search."""
        upper = self.phi_min_Veff0(T, gD, scale, ls0)
        return (1e-7, upper)

    def phi_min(self, T, gD, scale, ls0):
        """Symmetry-broken minimum of the full finite-T potential."""
        interval = self._get_search_interval(T, gD, scale, ls0)
        return optimize.minimize_scalar(
            lambda S: self.veff.Veff(S, T, gD, scale, ls0),
            bounds=interval,
            method="bounded",
        ).x

    def phi_max(self, T, gD, scale, ls0):
        """Location of the potential barrier (maximum of -V_eff)."""
        interval = self._get_search_interval(T, gD, scale, ls0)
        return optimize.minimize_scalar(
            lambda S: -np.real(self.veff.Veff(S, T, gD, scale, ls0)),
            bounds=interval,
            method="bounded",
        ).x

    def phi_root(self, T, gD, scale, ls0, scale_factor=1.0, verbose=False):
        """
        First zero of V_eff beyond the barrier top.

        A dense scan between phi_max and phi_min is used to bracket the sign
        change robustly even when the barrier is shallow or narrow.

        Parameters
        ----------
        scale_factor : float
            Multiplicative rescaling applied to V_eff before root finding.
            Useful for numerical conditioning; does not affect the root location.
        verbose : bool
            If True, prints the bracketing interval for debugging.
        """
        phi_top = self.phi_max(T, gD, scale, ls0)
        phi_brk = self.phi_min(T, gD, scale, ls0)
        Vscaled = lambda S: np.real(scale_factor * self.veff.Veff(S, T, gD, scale, ls0))

        S_arr = np.linspace(phi_top, phi_brk, 50_000)
        V_arr = np.array([Vscaled(s) for s in S_arr])

        changes = np.where(np.sign(V_arr[:-1]) != np.sign(V_arr[1:]))[0]
        if len(changes) == 0:
            raise ValueError("No root found between barrier top and broken minimum.")

        idx = changes[0]
        lo, hi = S_arr[idx], S_arr[idx + 1]

        if verbose:
            print(f"[phi_root] bracket: [{lo:.4e}, {hi:.4e}]  "
                  f"V={Vscaled(lo):.4e} / {Vscaled(hi):.4e}")

        return optimize.root_scalar(Vscaled, bracket=(lo, hi), xtol=1e-14).root

    # ------------------------------------------------------------------ #
    # Tunnelling potential                                                 #
    # ------------------------------------------------------------------ #

    def ConstructVt(self, phi0, T, gD, scale, ls0):
        """
        Quartic tunnelling-potential ansatz V_t(phi) matched at phi0 and phi_max.

        Constructs a degree-4 polynomial that satisfies:
          - V_t(phi0) = V_eff(phi0)
          - V_t'(phi0) = V_eff'(phi0)
          - V_t(phi_max) = V_eff(phi_max)

        Returns
        -------
        callable  V_t(phi)
        """
        phi_top = self.phi_max(T, gD, scale, ls0)
        V0      = self.veff.Veff(phi0, T, gD, scale, ls0)
        eps     = 1e-9

        def dVeff(s):
            return (self.veff.Veff(s + eps, T, gD, scale, ls0)
                    - self.veff.Veff(s - eps, T, gD, scale, ls0)) / (2.0 * eps)

        dV0 = dVeff(phi0)
        VT  = self.veff.Veff(phi_top, T, gD, scale, ls0)

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

        return vt

    # ------------------------------------------------------------------ #
    # Euclidean action                                                     #
    # ------------------------------------------------------------------ #

    def SE0(self, phi0, T, gD, scale, ls0, debug=False):
        """
        Euclidean action S_E for a fixed tunnelling endpoint phi0.

        Computed via:
          S_E = (32 pi sqrt(2) / 3) * int_0^{phi0} (V_eff - V_t)^{3/2} / (dV_t/dphi)^2 dphi

        Bad evaluation points (non-finite, imaginary, V_eff < V_t) contribute zero.
        """
        vt = self.ConstructVt(phi0, T, gD, scale, ls0)

        def integrand(S):
            try:
                h       = max(1e-9, abs(S) * 1e-3)
                vt_p    = vt(S + h)
                vt_m    = vt(S - h)
                vt_val  = vt(S)
                veff_val = self.veff.Veff(S, T, gD, scale, ls0)

                if not all(np.isfinite(x) for x in [vt_p, vt_m, vt_val, veff_val]):
                    return 0.0

                dvt   = (vt_p - vt_m) / (2 * h)
                dvt   = np.sign(dvt) * np.clip(np.abs(dvt), 1e-65, np.inf)
                diff  = veff_val - vt_val
                if diff < 0:
                    return 0.0

                return diff**1.5 / dvt**2
            except Exception:
                return 0.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = quad(integrand, 0, phi0, limit=20_000, epsrel=1e-8, epsabs=1e-8)[0]

        return (32.0 * np.pi * np.sqrt(2.0) / 3.0) * result

    def SE(self, T, gD, scale, ls0):
        """
        Minimised Euclidean action S_E(T) and the optimal tunnelling endpoint.

        Returns
        -------
        (S_E_min, phi0_opt) : tuple of floats
        """
        phi_lo = self.phi_root(T, gD, scale, ls0)
        phi_hi = self.phi_min(T, gD, scale, ls0)
        result = optimize.minimize_scalar(
            lambda S: self.SE0(S, T, gD, scale, ls0),
            bounds=(phi_lo, phi_hi),
            method="bounded",
        )
        return result.fun, result.x
