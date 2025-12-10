# BounceSolFull_RGE.py
# Author: Maura E. Ramirez-Quezada (updated)
# Date: 18.09.2025
# Full effective potential bounce solver (class-based) with (scale, ls0) threading

import numpy as np
import warnings
from scipy import optimize
from scipy.integrate import quad

import constants as cs
from VeffFunc_RGE import VeffRGE


class BounceSolver:
    """
    Bounce solution calculator for phase transitions using the **full** effective potential.

    Public API (all depend on T, gD, scale, ls0):
        - phi_min_Veff0(T, gD, scale, ls0)
        - phi_min(T, gD, scale, ls0)
        - phi_max(T, gD, scale, ls0)
        - phi_root(T, gD, scale, ls0, scale_factor=1.0, verbose=False)
        - ConstructVt(phi0, T, gD, scale, ls0)
        - SE0(phi0, T, gD, scale, ls0, debug=False)
        - SE(T, gD, scale, ls0)
    """

    def __init__(self, veff=None):
        """
        Parameters
        ----------
        veff : VeffRGE or None
            If given, use this preconstructed VeffRGE. Otherwise create a new one with default table.
        """
        self.veff = veff if veff is not None else VeffRGE()

    # =============================
    # Vacuum structure
    # =============================
    def phi_min_Veff0(self, T, gD, scale, ls0, bound_min=(1e-3, 1e7)):
        """Minimum of the tree-level/zero-T piece used as upper bound helper."""
        return optimize.minimize_scalar(
            lambda S: self.veff.Veff0_RGE(S, T, gD, scale, ls0),
            bounds=bound_min,
            method="bounded"
        ).x

    def get_intervals(self, T, gD, scale, ls0):
        """Return a few safe bounded intervals depending on current minimum."""
        phi_min_val = self.phi_min_Veff0(T, gD, scale, ls0)
        upper_bound = phi_min_val
        return (1e-7, upper_bound), (1e-7, upper_bound), (1e-7, 1.0)

    def phi_min(self, T, gD, scale, ls0):
        """Symmetry-broken minimum at finite T for the full potential."""
        phi_interval, _, _ = self.get_intervals(T, gD, scale, ls0)
        result = optimize.minimize_scalar(
            lambda S: self.veff.Veff(S, T, gD, scale, ls0),
            bounds=phi_interval,
            method="bounded"
        )
        return result.x

    def phi_max(self, T, gD, scale, ls0):
        """Location near the top of the barrier (max of -V)."""
        phi_interval, _, _ = self.get_intervals(T, gD, scale, ls0)
        result = optimize.minimize_scalar(
            lambda S: -np.real(self.veff.Veff(S, T, gD, scale, ls0)),
            bounds=phi_interval,
            method="bounded"
        )
        return result.x

    def phi_root(self, T, gD, scale, ls0, scale_factor=1.0, verbose=False):
        """
        Automatic root finder for Veff (rescaled), robust against shallow barriers.
        Finds a root between phi_max and phi_min where V crosses zero.
        """
        phiTmax = self.phi_max(T, gD, scale, ls0)
        phiTmin = self.phi_min(T, gD, scale, ls0)
        Veff_scaled = lambda S: np.real(scale_factor * self.veff.Veff(S, T, gD, scale, ls0))

        # Dense scan to bracket a sign change
        S_vals = np.linspace(phiTmax, phiTmin, 50000)
        V_vals = np.array([Veff_scaled(S) for S in S_vals])

        sign_changes = np.where(np.sign(V_vals[:-1]) != np.sign(V_vals[1:]))[0]
        if len(sign_changes) == 0:
            raise ValueError("No root found beyond the barrier.")

        i = sign_changes[0]
        S_low, S_high = S_vals[i], S_vals[i + 1]

        if verbose:
            print(f"[DEBUG] Root bracket: [{S_low:.4e}, {S_high:.4e}]")
            print(f"[DEBUG] V(S_low)={Veff_scaled(S_low):.4e}, V(S_high)={Veff_scaled(S_high):.4e}")

        result = optimize.root_scalar(Veff_scaled, bracket=(S_low, S_high), xtol=1e-14)
        return result.root

    # =============================
    # Tunneling potential
    # =============================
    def ConstructVt(self, phi0, T, gD, scale, ls0):
        """
        Construct the quartic ansatz V_t(φ) matching V and dV at φ=φ0 and V at φ=φT (near barrier top).
        Returns a callable vt(φ).
        """
        phiT = self.phi_max(T, gD, scale, ls0)
        V0 = self.veff.Veff(phi0, T, gD, scale, ls0)
        epsilon = 1e-9

        def dVeff_ds(s):
            return (self.veff.Veff(s + epsilon, T, gD, scale, ls0) -
                    self.veff.Veff(s - epsilon, T, gD, scale, ls0)) / (2 * epsilon)

        dV0 = dVeff_ds(phi0)
        VT = self.veff.Veff(phiT, T, gD, scale, ls0)

        d = 3
        a1 = V0 / phi0
        a2 = ((d - 1) * phi0 * dV0 - d * V0) / (d * phi0**2)
        a3 = ((d - 1) * phi0 * dV0 - 2 * d * V0) / (d * phi0**3)

        dVt3T = a1 + a2 * (2 * phiT - phi0) + a3 * (3 * phiT - phi0) * (phiT - phi0)
        d2Vt3T = 2 * a2 + 2 * a3 * (3 * phiT - 2 * phi0)

        phi0T = phi0 - phiT
        c = 4 * phiT**2 * phi0T**2 * phi0**2

        Vt3T = (V0 * phiT / phi0 +
                (2 * phi0 * dV0 - 3 * V0) / (3 * phi0**2) * phiT * (phiT - phi0) +
                (2 * phi0 * dV0 - 6 * V0) / (3 * phi0**3) * phiT * (phiT - phi0)**2)

        a0T = (-4 * (VT - Vt3T) * (phi0**2 - 6 * phiT * phi0T) -
               6 * phiT * (phi0T - phiT) * phi0T * dVt3T +
               2 * phiT**2 * phi0T**2 * d2Vt3T)

        Ut3T = 3 * dVt3T**2 + 4 * (VT - Vt3T) * d2Vt3T
        a4 = (a0T - np.sqrt(a0T**2 - c * Ut3T)) / c if a0T**2 - c * Ut3T > 0 else 0.0

        return lambda phi: (a1 * phi +
                            a2 * phi * (phi - phi0) +
                            a3 * phi * (phi - phi0)**2 +
                            a4 * phi**2 * (phi - phi0)**2)

    # =============================
    # Euclidean action
    # =============================
    def SE0(self, phi0, T, gD, scale, ls0, debug=False):
        """
        Euclidean action integrand using the constructed tunneling potential.
        """
        vt = self.ConstructVt(phi0, T, gD, scale, ls0)

        def integrand(S):
            try:
                h = max(1e-9, abs(S) * 1e-3)
                vtS_plus = vt(S + h)
                vtS_minus = vt(S - h)
                vtS = vt(S)
                veffS = self.veff.Veff(S, T, gD, scale, ls0)

                if not all(np.isfinite(x) for x in [vtS_plus, vtS_minus, vtS, veffS]):
                    return 0.0

                dvt_ds = (vtS_plus - vtS_minus) / (2 * h)
                dvt_ds = np.sign(dvt_ds) * np.clip(np.abs(dvt_ds), 1e-65, np.inf)

                v_eff_diff = veffS - vtS
                if v_eff_diff < 0:
                    return 0.0

                return v_eff_diff**1.5 / dvt_ds**2
            except Exception:
                return 0.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = quad(integrand, 0, phi0, limit=20000, epsrel=1e-8, epsabs=1e-8)[0]

        return (32 * np.pi * np.sqrt(2) / 3) * result

    def SE(self, T, gD, scale, ls0):
        """
        Minimize S_E(phi0) between the first root (beyond the barrier) and the broken minimum.
        """
        phi1 = self.phi_root(T, gD, scale, ls0)
        phi2 = self.phi_min(T, gD, scale, ls0)
        result = optimize.minimize_scalar(
            lambda S: self.SE0(S, T, gD, scale, ls0),
            bounds=(phi1, phi2),
            method="bounded"
        )
        return result.fun, result.x
