# BounceSolHighT_RGE.py
# Author: Maura E. Ramirez-Quezada
# Date: 16.07.2025
# High-T expansion bounce solver (class-based)

import numpy as np
import warnings
from scipy import optimize
from scipy.integrate import quad

import constants as cs
from VeffFunc_RGE import VeffRGE


class BounceSolverHighT:
    """
    Bounce solution calculator using the high-T expansion effective potential.

    Public API:
        - phi_min_Veff0(T, gD)
        - phi_max(T, gD)
        - phi_root_HT(T, gD)
        - ConstructVt(phi0, T, gD)
        - SE0(phi0, T, gD)
        - SE(T, gD)
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
    def phi_min_Veff0(self, T, gD, scale, ls0, bound_min=(0, 1e7)):
        return optimize.minimize_scalar(
            lambda S: self.veff.Veff0_RGE(S, T, gD, scale, ls0),
            bounds=bound_min,
            method="bounded"
        ).x

    def get_intervals(self, T, gD, scale, ls0):
        phi_min_val = self.phi_min_Veff0(T, gD, scale, ls0)
        upper_bound = phi_min_val
        return (1e-7, upper_bound), (1e-6, upper_bound), (1e-7, 0.3)

    def phi_max(self, T, gD, scale, ls0):
        phi_interval, _, _ = self.get_intervals(T, gD, scale,ls0)
        result = optimize.minimize_scalar(
            lambda S: -np.real(self.veff.Veff_HighT(S, T, gD, scale, ls0)),
            bounds=phi_interval,
            method="bounded"
        )
        return result.x

    def phi_root_HT(self, T, gD, scale, ls0, verbose=False):
        """
        Robust root finder for Veff_HighT, works even when the barrier is shallow/narrow.
        """
        phiTmax = self.phi_max(T, gD, scale, ls0)
        phiTmin = self.phi_min_Veff0(T, gD, scale, ls0)
        Veff_scaled = lambda S: np.real(self.veff.Veff_HighT(S, T, gD, scale, ls0))

        # Scan
        S_vals = np.linspace(phiTmax, phiTmin, 50000)
        V_vals = np.array([Veff_scaled(S) for S in S_vals])

        # Check sign change
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
        phiT = self.phi_max(T, gD, scale, ls0)
        V0 = self.veff.Veff_HighT(phi0, T, gD, scale, ls0)
        epsilon = 1e-9

        def dVeff_ds(s):
            return (self.veff.Veff_HighT(s + epsilon, T, gD, scale, ls0) -
                    self.veff.Veff_HighT(s - epsilon, T, gD, scale, ls0)) / (2 * epsilon)

        dV0 = dVeff_ds(phi0)
        VT = self.veff.Veff_HighT(phiT, T, gD, scale, ls0)

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
        a4 = (a0T - np.sqrt(a0T**2 - c * Ut3T)) / c if a0T**2 - c * Ut3T > 0 else 0

        def vt(phi):
            return (a1 * phi +
                    a2 * phi * (phi - phi0) +
                    a3 * phi * (phi - phi0)**2 +
                    a4 * phi**2 * (phi - phi0)**2)

        return vt, (a1, a2, a3, a4)

    def dVt(self, phi, phi0, a1, a2, a3, a4):
        return (
            a1 +
            a2 * (2 * phi - phi0) +
            a3 * (3 * phi**2 - 4 * phi * phi0 + phi0**2) +
            a4 * (4 * phi**3 - 6 * phi**2 * phi0 + 2 * phi * phi0**2)
        )

    # =============================
    # Euclidean action
    # =============================
    def SE0(self, phi0, T, gD, scale, ls0):
        vt, (a1, a2, a3, a4) = self.ConstructVt(phi0, T, gD, scale, ls0)

        def integrand(S):
            vt_val = vt(S)
            veff_val = self.veff.Veff_HighT(S, T, gD, scale, ls0)
            dvt_val = self.dVt(S, phi0, a1, a2, a3, a4)

            dvt_val = max(np.abs(dvt_val), 1e-65)
            diff = max(veff_val - vt_val, 0.0)
            return diff**1.5 / dvt_val**2

        result = quad(integrand, 0, phi0, limit=5000, epsrel=1e-6, epsabs=1e-6)[0]
        return (32 * np.pi * np.sqrt(2) / 3) * result

    def SE(self, T, gD, scale, ls0):
        phi1 = self.phi_root_HT(T, gD, scale, ls0)
        phi2 = self.phi_min_Veff0(T, gD,scale,ls0)
        result = optimize.minimize_scalar(
            lambda S: self.SE0(S, T, gD,scale,ls0),
            bounds=(phi1, phi2),
            method="bounded"
        )
        return result.fun, result.x

