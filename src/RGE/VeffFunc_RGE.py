# VeffFunc_RGE.py
"""
Effective potential with one-loop RGE running and finite-temperature corrections.

This module is the single physics entry point for the FOPT pipeline.
To study a different scalar model, subclass VeffRGE and override:
  - Tree-level and mass methods  (Vtree, mPhi2, mSigma2, mAp2)
  - Debye-mass methods           (PiPhi, PiAp)
  - Coleman-Weinberg term        (Vcw)
  - Daisy resummation            (Vdaisy)

The RGE running is handled by RGESolver, whose beta functions must also be
updated for the new model.

Pipeline position
-----------------
VeffRGE  <--  BounceSolFull_RGE / BounceSolHighT_RGE  <--  FOPT_RGE / FOPT_RGE_real
"""

import numpy as np
from scipy.special import kn
from scipy.interpolate import interp1d

import constants as cs
from src.RGE.RGEsolver import RGESolver


class VeffRGE:
    """
    Effective potential for a dark U(1) scalar model with RGE running.

    The potential has four contributions:
      V_eff = V_tree + V_CW + V_T + V_daisy

    All methods are pure functions of their arguments (no hidden state beyond
    the J_B interpolation table), so the class is safe to share across solvers.

    Parameters
    ----------
    vt_table_path : str
        Path to the tabulated J_B integral (VT_integralNumeric.dat).
    """

    # Convenience complex unit (used to take complex square-roots of possibly
    # negative mass-squared arguments while keeping track of the sign).
    _Re1 = complex(1.0, 0.0)

    def __init__(self, vt_table_path: str = "VT_integralNumeric.dat"):
        data = np.loadtxt(vt_table_path)
        self._JB_interp = interp1d(
            data[:, 0],
            data[:, 1],
            kind="linear",
            fill_value=0.0,
            bounds_error=False,
        )
        # V0 = Veff(S=0) is constant for fixed (T, gD0, scale, lambdaS0).
        # Cache it to avoid recomputing inside tight loops (e.g. quad integrand).
        self._v0_cache_full:  dict = {}
        self._v0_cache_highT: dict = {}
        self._v0_cache_zero:  dict = {}

    # ------------------------------------------------------------------ #
    # Tree-level potential and field-dependent masses                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def Vtree(S, lambdaS, mS):
        """Classical scalar potential: -m^2 S^2/2 + lambda S^4/4."""
        return -mS**2 * S**2 / 2.0 + lambdaS * S**4 / 4.0

    @staticmethod
    def mPhi2(S, lambdaS, mS):
        """Field-space mass squared of the radial (Higgs) mode."""
        return -mS**2 + 3.0 * lambdaS * S**2

    @staticmethod
    def mSigma2(S, lambdaS, mS):
        """Field-space mass squared of the Goldstone (angular) mode."""
        return -mS**2 + lambdaS * S**2

    @staticmethod
    def mAp2(S, gD):
        """Field-space mass squared of the dark photon."""
        return gD**2 * S**2

    # ------------------------------------------------------------------ #
    # Thermal self-energies (Debye masses)                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def PiPhi(gD, T, lambdaS):
        """Thermal self-energy (Debye mass squared) for the scalar modes."""
        return (lambdaS / 3.0 + 0.25 * gD**2) * T**2

    @staticmethod
    def PiAp(gD, T):
        """Thermal self-energy (Debye mass squared) for the dark photon."""
        return gD**2 / 3.0 * T**2

    # ------------------------------------------------------------------ #
    # Coleman–Weinberg potential (MS-bar, vanishing counterterms)          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _kdelta(a, b):
        """Kronecker delta; handles both scalar and array inputs."""
        if isinstance(a, np.ndarray):
            return 1.0 * (a == b)
        return 1.0 if a == b else 0.0

    def Vcw(self, S, mu, gD, lambdaS, mS):
        """
        One-loop Coleman-Weinberg potential in the MS-bar scheme.

        The log arguments are regulated by a Kronecker delta that inserts
        a zero whenever the mass squared vanishes exactly (avoids log(0)).
        The result is taken real to handle tachyonic directions via complex logs.
        """
        Re1 = self._Re1
        m1 = self.mPhi2(S, lambdaS, mS)
        m2 = self.mSigma2(S, lambdaS, mS)
        m3 = self.mAp2(S, gD)

        t1 = m1**2 * (np.log(Re1 * m1 / mu**2 + self._kdelta(m1 / mu**2, 0)) - 3.0 / 2.0)
        t2 = m2**2 * (np.log(Re1 * m2 / mu**2 + self._kdelta(m2 / mu**2, 0)) - 3.0 / 2.0)
        t3 = 3.0 * m3**2 * (np.log(Re1 * m3 / mu**2 + self._kdelta(m3 / mu**2, 0)) - 5.0 / 6.0)

        return np.real((t1 + t2 + t3) / (64.0 * np.pi**2))

    # ------------------------------------------------------------------ #
    # Daisy resummation                                                    #
    # ------------------------------------------------------------------ #

    @classmethod
    def _FD(cls, mi, Pi):
        """Arnold-Espinosa daisy kernel: (m^2+Pi)^{3/2} - (m^2)^{3/2}."""
        return (cls._Re1 * (mi + Pi)) ** 1.5 - (cls._Re1 * mi) ** 1.5

    def Vdaisy(self, S, T, gD, lambdaS, mS):
        """
        Daisy (ring) resummation contribution.

        Resums the leading IR divergences from the zero Matsubara mode of the
        bosonic fields by replacing m^2 -> m^2 + Pi(T).
        """
        Pi_phi = self.PiPhi(gD, T, lambdaS)
        Pi_Ap  = self.PiAp(gD, T)

        fd_phi   = self._FD(self.mPhi2(S, lambdaS, mS),   Pi_phi)
        fd_sigma = self._FD(self.mSigma2(S, lambdaS, mS), Pi_phi)
        fd_Ap    = self._FD(self.mAp2(S, gD),              Pi_Ap)

        return np.real(-(T / (12.0 * np.pi)) * (fd_phi + fd_sigma + fd_Ap))

    # ------------------------------------------------------------------ #
    # Thermal functions                                                    #
    # ------------------------------------------------------------------ #

    def JB(self, y):
        """
        Bosonic thermal function J_B(y) evaluated from the numerical table.

        Returns zero for y >= 600 (exponentially suppressed in that regime).
        """
        return np.where(y < 600, self._JB_interp(y), 0.0)

    @staticmethod
    def JBhighexp(y):
        """
        High-temperature expansion of J_B(y) to order y^2 log(y).

        Valid when m/T << 1.  The log(|y|) term uses a floor of 1e-10 to
        avoid log(0) for massless fields.
        """
        pi = np.pi
        return (
            -pi**4 / 45.0
            + (pi**2 / 12.0) * y
            - (pi / 6.0) * np.maximum(0.0, y) ** 1.5
            - (1.0 / 32.0) * y**2 * (np.log(np.maximum(1e-10, np.abs(y))) - cs.ab)
        )

    # ------------------------------------------------------------------ #
    # Finite-temperature potentials                                        #
    # ------------------------------------------------------------------ #

    def VTfull(self, S, T, gD, lambdaS, mS):
        """Finite-T potential using the full numerical J_B integral."""
        y1 = self.mPhi2(S, lambdaS, mS) / T**2
        y2 = self.mSigma2(S, lambdaS, mS) / T**2
        y3 = self.mAp2(S, gD) / T**2
        return T**4 / (2.0 * np.pi**2) * (self.JB(y1) + self.JB(y2) + 3.0 * self.JB(y3))

    def V_highT(self, S, T, gD, lambdaS, mS):
        """Finite-T potential using the high-T expansion of J_B."""
        y1 = self.mPhi2(S, lambdaS, mS) / T**2
        y2 = self.mSigma2(S, lambdaS, mS) / T**2
        y3 = self.mAp2(S, gD) / T**2
        return T**4 / (2.0 * np.pi**2) * (
            self.JBhighexp(y1) + self.JBhighexp(y2) + 3.0 * self.JBhighexp(y3)
        )

    # ------------------------------------------------------------------ #
    # Effective potentials                                                 #
    # ------------------------------------------------------------------ #

    def Veff0(self, S, gD0, ls0=cs.lambdaS0, mS0=cs.mS0):
        """
        Zero-temperature effective potential normalised to V(0) = 0.

        Uses the bare (IR) parameters at mu0 = 1 GeV without RGE running.
        This is the T=0 fallback branch of Veff.

        Parameters
        ----------
        S    : field value [GeV]
        gD0  : dark gauge coupling at mu0
        ls0  : scalar quartic at mu0 (default: cs.lambdaS0)
        mS0  : scalar mass parameter at mu0 (default: cs.mS0)
        """
        mu0 = 1.0
        V0  = self.Vtree(0, ls0, mS0) + self.Vcw(0, mu0, gD0, ls0, mS0)
        return np.real(self.Vtree(S, ls0, mS0) + self.Vcw(S, mu0, gD0, ls0, mS0) - V0)

    def Veff0_RGE(self, S, T, gD0, scale, lambdaS0):
        """
        Zero-temperature effective potential with RGE-improved parameters.

        Runs (gD, lambdaS, mS) from mu0 = 1 GeV to mu = scale*T, then
        evaluates (V_tree + V_CW) normalised to zero at S = 0.

        Parameters
        ----------
        S        : field value [GeV]
        T        : temperature used to set the renormalisation scale [GeV]
        gD0      : dark gauge coupling at mu0
        scale    : dimensionless scale factor; mu = scale * T
        lambdaS0 : scalar quartic at mu0
        """
        mu = scale * T
        gD, lambdaS, mS = RGESolver.run_params(mu, gD0, lambdaS0, cs.mS0)
        key = (float(T), float(gD0), float(scale), float(lambdaS0))
        if key not in self._v0_cache_zero:
            self._v0_cache_zero[key] = complex(
                self.Vtree(0, lambdaS, mS) + self.Vcw(0, mu, gD, lambdaS, mS)
            )
        V0 = self._v0_cache_zero[key]
        return np.real(self.Vtree(S, lambdaS, mS) + self.Vcw(S, mu, gD, lambdaS, mS) - V0)

    def Veff(self, S, T, gD0, scale, lambdaS0):
        """
        Full finite-temperature effective potential with RGE running.

        Includes tree-level, CW, full thermal J_B, and daisy terms.
        For T <= 0 the call falls back to Veff0 (zero-temperature potential).

        Parameters
        ----------
        S        : field value [GeV]
        T        : temperature [GeV]
        gD0      : dark gauge coupling at mu0
        scale    : renormalisation scale factor (mu = scale * T)
        lambdaS0 : scalar quartic at mu0
        """
        mu = scale * T
        gD, lambdaS, mS = RGESolver.run_params(mu, gD0, lambdaS0, cs.mS0)
        key = (float(T), float(gD0), float(scale), float(lambdaS0))
        if key not in self._v0_cache_full:
            self._v0_cache_full[key] = complex(
                self.Vtree(0, lambdaS, mS)
                + self.Vcw(0, mu, gD, lambdaS, mS)
                + self.VTfull(0, T, gD, lambdaS, mS)
                + self.Vdaisy(0, T, gD, lambdaS, mS)
            )
        V0 = self._v0_cache_full[key]
        result = (
            self.Vtree(S, lambdaS, mS)
            + self.Vcw(S, mu, gD, lambdaS, mS)
            + self.VTfull(S, T, gD, lambdaS, mS)
            + self.Vdaisy(S, T, gD, lambdaS, mS)
            - V0
        )
        return np.where(T > 0, np.real(result), self.Veff0(S, gD0, lambdaS0))

    def Veff_HighT(self, S, T, gD0, scale, lambdaS0):
        """
        High-temperature effective potential with RGE running.

        Replaces the full J_B integral with its high-T expansion.
        Useful above the electroweak scale where m/T << 1.

        Parameters
        ----------
        S        : field value [GeV]
        T        : temperature [GeV]
        gD0      : dark gauge coupling at mu0
        scale    : renormalisation scale factor (mu = scale * T)
        lambdaS0 : scalar quartic at mu0
        """
        mu = scale * T
        gD, lambdaS, mS = RGESolver.run_params(mu, gD0, lambdaS0, cs.mS0)
        key = (float(T), float(gD0), float(scale), float(lambdaS0))
        if key not in self._v0_cache_highT:
            self._v0_cache_highT[key] = complex(
                self.Vtree(0, lambdaS, mS)
                + self.Vcw(0, mu, gD, lambdaS, mS)
                + self.V_highT(0, T, gD, lambdaS, mS)
                + self.Vdaisy(0, T, gD, lambdaS, mS)
            )
        V0 = self._v0_cache_highT[key]
        result = (
            self.Vtree(S, lambdaS, mS)
            + self.Vcw(S, mu, gD, lambdaS, mS)
            + self.V_highT(S, T, gD, lambdaS, mS)
            + self.Vdaisy(S, T, gD, lambdaS, mS)
            - V0
        )
        return np.where(T > 0, np.real(result), self.Veff0(S, gD0, lambdaS0))
