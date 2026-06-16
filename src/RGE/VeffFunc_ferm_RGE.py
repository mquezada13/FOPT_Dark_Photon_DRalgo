# VeffFunc_ferm_RGE.py
"""
Effective potential for U(1)_D + complex scalar + Dirac fermion with RGE running.

The Dirac fermion chi has charges ±1/2 under U(1)_D and couples to the complex
scalar S via a Yukawa interaction y * S * chi_L * chi_R + h.c., giving a
field-dependent mass m_chi = y*S/sqrt(2).

Differences from VeffFunc_RGE (no-fermion model):
  - Vcw adds a fermion loop: -4 * m_chi^4 / (64 pi^2) * (log ... - 3/2)
  - V_highT adds a fermionic thermal function: -4 * T^4/(2pi^2) * J_F(m_chi^2/T^2)
  - PiPhi gains a +y^2/12 * T^2 correction
  - PiAp = 5/12 * gD^2 * T^2  (scalar 1/3 + fermion 1/12)
  - Four RGE parameters are run: gD, lambdaS, mS^2, y  (via RGESolver_fermion)
  - No full J_B table is loaded; Veff falls back to Veff_HighT

Pipeline interface:
  Identical to VeffRGE — bounce solvers and FOPTUtilities can be passed a
  VeffRGE_fermion instance without any changes to those modules.

  The Yukawa coupling y0 and initial mass-squared mS02 are baked in at
  construction time; gD0 and lambdaS0 remain per-call scan parameters.

Pipeline position:
  VeffRGE_fermion  <--  BounceSolFull_RGE / BounceSolHighT_RGE  <--  FOPT_RGE_real
"""

import numpy as np

import constants as cs
from src.RGE.RGESolver_fermion import RGESolver_fermion


class VeffRGE_fermion:
    """
    Effective potential for the dark U(1) + Dirac fermion model with RGE running.

    Parameters
    ----------
    y0   : float
        Yukawa coupling at the reference scale mu0 = 1 GeV.
    mS02 : float, optional
        Initial scalar mass-squared parameter mS^2(mu0) [GeV^2].  Default 0.
    """

    _Re1 = complex(1.0, 0.0)

    def __init__(self, y0: float, mS02: float = 0.0):
        self.y0   = float(y0)
        self.mS02 = float(mS02)
        self._v0_cache_highT: dict = {}
        self._v0_cache_zero:  dict = {}

    # ------------------------------------------------------------------ #
    # Tree-level potential and field-dependent masses                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def Vtree(S, lambdaS, mS2):
        """Classical scalar potential: -mS2 * S^2/2 + lambdaS * S^4/4."""
        return -mS2 * S**2 / 2.0 + lambdaS * S**4 / 4.0

    @staticmethod
    def mPhi2(S, lambdaS, mS2):
        """Mass squared of the radial (Higgs) mode."""
        return -mS2 + 3.0 * lambdaS * S**2

    @staticmethod
    def mSigma2(S, lambdaS, mS2):
        """Mass squared of the Goldstone (angular) mode."""
        return -mS2 + lambdaS * S**2

    @staticmethod
    def mAp2(S, gD):
        """Mass squared of the dark photon."""
        return gD**2 * S**2

    @staticmethod
    def mchi2(S, y):
        """Mass squared of the Dirac fermion chi: m_chi = y*S/sqrt(2)."""
        return y**2 * S**2 / 2.0

    # ------------------------------------------------------------------ #
    # Thermal self-energies (Debye masses)                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def PiPhi(gD, y, T, lambdaS):
        """Scalar Debye mass squared. Fermion adds y^2/12 * T^2."""
        return (lambdaS / 3.0 + 0.25 * gD**2 + y**2 / 12.0) * T**2

    @staticmethod
    def PiAp(gD, T):
        """Dark-photon Debye mass squared: 5/12 * gD^2 * T^2 (scalar 1/3 + fermion 1/12)."""
        return 5.0 * gD**2 / 12.0 * T**2

    # ------------------------------------------------------------------ #
    # Coleman–Weinberg potential                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _kdelta(a, b):
        if isinstance(a, np.ndarray):
            return 1.0 * (a == b)
        return 1.0 if a == b else 0.0

    @classmethod
    def Vcw(cls, S, mu, gD, y, lambdaS, mS2):
        """
        One-loop CW potential in the MS-bar scheme.

        Boson loops get c = 3/2 (scalar) or 5/6 (gauge).
        Fermion loop gets c = 3/2 with an overall factor of -4
        (2 spin × 2 particle/antiparticle for one Dirac fermion).
        """
        Re1 = cls._Re1
        m1 = cls.mPhi2(S, lambdaS, mS2)
        m2 = cls.mSigma2(S, lambdaS, mS2)
        m3 = cls.mAp2(S, gD)
        m4 = cls.mchi2(S, y)

        kd = cls._kdelta
        t1 =       m1**2 * (np.log(Re1 * m1 / mu**2 + kd(m1 / mu**2, 0)) - 3.0 / 2.0)
        t2 =       m2**2 * (np.log(Re1 * m2 / mu**2 + kd(m2 / mu**2, 0)) - 3.0 / 2.0)
        t3 = 3.0 * m3**2 * (np.log(Re1 * m3 / mu**2 + kd(m3 / mu**2, 0)) - 5.0 / 6.0)
        t4 = -4.0 * m4**2 * (np.log(Re1 * m4 / mu**2 + kd(m4 / mu**2, 0)) - 3.0 / 2.0)

        return np.real((t1 + t2 + t3 + t4) / (64.0 * np.pi**2))

    # ------------------------------------------------------------------ #
    # Daisy resummation (bosons only; fermions have no zero Matsubara mode) #
    # ------------------------------------------------------------------ #

    @classmethod
    def _FD(cls, mi, Pi):
        return (cls._Re1 * (mi + Pi)) ** 1.5 - (cls._Re1 * mi) ** 1.5

    @classmethod
    def Vdaisy(cls, S, T, gD, y, lambdaS, mS2):
        """Arnold-Espinosa daisy resummation for the bosonic zero modes."""
        Pi_phi = cls.PiPhi(gD, y, T, lambdaS)
        Pi_Ap  = cls.PiAp(gD, T)
        fd_phi   = cls._FD(cls.mPhi2(S, lambdaS, mS2),   Pi_phi)
        fd_sigma = cls._FD(cls.mSigma2(S, lambdaS, mS2), Pi_phi)
        fd_Ap    = cls._FD(cls.mAp2(S, gD),               Pi_Ap)
        return np.real(-(T / (12.0 * np.pi)) * (fd_phi + fd_sigma + fd_Ap))

    # ------------------------------------------------------------------ #
    # Thermal functions                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def JBhighexp(y):
        """High-T expansion of the bosonic thermal function J_B(y)."""
        pi = np.pi
        return (
            -pi**4 / 45.0
            + (pi**2 / 12.0) * y
            - (pi / 6.0) * np.maximum(0.0, y) ** 1.5
            - (1.0 / 32.0) * y**2 * (np.log(np.maximum(1e-10, np.abs(y))) - cs.ab)
        )

    @staticmethod
    def JFhighexp(y):
        """High-T expansion of the fermionic thermal function J_F(y)."""
        pi = np.pi
        return (
            7.0 * pi**4 / 360.0
            - (pi**2 / 24.0) * y
            - (1.0 / 32.0) * y**2 * (np.log(np.maximum(1e-10, np.abs(y))) - cs.af)
        )

    @classmethod
    def _VT_highT(cls, S, T, gD, y, lambdaS, mS2):
        """High-T expanded thermal potential (bosons + fermion)."""
        if T <= 0.0:
            return 0.0 * S
        y1 = cls.mPhi2(S, lambdaS, mS2)   / T**2
        y2 = cls.mSigma2(S, lambdaS, mS2) / T**2
        y3 = cls.mAp2(S, gD)               / T**2
        y4 = cls.mchi2(S, y)               / T**2
        return T**4 / (2.0 * np.pi**2) * (
            cls.JBhighexp(y1) + cls.JBhighexp(y2) + 3.0 * cls.JBhighexp(y3)
            - 4.0 * cls.JFhighexp(y4)
        )

    # ------------------------------------------------------------------ #
    # Pipeline-compatible effective potentials                             #
    # ------------------------------------------------------------------ #

    def Veff0(self, S, gD0, ls0=0.0):
        """
        Zero-temperature effective potential normalised to V(0) = 0.

        Uses bare parameters at mu0 = 1 GeV without RGE running.
        """
        mu0 = 1.0
        y   = self.y0
        m2  = self.mS02
        V0 = self.Vtree(0, ls0, m2) + self.Vcw(0, mu0, gD0, y, ls0, m2)
        return np.real(self.Vtree(S, ls0, m2) + self.Vcw(S, mu0, gD0, y, ls0, m2) - V0)

    def Veff0_RGE(self, S, T, gD0, scale, ls0):
        """
        Zero-temperature effective potential with RGE-improved parameters.

        Runs (gD, lambdaS, mS^2, y) from mu0 = 1 GeV to mu = scale * T,
        then evaluates (V_tree + V_CW) normalised to zero at S = 0.
        """
        mu = scale * T
        gD, lambdaS, mS2, y = RGESolver_fermion.run_params(
            mu, gD0, ls0, self.mS02, self.y0
        )
        key = (float(T), float(gD0), float(scale), float(ls0))
        if key not in self._v0_cache_zero:
            self._v0_cache_zero[key] = complex(
                self.Vtree(0, lambdaS, mS2) + self.Vcw(0, mu, gD, y, lambdaS, mS2)
            )
        V0 = self._v0_cache_zero[key]
        return np.real(
            self.Vtree(S, lambdaS, mS2) + self.Vcw(S, mu, gD, y, lambdaS, mS2) - V0
        )

    def Veff_HighT(self, S, T, gD0, scale, ls0):
        """
        Full high-T effective potential with RGE running, normalised to V(0) = 0.

        Includes tree, CW, high-T bosonic + fermionic thermal functions, and daisy.
        Falls back to Veff0 when T <= 0.
        """
        mu = scale * T
        gD, lambdaS, mS2, y = RGESolver_fermion.run_params(
            mu, gD0, ls0, self.mS02, self.y0
        )
        key = (float(T), float(gD0), float(scale), float(ls0))
        if key not in self._v0_cache_highT:
            self._v0_cache_highT[key] = complex(
                self.Vtree(0, lambdaS, mS2)
                + self.Vcw(0, mu, gD, y, lambdaS, mS2)
                + self._VT_highT(0, T, gD, y, lambdaS, mS2)
                + self.Vdaisy(0, T, gD, y, lambdaS, mS2)
            )
        V0 = self._v0_cache_highT[key]
        result = (
            self.Vtree(S, lambdaS, mS2)
            + self.Vcw(S, mu, gD, y, lambdaS, mS2)
            + self._VT_highT(S, T, gD, y, lambdaS, mS2)
            + self.Vdaisy(S, T, gD, y, lambdaS, mS2)
            - V0
        )
        return np.where(T > 0, np.real(result), self.Veff0(S, gD0, ls0))

    def Veff(self, S, T, gD0, scale, ls0):
        """
        Full effective potential (alias for Veff_HighT).

        No J_B numerical table is needed for the fermion model; the high-T
        expansion is used throughout.
        """
        return self.Veff_HighT(S, T, gD0, scale, ls0)
