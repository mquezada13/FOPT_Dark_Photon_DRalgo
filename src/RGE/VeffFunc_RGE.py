# VeffFunc_RGE.py
"""
Effective potential backend with RGE running and finite-T corrections.
"""

import numpy as np
from scipy.special import kn
from scipy.interpolate import interp1d
import cmath

import constants as cs
from RGEsolver import RGESolver


class VeffRGE:
    """
    Effective potential with RGE and finite-temperature corrections.
    """

    # Complex helpers
    Re1 = complex(1., 0.)
    Imi = complex(0., 1.)

    def __init__(self, vt_table_path: str = "VT_integralNumeric.dat"):
        """
        Load the J-integral table for thermal functions.

        Parameters
        ----------
        vt_table_path : str
            Path to tabulated integral for JB.
        """
        IntforVTdata = np.loadtxt(vt_table_path)
        self.Integrand = interp1d(
            IntforVTdata[:, 0],
            IntforVTdata[:, 1],
            kind="linear",
            fill_value=0,
            bounds_error=False
        )

    # ========================
    # Tree-level and masses
    # ========================
    @staticmethod
    def Vtree(S, lambdaS, mS):
        return -mS**2 * S**2 / 2 + lambdaS * (S**4 / 4)

    @staticmethod
    def mPhi2(S, lambdaS, mS):
        return -mS**2 + 3 * lambdaS * S**2

    @staticmethod
    def mSigma2(S, lambdaS, mS):
        return -mS**2 + lambdaS * S**2

    @staticmethod
    def mAp2(S, gD):
        return gD**2 * S**2

    # ========================
    # Debye masses
    # ========================
    @staticmethod
    def PiPhi(gD, T, lambdaS):
        return (lambdaS / 3 + 0.25 * gD**2) * T**2

    @classmethod
    def cut(cls, y):
        return y / 2 * kn(2, cls.Re1 * np.sqrt(cls.Re1 * y))

    @staticmethod
    def PiAp(gD, T):
        return gD**2 / 3 * T**2

    # ========================
    # Coleman–Weinberg
    # ========================
    @staticmethod
    def _kdelta(a, b):
        if isinstance(a, np.ndarray):
            return 1.0 * (a == b)
        return 1.0 if a == b else 0.0

    @classmethod
    def Vcw(cls, S, mu, gD, lambdaS, mS):
        mPhi2_val = cls.mPhi2(S, lambdaS, mS)
        mSigma2_val = cls.mSigma2(S, lambdaS, mS)
        mAp2_val = cls.mAp2(S, gD)

        term1 = mPhi2_val**2 * (np.log(cls.Re1 * mPhi2_val / mu**2 + cls._kdelta(mPhi2_val / mu**2, 0)) - 3/2)
        term2 = mSigma2_val**2 * (np.log(cls.Re1 * mSigma2_val / mu**2 + cls._kdelta(mSigma2_val / mu**2, 0)) - 3/2)
        term3 = 3 * mAp2_val**2 * (np.log(cls.Re1 * mAp2_val / mu**2 + cls._kdelta(mAp2_val / mu**2, 0)) - 5/6)

        return np.real(1 / (64 * np.pi**2) * (term1 + term2 + term3))

    # ========================
    # Daisy
    # ========================
    @classmethod
    def _FD(cls, mi, Pi):
        return (cls.Re1 * (mi + Pi))**1.5 - (cls.Re1 * mi)**1.5

    @classmethod
    def Vdaisy(cls, S, T, gD, lambdaS, mS):
        PiPhi_val = cls.PiPhi(gD, T, lambdaS)
        PiAp_val = cls.PiAp(gD, T)

        FD_mPhi = cls._FD(cls.mPhi2(S, lambdaS, mS), PiPhi_val)
        FD_mSigma = cls._FD(cls.mSigma2(S, lambdaS, mS), PiPhi_val)
        FD_mAp = cls._FD(cls.mAp2(S, gD), PiAp_val)

        return np.real(-(T / (12 * np.pi)) * (FD_mPhi + FD_mSigma + FD_mAp))

    # ========================
    # Thermal functions
    # ========================
    def JB(self, y):
        return np.where(y < 600, self.Integrand(y), 0)

    @staticmethod
    def JBhighexp(y):
        pi = np.pi
        term1 = -pi**4 / 45
        term2 = (pi**2 / 12) * y
        term3 = -(pi / 6) * np.maximum(0, y)**1.5
        term4 = -(1 / 32) * y**2 * (np.log(np.maximum(1e-10, np.abs(y))) - cs.ab)
        return term1 + term2 + term3 + term4

    # ========================
    # Finite-T potentials
    # ========================
    def VTfull(self, S, T, gD, lambdaS, mS):
        mPhi2_val = self.mPhi2(S, lambdaS, mS) / T**2
        mSigma2_val = self.mSigma2(S, lambdaS, mS) / T**2
        mAp2_val = self.mAp2(S, gD) / T**2
        return (T**4) / (2 * np.pi**2) * (self.JB(mPhi2_val) + self.JB(mSigma2_val) + 3 * self.JB(mAp2_val))

    def V_highT(self, S, T, gD, lambdaS, mS):
        mPhi2_val = self.mPhi2(S, lambdaS, mS) / T**2
        mSigma2_val = self.mSigma2(S, lambdaS, mS) / T**2
        mAp2_val = self.mAp2(S, gD) / T**2
        return (T**4) / (2 * np.pi**2) * (
            self.JBhighexp(mPhi2_val) + self.JBhighexp(mSigma2_val) + 3 * self.JBhighexp(mAp2_val)
        )

    # ========================
    # Effective potential
    # ========================
    def Veff0(self, S, gD0):
        mu0 = 1.0
        Veff_at_zero = self.Vtree(0, cs.lambdaS0, cs.mS0) + self.Vcw(0, mu0, gD0, cs.lambdaS0, cs.mS0)
        return np.real(self.Vtree(S, cs.lambdaS0, cs.mS0) + self.Vcw(S, mu0, gD0, cs.lambdaS0, cs.mS0) - Veff_at_zero)

    def Veff0_RGE(self, S, T, gD0, scale, lambdaS0):
        mu = scale * T
        gD, lambdaS, mS = RGESolver.run_params(mu, gD0, lambdaS0, cs.mS0)
        Veff_at_zero = self.Vtree(0, lambdaS, mS) + self.Vcw(0, mu, gD, lambdaS, mS)
        return np.real(self.Vtree(S, lambdaS, mS) + self.Vcw(S, mu, gD, lambdaS, mS) - Veff_at_zero)

    def Veff(self, S, T, gD0, scale, lambdaS0 ):
        mu = scale * T
        gD, lambdaS, mS = RGESolver.run_params(mu, gD0, lambdaS0, cs.mS0)

        Veff_at_zero_S = (
            self.Vtree(0, lambdaS, mS)
            + self.Vcw(0, mu, gD, lambdaS, mS)
            + self.VTfull(0, T, gD, lambdaS, mS)
            + self.Vdaisy(0, T, gD, lambdaS, mS)
        )
        result = (
            self.Vtree(S, lambdaS, mS)
            + self.Vcw(S, mu, gD, lambdaS, mS)
            + self.VTfull(S, T, gD, lambdaS, mS)
            + self.Vdaisy(S, T, gD, lambdaS, mS)
            - Veff_at_zero_S
        )
        return np.where(T > 0, np.real(result), self.Veff0(S, gD0))

    def Veff_HighT(self, S, T, gD0, scale, lambdaS0):
        mu = scale * T
        gD, lambdaS, mS = RGESolver.run_params(mu, gD0, lambdaS0, cs.mS0)

        Veff_at_zero_S = (
            self.Vtree(0, lambdaS, mS)
            + self.Vcw(0, mu, gD, lambdaS, mS)
            + self.V_highT(0, T, gD, lambdaS, mS)
            + self.Vdaisy(0, T, gD, lambdaS, mS)
        )
        result = (
            self.Vtree(S, lambdaS, mS)
            + self.Vcw(S, mu, gD, lambdaS, mS)
            + self.V_highT(S, T, gD, lambdaS, mS)
            + self.Vdaisy(S, T, gD, lambdaS, mS)
            - Veff_at_zero_S
        )
        return np.where(T > 0, np.real(result), self.Veff0(S, gD0))

