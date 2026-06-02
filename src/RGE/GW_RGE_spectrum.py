# GW_RGE_spectrum.py
"""
Generic gravitational-wave spectrum builder from macroscopic FOPT parameters.

This module is purely phenomenological: it takes (T_n, alpha, beta/H) as inputs
and returns callable GW spectra for the three standard sources — bubble collisions,
sound waves, and magnetohydrodynamic turbulence.

No Veff calls are made here; microphysics (e.g. kappa_col) must be supplied
externally by the caller.

References
----------
Sound waves / turbulence :  Espinosa et al. (2010); Hindmarsh et al. (2017)
Bubble (PT) template     :  Caprini et al. (2020)
Peak frequencies         :  Caprini & Figueroa (2018) review conventions
"""

import numpy as np
from typing import Callable, Tuple
from scipy.integrate import quad

import constants as cs


# ------------------------------------------------------------------ #
# Peak frequencies and spectral shapes                                  #
# ------------------------------------------------------------------ #

def h_star(T: float) -> float:
    """Characteristic frequency h_*(T) [Hz] at temperature T [GeV]."""
    return 1.65e-5 * (T / 100.0) * (cs.g_dof / 100.0) ** (1.0 / 6.0)


def f_sw(beta_over_H: float, hstar: float) -> float:
    """Peak frequency for the sound-wave source [Hz]."""
    return 0.54 * hstar * beta_over_H


def f_col(beta_over_H: float, hstar: float) -> float:
    """Peak frequency for the bubble-collision source [Hz]."""
    return 0.17 * hstar * beta_over_H


def f_turb(beta_over_H: float, hstar: float) -> float:
    """Peak frequency for the MHD turbulence source [Hz]."""
    return 1.75 * hstar * beta_over_H


def f_PT(beta_over_H: float, hstar: float) -> float:
    """Peak frequency for the PT (bubble-wall) template [Hz]."""
    return (0.8 / (2.0 * np.pi)) * hstar * beta_over_H


def f_star_only(hstar: float) -> float:
    """Hubble-crossing frequency hstar / (2 pi) [Hz]."""
    return hstar / (2.0 * np.pi)


def S_PT(f: float, fPT: float) -> float:
    """Spectral shape of the PT (bubble-wall) template."""
    x = f / fPT
    return 3.0 * x**0.9 / (2.1 + 0.9 * x**3)


def S_H(f: float, fstar: float) -> float:
    """Spectral shape suppression from finite Hubble time (sound-wave lifetime)."""
    x = f / fstar
    return x**2.1 / (1.0 + x**2.1)


def kappa_sw(alpha: float) -> float:
    """
    Efficiency factor for sound-wave energy injection.

    kappa_sw(alpha) from Espinosa et al. (2010).
    """
    return alpha / (0.73 + 0.083 * np.sqrt(alpha) + alpha)


def Htau(alpha: float, beta_over_H: float) -> float:
    """
    Dimensionless sound-shell lifetime H * tau_sw.

    Capped at 1 to avoid unphysical values when the sound shell persists
    longer than a Hubble time.
    """
    term  = (8.0 * np.pi) ** (1.0 / 3.0) / beta_over_H
    denom = ((3.0 / 4.0) * kappa_sw(alpha) * alpha / (1.0 + alpha)) ** 0.5
    if denom <= 0:
        return 1.0
    return min(1.0, term / denom)


# ------------------------------------------------------------------ #
# Spectrum builder class                                                #
# ------------------------------------------------------------------ #

class GWFromParams:
    """
    Build GW spectra from macroscopic FOPT parameters (Tn, alpha, beta/H).

    No effective potential or bounce solver is called.  All microphysics
    must be encoded in the input parameters.

    Parameters
    ----------
    g_star : float
        Effective d.o.f. at the transition (default: cs.g_dof).

    Usage
    -----
    >>> gw = GWFromParams()
    >>> h2_total, h2_col, h2_sw, h2_turb, h2_pT = gw.spectra(Tn, alpha, betaH)
    >>> SNR = gw.snr_lisa(h2_total)
    """

    def __init__(self, g_star: float = cs.g_dof):
        self.g_star = g_star

    def spectra(
        self,
        T: float,
        alpha: float,
        beta_over_H: float,
        kappa_col: float = 0.0,
    ) -> Tuple[Callable[[float], float], ...]:
        """
        Return five callable spectra: (total, collision, sound wave, turbulence, PT template).

        Parameters
        ----------
        T           : nucleation temperature [GeV].
        alpha       : transition strength.
        beta_over_H : inverse transition duration.
        kappa_col   : fraction of latent heat converted directly to bubble-wall
                      kinetic energy (0 = all to plasma; 1 = all to walls).

        Returns
        -------
        (h2_Omega_total, h2_Omega_col, h2_Omega_sw, h2_Omega_turb, h2_Omega_pT)
        each a callable f [Hz] -> h^2 Omega (dimensionless).
        """
        Rval  = cs.R
        hstar = h_star(T)

        fcol  = f_col(beta_over_H, hstar)
        fsw   = f_sw(beta_over_H, hstar)
        fturb = f_turb(beta_over_H, hstar)
        fstar = f_star_only(hstar)
        fPT   = f_PT(beta_over_H, hstar)

        k_sw = kappa_sw(alpha * (1.0 - kappa_col)) * (1.0 - kappa_col)
        Ht   = Htau(alpha, beta_over_H)

        def h2_Omega_col(f):
            x = f / fcol
            return (
                0.028 * Rval * beta_over_H**-2
                * ((kappa_col * alpha) / (1.0 + alpha)) ** 2
                * x**3 * (4.51 / (1.51 + 3.0 * x**2.07)) ** 2.18
            )

        def h2_Omega_sw(f):
            x = f / fsw
            return (
                0.29 * Rval * beta_over_H**-1 * Ht
                * ((k_sw * alpha) / (1.0 + alpha)) ** 2
                * x**3 * (7.0 / (3.0 + 4.0 * x**2)) ** (7.0 / 2.0)
            )

        def h2_Omega_turb(f):
            x = f / fturb
            return (
                20.0 * Rval * beta_over_H**-1 * (1.0 - Ht)
                * ((k_sw * alpha) / (1.0 + alpha)) ** (3.0 / 2.0)
                * x**3 * (1.0 / (1.0 + x)) ** (11.0 / 3.0)
                * (1.0 / (1.0 + 8.0 * np.pi * f / hstar))
            )

        def h2_Omega_pT(f):
            return (
                1e-6 / (self.g_star / 100.0) ** (1.0 / 3.0)
                * beta_over_H**-2
                * (alpha / (1.0 + alpha)) ** 2
                * S_PT(f, fPT) * S_H(f, fstar)
            )

        def h2_Omega_total(f):
            if kappa_col >= 1.0:
                return h2_Omega_pT(f)
            elif kappa_col <= 0.0:
                return h2_Omega_sw(f) + h2_Omega_turb(f)
            else:
                return h2_Omega_col(f) + h2_Omega_sw(f) + h2_Omega_turb(f)

        return h2_Omega_total, h2_Omega_col, h2_Omega_sw, h2_Omega_turb, h2_Omega_pT
