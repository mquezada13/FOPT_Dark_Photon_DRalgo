# GW_field_LV20.py
"""
Gravitational-wave spectrum from bubble-wall field gradients (runaway regime).

Based on Lewicki & Vaskonen (arXiv:2007.04967) — "LV20".

This module is self-contained: it takes (Tn, alpha, beta/H) and the LV20
spectral-shape parameters as inputs, and returns a callable h^2 Omega_GW(f)
evaluated today.

Usage (minimal)
---------------
>>> from src.RGE.GW_field_LV20 import GWFieldLV20
>>> gwf = GWFieldLV20(lv20_key="U1")        # reads cs.LV20_PRESETS["U1"]
>>> h2  = gwf.from_params(Tn, alpha, betaH)  # returns callable h2(f)
>>> SNR = gwf.snr_lisa(h2)

Or using the module-level convenience function:
>>> h2, gwf = from_params(Tn, alpha, betaH, lv20_key="U1")
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Dict, Optional, Tuple
from scipy.integrate import quad

import constants as cs


# ------------------------------------------------------------------ #
# Cosmological helpers                                                  #
# ------------------------------------------------------------------ #

def h_star(T: float) -> float:
    """
    Characteristic frequency h_*(T) [Hz] at the transition temperature T [GeV].

    Uses the standard LISA convention (cs.g_dof).
    """
    return 1.65e-5 * (T / 100.0) * (cs.g_dof / 100.0) ** (1.0 / 6.0)


def T_reh(Tn: float, alpha: float) -> float:
    """
    Estimate of the reheating temperature T_* for strongly supercooled transitions.

    T_* = Tn * (1 + alpha)^{1/4}.
    """
    return Tn * (1.0 + max(alpha, 0.0)) ** 0.25


# ------------------------------------------------------------------ #
# Spectral shape                                                        #
# ------------------------------------------------------------------ #

def lv20_shape(
    omega_over_beta: np.ndarray | float,
    A: float,
    omegabar_over_beta: float,
    a: float,
    b: float,
    c: float,
    d: Optional[float] = None,
    omegad_over_beta: Optional[float] = None,
) -> np.ndarray | float:
    """
    LV20 broken power-law spectral shape (Eq. 19).

    For the U(1) gauge field case an additional bracket suppression
    (parameterised by d and omegad_over_beta) is included in the denominator.

    Parameters
    ----------
    omega_over_beta     : omega / beta evaluated at the frequencies of interest.
    A                   : overall amplitude.
    omegabar_over_beta  : peak position omega_bar / beta.
    a, b                : low- and high-frequency spectral indices.
    c                   : broken power-law transition sharpness.
    d, omegad_over_beta : U(1) gauge suppression parameters (optional).
    """
    x    = np.asarray(omega_over_beta, dtype=float) / omegabar_over_beta
    core = (A * (a + b) / c) * (b * x ** (-a / c) + a * x ** (b / c)) ** (-c)

    if d is None or omegad_over_beta is None:
        return core

    y   = np.asarray(omega_over_beta, dtype=float) / omegad_over_beta
    pre = (1.0 + x**a / y ** (d - a)) / (1.0 + (omegabar_over_beta / omegad_over_beta) ** (d - a))
    return core / pre


def h2Omega_LV20(
    f: np.ndarray | float,
    alpha: float,
    beta_over_H: float,
    Tstar: float,
    gstar: float,
    lv20_params: Dict[str, float],
    h: float = 0.674,
) -> np.ndarray | float:
    """
    GW spectrum h^2 Omega_GW(f) today from LV20, Eqs. (20)-(21).

    Parameters
    ----------
    f           : frequency array [Hz].
    alpha       : transition strength.
    beta_over_H : inverse transition duration.
    Tstar       : T_* used to set the peak frequency [GeV].
    gstar       : effective d.o.f. at T_*.
    lv20_params : dict of shape parameters forwarded to lv20_shape().
    h           : dimensionless Hubble constant (default 0.674).
    """
    hstar           = h_star(Tstar)
    omega_over_beta = 2.0 * np.pi * np.asarray(f, dtype=float) / (beta_over_H * hstar)
    S               = lv20_shape(omega_over_beta, **lv20_params)
    pref            = (
        1.67e-5 * h**2
        * beta_over_H**-2
        * (alpha / (1.0 + alpha)) ** 2
        * (100.0 / gstar) ** (1.0 / 3.0)
    )
    return pref * S


# ------------------------------------------------------------------ #
# Builder class                                                         #
# ------------------------------------------------------------------ #

class GWFieldLV20:
    """
    Builder for LV20 field-gradient GW spectra.

    Parameters
    ----------
    g_star      : effective d.o.f. at T_* (default: cs.g_dof).
    lv20_params : explicit shape-parameter dict.  If given, takes priority
                  over lv20_key.
    lv20_key    : key into cs.LV20_PRESETS (default "U1").
    """

    def __init__(
        self,
        g_star: float = cs.g_dof,
        lv20_params: Optional[Dict[str, float]] = None,
        lv20_key: str = "U1",
    ):
        self.g_star = g_star
        if lv20_params is not None:
            self.params = lv20_params
        else:
            try:
                self.params = cs.LV20_PRESETS[lv20_key]
            except (AttributeError, KeyError) as exc:
                raise ValueError(
                    f"cs.LV20_PRESETS['{lv20_key}'] not found.  "
                    f"Define LV20_PRESETS in constants.py or pass lv20_params directly."
                ) from exc

    def spectrum(
        self, Tstar: float, alpha: float, beta_over_H: float
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """Return h2(f) as a callable for the given (T*, alpha, beta/H)."""
        def _h2(f):
            return h2Omega_LV20(f, alpha, beta_over_H, Tstar, self.g_star, self.params)
        return _h2

    def from_params(
        self,
        Tn: float,
        alpha: float,
        beta_over_H: float,
        Tstar_mode: str = "reh",
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """
        Build the callable h2(f) from (Tn, alpha, beta/H).

        Parameters
        ----------
        Tstar_mode : 'reh'  ->  T_* = T_reh(Tn, alpha)   (recommended for supercooling)
                     'nuc'  ->  T_* = Tn
        """
        if Tstar_mode == "reh":
            Tstar = T_reh(Tn, alpha)
        elif Tstar_mode == "nuc":
            Tstar = Tn
        else:
            raise ValueError("Tstar_mode must be 'reh' or 'nuc'.")
        return self.spectrum(Tstar, alpha, beta_over_H)

    # ------------------------------------------------------------------ #
    # LISA sensitivity and SNR                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def lisa_noise_h2Omega(f: float) -> float:
        """
        LISA effective noise power-spectral density in units of h^2 Omega.

        Combines instrumental noise and residual confusion foreground.
        """
        H100  = cs.Hhund
        term1 = (5.76e-48 / (2 * np.pi * f) ** 4) * (1.0 + (4e-4 / f) ** 2)
        term2 = 3.6e-41
        term3 = 1.0 + (f / 0.025) ** 2
        pref  = (4 * np.pi**2 / (3 * H100**2)) * f**3 * (10.0 / 3.0)
        return pref * (term1 + term2) * term3

    def snr_lisa(
        self,
        h2_func: Callable[[float], float],
        fmin: float = cs.fmin,
        fmax: float = cs.fmax,
        TLISA: float = cs.T_LISA,
    ) -> float:
        """
        Signal-to-noise ratio for LISA.

        SNR = sqrt( T_LISA * int_{fmin}^{fmax} [h2(f) / N(f)]^2 df )
        """
        integrand = lambda f: (h2_func(f) / self.lisa_noise_h2Omega(f)) ** 2
        val, _    = quad(integrand, fmin, fmax, limit=2000)
        return float(np.sqrt(TLISA * val))

    def h2Omega_thr(self, p: float, f0: float = 1e-3) -> float:
        """
        Threshold GW amplitude for detection above cs.SNRthr.

        Computes the amplitude A such that SNR = cs.SNRthr for the power-law
        spectrum h^2 Omega ~ A (f/f0)^p.
        """
        func = lambda f: (f / f0) ** p
        return cs.SNRthr / self.snr_lisa(func)


# ------------------------------------------------------------------ #
# Module-level convenience function                                     #
# ------------------------------------------------------------------ #

def from_params(
    Tn: float,
    alpha: float,
    beta_over_H: float,
    Tstar_mode: str = "reh",
    g_star: float = cs.g_dof,
    lv20_key: str = "U1",
    lv20_params: Optional[Dict[str, float]] = None,
) -> Tuple[Callable[[float | np.ndarray], float | np.ndarray], GWFieldLV20]:
    """
    Convenience wrapper that returns (h2_callable, GWFieldLV20_builder).

    Parameters mirror GWFieldLV20.from_params plus the constructor arguments.
    """
    builder = GWFieldLV20(g_star=g_star, lv20_params=lv20_params, lv20_key=lv20_key)
    h2      = builder.from_params(Tn, alpha, beta_over_H, Tstar_mode=Tstar_mode)
    return h2, builder


# ------------------------------------------------------------------ #
# Minimal usage example                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Tn, alpha, betaH = 1.0e-2, 1.0e5, 40.0
    f   = np.logspace(-14, -6, 300)

    h2_reh, gwf = from_params(Tn, alpha, betaH, Tstar_mode="reh", lv20_key="U1")
    h2_nuc, _   = from_params(Tn, alpha, betaH, Tstar_mode="nuc", lv20_key="U1")

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=120)
    ax.plot(f, h2_reh(f), lw=2.2, label=r"LV20 — $T_*=T_{\rm reh}$")
    ax.plot(f, h2_nuc(f), lw=2.0, ls="--", label=r"LV20 — $T_*=T_n$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$f\,[\mathrm{Hz}]$")
    ax.set_ylabel(r"$h^2\Omega_{\rm GW}(f)$")
    ax.set_xlim(1e-11, 1e-6)
    ax.set_ylim(1e-14, 1e-6)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()
