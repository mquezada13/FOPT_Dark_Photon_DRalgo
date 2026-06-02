# GW_bubbles_LVV22.py
"""
Gravitational-wave spectrum from bubble collisions for strongly supercooled FOPTs.

Based on Lewicki, Vaskonen & von Harling (arXiv:2208.11697) — "LVV22".

The module is self-contained: it requires only (Tn, alpha, beta/H) as inputs
and returns a callable h^2 Omega_GW(f) evaluated today.

Formulas used
-------------
- Spectral shape :  Table I + Eqs. (24), (25)
- Redshift to today : Eqs. (27), (28)
- Bubble-wall efficiency kappa(R_eff) : Eq. (23)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict

import constants as cs


# ------------------------------------------------------------------ #
# Cosmological helper                                                  #
# ------------------------------------------------------------------ #

def _h_star(T: float, gstar: float) -> float:
    """
    Characteristic frequency h_*(T) [Hz] evaluated at the transition temperature.

    Eq. (28) of LVV22.
    """
    return 1.65e-5 * (T / 100.0) * (gstar / 100.0) ** (1.0 / 6.0)


# ------------------------------------------------------------------ #
# Spectral presets (Table I of LVV22)                                  #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class BubbleParams:
    """
    Broken power-law parameters for a given source morphology.

    Attributes
    ----------
    A, a, b, c           : shape parameters for Eq. (24).
    fbeta_over_beta      : f_peak / beta  (already divided by 2*pi).
    beta_Reff            : beta * R_eff   (Table I, last row).
    """
    A: float
    a: float
    b: float
    c: float
    fbeta_over_beta: float
    beta_Reff: float


BUBBLE_PRESETS: Dict[str, BubbleParams] = {
    # T_mu_nu ~ R^{-2}  (global symmetry)
    "bubbles_Rm2": BubbleParams(
        A=5.93e-2, a=1.03, b=1.84, c=1.45,
        fbeta_over_beta=0.64 / (2 * np.pi),
        beta_Reff=5.07,
    ),
    # T_mu_nu ~ R^{-3}  (gauge symmetry — typical for dark photon models)
    "bubbles_Rm3": BubbleParams(
        A=5.13e-2, a=2.41, b=2.42, c=4.08,
        fbeta_over_beta=0.77 / (2 * np.pi),
        beta_Reff=4.81,
    ),
}


# ------------------------------------------------------------------ #
# Spectral shape and efficiency                                         #
# ------------------------------------------------------------------ #

def _S_shape(f_over_beta: np.ndarray | float, P: BubbleParams) -> np.ndarray:
    """
    Broken power-law spectral shape as a function of f / f_beta.

    Eq. (24) of LVV22.
    """
    x   = np.asarray(f_over_beta, dtype=float) / P.fbeta_over_beta
    num = P.A * (P.a + P.b) ** P.c
    den = (P.b * x ** (-P.a / P.c) + P.a * x ** (P.b / P.c)) ** P.c
    return num / den


def _kappa_at_Reff(P: BubbleParams, beta_R_eq: float = 50.0) -> float:
    """
    Bubble-wall efficiency kappa(R_eff) from Eq. (23) of LVV22.

    kappa(R) ~ 1 / (1 + R/R_eq), evaluated at R = R_eff.
    """
    return 1.0 / (1.0 + P.beta_Reff / float(beta_R_eq))


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def spectrum_bubbles(
    Tn: float,
    alpha: float,
    beta_over_H: float,
    preset_key: str = "bubbles_Rm3",
    beta_R_eq: float = 5.0,
    g_star: float | None = None,
) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """
    Build the callable h^2 Omega_GW(f) for bubble-collision GWs (LVV22).

    Parameters
    ----------
    Tn          : Nucleation temperature used as T_* [GeV].
    alpha       : Transition strength parameter.
    beta_over_H : Inverse transition duration beta / H.
    preset_key  : Spectral preset — "bubbles_Rm3" (gauge, default) or
                  "bubbles_Rm2" (global).
    beta_R_eq   : Controls kappa(R_eff) via Eq. (23); typical value ~ 5.
    g_star      : Effective d.o.f. at T_*.  If None, uses cs.g_dof.

    Returns
    -------
    h2Omega_today : callable  f [Hz] -> h^2 Omega_GW(f) [dimensionless]
    """
    if preset_key not in BUBBLE_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset_key}'.  Available: {list(BUBBLE_PRESETS.keys())}"
        )

    g   = float(g_star) if g_star is not None else float(cs.g_dof)
    P   = BUBBLE_PRESETS[preset_key]

    hstar       = _h_star(Tn, g)
    kappa_eff   = _kappa_at_Reff(P, beta_R_eq=beta_R_eq)
    pref_src    = beta_over_H**-2 * ((kappa_eff * alpha) / (1.0 + alpha)) ** 2
    omega_fac   = 1.67e-5 * (100.0 / g) ** (1.0 / 3.0)

    def h2Omega_today(f_Hz: float | np.ndarray) -> float | np.ndarray:
        f_over_beta = np.asarray(f_Hz, dtype=float) / (hstar * beta_over_H)
        return omega_fac * pref_src * _S_shape(f_over_beta, P)

    return h2Omega_today


def f_peak_today(
    Tn: float,
    beta_over_H: float,
    preset_key: str = "bubbles_Rm3",
    g_star: float | None = None,
) -> float:
    """
    Peak frequency today f_{p,0} = h_* (f_p/beta) (beta/H) [Hz].

    Useful for marking the peak on GW plots.
    """
    if preset_key not in BUBBLE_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset_key}'.  Available: {list(BUBBLE_PRESETS.keys())}"
        )
    g   = float(g_star) if g_star is not None else float(cs.g_dof)
    P   = BUBBLE_PRESETS[preset_key]
    return _h_star(Tn, g) * P.fbeta_over_beta * beta_over_H


# ------------------------------------------------------------------ #
# Minimal usage example                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Tn, alpha, betaH = 1.0e-2, 10.0, 40.0
    h2   = spectrum_bubbles(Tn, alpha, betaH, preset_key="bubbles_Rm3", beta_R_eq=5.0)
    fp   = f_peak_today(Tn, betaH, "bubbles_Rm3")
    f    = np.logspace(-14, -6, 400)

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=120)
    ax.plot(f, h2(f), lw=2.0, label=r"LVV22 bubbles $R^{-3}$")
    ax.axvline(fp, ls="--", color="k", alpha=0.6, label=r"$f_{\rm peak}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$f\,[\mathrm{Hz}]$")
    ax.set_ylabel(r"$h^2\Omega_{\rm GW}(f)$")
    ax.set_xlim(1e-11, 1e-6)
    ax.set_ylim(1e-14, 1e-6)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()
