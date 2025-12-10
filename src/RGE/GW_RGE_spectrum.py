# GW_RGE_spectrum.py
# Minimal GW spectra builder that takes T, alpha, beta/H (and kappa_col) as inputs.
# No microphysics nor Veff calls unless you explicitly ask for it.

import numpy as np
from math import pi
from typing import Callable, Tuple
from scipy.integrate import quad

import constants as cs  # expects: g_dof, R, Hhund, fmin, fmax, T_LISA, SNRthr

# ---------- peak frequencies & shapes ----------
def h_star(T):                       return 1.65e-5 * (T / 100.) * (cs.g_dof / 100.)**(1/6)
def f_sw(beta_over_H, hstar):        return 0.54 * hstar * beta_over_H
def f_col(beta_over_H, hstar):       return 0.17 * hstar * beta_over_H
def f_turb(beta_over_H, hstar):      return 1.75 * hstar * beta_over_H
def f_PT(beta_over_H, hstar):        return (0.8 / (2 * np.pi)) * hstar * beta_over_H
def f_star_only(hstar):              return 1/(2*np.pi) * hstar

def S_PT(f, fPT):
    x = f / fPT
    return 3 * x**0.9 / (2.1 + 0.9 * x**3)

def S_H(f, fstar):
    x = f / fstar
    return x**2.1 / (1 + x**2.1)

def kappa_sw(alpha):
    return alpha / (0.73 + 0.083 * np.sqrt(alpha) + alpha)

def Htau(alpha, beta_over_H):
    term = (8 * np.pi)**(1/3) / beta_over_H
    denom = ((3 / 4) * (kappa_sw(alpha) * alpha) / (1 + alpha))**0.5
    if denom <= 0:
        return 1.0
    return min(1.0, term / denom)



class GWFromParams:
    """
    Usage:
      gw = GWFromParams()
      (h2_tot, h2_col, h2_sw, h2_turb, h2_pT) = gw.spectra(Tn, alpha, beta_over_H, kappa_col=0.0)

      # SNR:
      SNR = gw.snr_lisa(h2_tot)

      # Threshold for power-law f^p at f0:
      thr = gw.h2Omega_thr(p=2/3, f0=1e-3)

    Notes:
      - No Veff calls in this class.
      - If you want kappa_col from microphysics, compute it outside and pass it in.
    """
    def __init__(self, g_star: float = cs.g_dof):
        self.g_star = g_star

    def spectra(self, T: float, alpha: float, beta_over_H: float, kappa_col: float = 0.0
               ) -> Tuple[Callable[[float], float], ...]:
        """Return callables (h2_Omega_total, h2_Omega_col, h2_Omega_sw, h2_Omega_turb, h2_Omega_pT)."""
        Rval = cs.R
        hstar = h_star(T)

        fcol  = f_col(beta_over_H, hstar)
        fsw   = f_sw(beta_over_H, hstar)
        fturb = f_turb(beta_over_H, hstar)
        fstar = f_star_only(hstar)
        fPT   = f_PT(beta_over_H, hstar)

        k_sw = kappa_sw(alpha * (1 - kappa_col)) * (1 - kappa_col)
        Ht   = Htau(alpha, beta_over_H)

        def h2_Omega_col(f):
            x = f / fcol
            return (0.028 * Rval * beta_over_H**-2 *
                    ((kappa_col * alpha)/(1+alpha))**2 *
                    x**3 * (4.51 / (1.51 + 3*x**2.07))**2.18)

        def h2_Omega_sw(f):
            x = f / fsw
            return (0.29 * Rval * beta_over_H**-1 * Ht *
                    ((k_sw * alpha)/(1+alpha))**2 *
                    x**3 * (7/(3 + 4*x**2))**(7/2))

        def h2_Omega_turb(f):
            x = f / fturb
            return (20 * Rval * beta_over_H**-1 * (1 - Ht) *
                    ((k_sw * alpha)/(1+alpha))**(3/2) *
                    x**3 * (1/(1+x))**(11/3) * (1/(1 + 8*np.pi*f/hstar)))

        def h2_Omega_pT(f):
            return (1e-6 / (self.g_star/100)**(1/3) *
                    beta_over_H**-2 * (alpha/(1+alpha))**2 *
                    S_PT(f, fPT) * S_H(f, fstar))

        def h2_Omega_total(f):
            if kappa_col >= 1.0:
                return h2_Omega_pT(f)
            elif kappa_col <= 0.0:
                return h2_Omega_sw(f) + h2_Omega_turb(f)
            else:
                return h2_Omega_col(f) + h2_Omega_sw(f) + h2_Omega_turb(f)

        return h2_Omega_total, h2_Omega_col, h2_Omega_sw, h2_Omega_turb, h2_Omega_pT


