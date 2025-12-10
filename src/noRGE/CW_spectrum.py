import numpy as np
from scipy.integrate import quad

import Miselanie.constants as cs
from FOPT_params import alpha_High, beta_High, nucTemp_HT, alpha_Full, beta_Full, nucTemp, hubble

# === Some constants ===
g_star = cs.g_dof
R = cs.R
H100 = cs.Hhund

import BounceSolFull as bs
import BounceSolHighT as bh
import RGE_code.src.VeffFunc_RGE as fs



import RGE_code.src.VeffFunc_RGE as fs
import BounceSolFull as bs
import BounceSolHighT as bh


# ΔV = -Vfalse evaluated at φ_true from Veff0, but Veff(T)
def deltaV(gD, phi, T, v, is_HT=False):
    Veff = fs.Veff_HighT if is_HT else fs.Veff
    return -Veff(phi, gD, T, v)

# φ_min at T = 0
def phi_min_T0(gD, v, is_HT=False):
    return bh.phi_min_Veff0(gD, v) if is_HT else bs.phi_min_Veff0(gD, v)

# Wall tension σ = ∫ sqrt(2 ΔV(ϕ)) dϕ from 0 to φ_min
def sigma(gD, T, v, is_HT=False):
    phi_min = phi_min_T0(gD, v, is_HT)
    def integrand(phi):
        val = deltaV(gD, phi, T, v, is_HT)
        return np.sqrt(2 * val) if val > 0 else 0
    result, _ = quad(integrand, 0, phi_min, epsrel=1e-6)
    return result

# β(T)
def beta(gD, T, v, is_HT=False):
    return beta_High(T, gD, v) if is_HT else beta_Full(T, gD, v)

# R_coll = π^{1/3} / β / H
def Rcoll(gD, T, v, is_HT=False):
    return np.pi**(1/3) / beta(gD, T, v, is_HT) / hubble(gD, T, v)

# R_crit = 2σ / ΔV
def Rcrit(gD, T, v, is_HT=False):
    phi_min = phi_min_T0(gD, v, is_HT)
    return 2 * sigma(gD, T, v, is_HT) / deltaV(gD, phi_min, T, v, is_HT)

# γ_max = 2 R_coll / (3 R_crit)
def gammaMax(gD, T, v, is_HT=False):
    return 2 * Rcoll(gD, T, v, is_HT) / (3 * Rcrit(gD, T, v, is_HT))

# γ_LL = ΔV / (g^3 v T^3 log(v/T))
def gammaLL(gD, T, v, c0=0.5):
    phi_min = phi_min_T0(gD, v)
    dV = deltaV(gD, phi_min, T, v)
    return dV / (c0 * gD**3 * v * T**3 * np.log(v / T))

# κ_coll = min(γ_max, γ_LL) / γ_max
def kappaColl(gD, T, v, is_HT=False, c0=0.5):
    gamma_1 = gammaMax(gD, T, v, is_HT)
    gamma_2 = gammaLL(gD, T, v, c0)
    if gamma_1 == 0:
        return 0.0
    return min(gamma_1, gamma_2) / gamma_1










# === Hubble frequency today ===
def h_star(T):
    return 1.65e-5 * (T / 100.) * (cs.g_dof / 100.)**(1/6)

def f_sw(beta_over_H, hstar_val):
    return 0.54 * hstar_val * beta_over_H

def f_col(beta_over_H, hstar_val):
    return 0.17 * hstar_val * beta_over_H

def f_turb(beta_over_H, hstar_val):
    return 1.75 * hstar_val * beta_over_H

def f_PT(beta_over_H, hstar_val):
    return (0.8 / (2 * np.pi)) * hstar_val * beta_over_H

def f_star(hstar_val):
    return 1/2/np.pi * hstar_val

def S_PT(f, fPT_val):
    x = f / fPT_val
    return 3 * x**0.9 / (2.1 + 0.9 * x**3)

def S_H(f, fstar_val):
    x = f/fstar_val
    return x**2.1 / (1 + x**2.1)

def kappa_sw(alpha):
    return alpha / (0.73 + 0.083 * np.sqrt(alpha) + alpha)

def Htau(alpha, beta_over_H):
    term = (8 * np.pi)**(1/3) / beta_over_H
    denom = ((3 / 4) * (kappa_sw(alpha) * alpha) / (1 + alpha))**0.5
    return min(1.0, term / denom)


# === Full spectrum builder ===
def create_spectra(T, alpha, beta_over_H, kappa_col, v, g_star=cs.g_dof):
    Rval = cs.R  # e.g., 1.0
    hstar_val = h_star(T)  # Hz


    # Peak frequencies
    fcol_val = f_col(beta_over_H, hstar_val)
    fsw_val = f_sw(beta_over_H, hstar_val)
    fturb_val = f_turb(beta_over_H, hstar_val)
    fstar_val = f_star(hstar_val)
    fPT_val = f_PT(beta_over_H, hstar_val)

    # Efficiency
    kappa_sw_val = kappa_sw(alpha * (1 - kappa_col)) * (1 - kappa_col)
    Htau_val = Htau(alpha, beta_over_H)

    # === Individual contributions ===
    def h2_Omega_col(f):
        x = f / fcol_val
        return (
            0.028 * Rval * beta_over_H**-2 *
            ((kappa_col * alpha) / (1 + alpha))**2 *
            x**3 * (4.51 / (1.51 + 3 * x**2.07))**2.18
        )

    def h2_Omega_sw(f):
        x = f / fsw_val
        return (
            0.29 * Rval * beta_over_H**-1 * Htau_val *
            ((kappa_sw_val * alpha) / (1 + alpha))**2 *
            x**3 * (7 / (3 + 4 * x**2))**(7 / 2)
        )

    def h2_Omega_turb(f):
        x = f / fturb_val
        return (
            20 * Rval * beta_over_H**-1 * (1 - Htau_val) *
            ((kappa_sw_val * alpha) / (1 + alpha))**(3 / 2) *
            x**3 * (1 / (1 + x))**(11 / 3) * (1 / (1 + 8 * np.pi * f / hstar_val))
        )

    def h2_Omega_pT(f):
        return (
            1e-6 /(g_star/100)**(1/3)*
            (beta_over_H)**-2 *
            (alpha / (1 + alpha))**2 *
            S_PT(f, fPT_val) *
            S_H(f, fstar_val)
        )

    def h2_Omega_total(f):
        if kappa_col == 1.0:
            # Colisión de burbujas (vacuum domination): solo Omega_PT
            return h2_Omega_pT(f)
        elif kappa_col == 0.0:
            # Plasma domination: solo sonido + turbulencia
            return h2_Omega_sw(f) + h2_Omega_turb(f)
        else:
            # Régimen mixto: colisión (envelope) + plasma
            return h2_Omega_col(f) + h2_Omega_sw(f) + h2_Omega_turb(f)

    return h2_Omega_total, h2_Omega_col, h2_Omega_sw, h2_Omega_turb, h2_Omega_pT













# === Noise density parameter ===
def h2_Omega_LISA(f):  # H100 en GeV, por defecto H_0 = 67.36 km/s/Mpc
    term1 = ((5.76e-48) / (2 * np.pi * f)**4) * (1 + (0.0004 / f)**2)
    term2 = 3.6e-41
    term3 = (1 + (f / 0.025)**2)
    prefactor = (4 * np.pi**2) / (3 * H100**2) * f**3 * (10 / 3)
    return prefactor * (term1 + term2) * term3

# === SNR LISA ===




def snr_lisa(h2_Omega_GW_func, fmin=cs.fmin, fmax=cs.fmax, TLISA=cs.T_LISA):  # TLISA = 4 años en segundos
    """
    Compute the Signal-to-Noise Ratio (SNR) for LISA given a GW spectrum.
    
    Parameters:
    - h2_Omega_GW_func: function h^2 Omega_GW(f) to evaluate
    - fmin, fmax: frequency range in Hz
    - TLISA: mission duration in seconds (default: 4 years)

    Returns:
    - SNR (float)
    """
    integrand = lambda f: (h2_Omega_GW_func(f) / h2_Omega_LISA(f))**2
    integral, _ = quad(integrand, fmin, fmax, limit=1000)
    return np.sqrt(TLISA * integral)



def h2Omega_thr(p):
    f0 = 1e-3 
    func = lambda f: (f / f0)**p
    return cs.SNRthr / snr_lisa(func)


def h2Omega_PL(f, h2Omega_p_list, f0=1e-3):
    return max(thr * (f / f0)**p for p, thr in h2Omega_p_list)