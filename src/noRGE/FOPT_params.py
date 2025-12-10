import Miselanie.constants as cs
import NoRGEVeff as fs
import BounceSolFull as bs
import BounceSolHighT as bh
import numpy as np
#from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.integrate import quad
from numpy.polynomial.polynomial import Polynomial
import scipy.optimize as optimize

#############################################################Martin's############################################################################################################

def SE(T, gD, v):
    return bs.SE(T, gD, v)[0]

def SE_HT(T, gD, v):
    return bh.SE(T, gD, v)[0]
	
def Gamma(T, gD, v, is_HT=False):
    S = SE_HT(T, gD, v) if is_HT else SE(T, gD, v)
    return T**4 * (S / (2 * np.pi * T))**(3 / 2) * np.exp(-S/T)




def critTemp(gD, v, is_HT=False):
    """
    Encuentra la temperatura crítica: cuando Veff(phi_min, T) = 0.
    """
    def func(T):
        # Elegir phi_min y Veff según el caso
        if is_HT:
            phi_min_T = bh.phi_min_Veff0(T, gD, v)
            Veff = fs.Veff_HighT(phi_min_T, gD, T, v)
        else:
            phi_min_T = bs.phi_min(T, gD, v)
            Veff = fs.Veff(phi_min_T, gD, T, v)
        return np.abs(Veff)  

    result = optimize.minimize_scalar(func, bounds=(0.01 * v, 0.5 * v), method='bounded')
    return result.x


# Constants defined once outside for reusability
G = cs.GF  # Gravitational constant
gstar = cs.g_dof  # Degrees of freedom

def hubble(TTT, gD, v):
    """Calculate the Hubble parameter directly as a function of TTT, gD, and v."""
    rhoH = -fs.Veff0(bs.phi_min_Veff0(gD, v), gD, v) + (np.pi**2 / 30) * gstar * TTT**4  # Energy density
    return np.sqrt((8 * np.pi * G / 3) * rhoH)  # Hubble parameter

def outer_integrand(T, TT, gD, v, is_HT=False):
    """Calculate the outer integrand for Pf using a nested integral over TTT."""
    integrand, _ = quad(lambda TTT: 1 / hubble(TTT, gD, v), T, TT, limit=5000, epsrel=1.49e-9, epsabs=1.49e-9)
    gamma_func = Gamma(TT, gD, v, is_HT)
    hubble_param_outer = hubble(TT, gD, v)
    return (gamma_func / (TT**4 * hubble_param_outer)) * integrand**3

def Pf(T, gD, v, is_HT=False):
    TC = critTemp(gD, v, is_HT)  # Asignamos el valor de critTemp a TC
    integral, _ = quad(lambda TT: outer_integrand(T, TT, gD, v, is_HT), T, TC, limit=5000, epsrel=1.49e-9, epsabs=1.49e-9)
    return np.exp((-4 * np.pi / 3) * integral)



def get_bounds(gD):
    bounds = [
        (1e-6, 1e-4),       # gD < 0.51
        (2e-6, 2e-4),       # 0.51–0.52
        (5e-6, 5e-4),       # 0.52–0.54
        (1e-5, 1e-3),       # 0.54–0.56
        (2e-5, 2e-3),       # 0.56–0.58
        (5e-5, 1e-2),       # 0.58–0.6
        (1e-4, 1e-2),       # 0.6–0.62
        (1e-3, 1e-2),       # 0.62–0.64
        (1e-2, 1e-1),       # 0.64–0.68
        (5e-3, 6e-2),       # 0.68–0.71
        (5e-3, 7e-2),       # 0.71–0.74
        (1e-3, 2e-2),       # 0.74–0.77
        (1e-3, 3e-1),       # 0.77–0.8
        (1.e-2, 5e-1)        # 0.8–1.0
    ]

    thresholds = [
        0.51, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.65,
        0.68, 0.71, 0.74, 0.77, 0.8, 0.84, 0.88, 0.92
    ]

    idx = np.searchsorted(thresholds, gD)
    return bounds[idx]




def perTemp(gD, v):
    """ Encontrar la temperatura para el caso estándar """
    def func(T):
        return np.abs(Pf(T, gD, v) - 0.71)

    bounds = get_bounds(gD)
    result = optimize.minimize_scalar(func, bounds=bounds, method='bounded')
    return result.x

def perTemp_HT(gD, v):
    """ Encontrar la temperatura para el caso HT """
    def func(T):
        return np.abs(Pf(T, gD, v, is_HT=True) - 0.71)

    bounds = get_bounds(gD)
    result = optimize.minimize_scalar(func, bounds=(1e-6,0.3), method='bounded')
    return result.x

def Next(T, gD, v):
    rhoH = -fs.Veff0(bs.phi_min_Veff0(gD, v), gD, v) + np.pi**2 / 30 * gstar * T**4
    hubble_param = np.sqrt((8 * np.pi * G / 3) * rhoH)
    result = Gamma(T, gD, v) / (hubble_param**4)
    return result

def Next_HT(T, gD, v):
    rhoH = -fs.Veff0(bs.phi_min_Veff0(gD, v), gD, v) + np.pi**2 / 30 * gstar * T**4
    hubble_param = np.sqrt((8 * np.pi * G / 3) * rhoH)
    result = Gamma(T, gD, v, is_HT=True) / (hubble_param**4)
    return result







def nucTemp(gD, v):
    """ Encontrar la temperatura para el caso HT """
    def func(T):
        return np.abs(Next(T, gD, v) - 1)

    bounds = get_bounds(gD)  # Define los límites para T
    result = optimize.minimize_scalar(func, bounds=bounds, method='bounded')
    return result.x



def nucTemp_HT(gD, v):
    """ Encontrar la temperatura para el caso HT """
    def func(T):
        return np.abs(Next_HT(T, gD, v) - 1)

    bounds = get_bounds(gD)  # Define los límites para T
    result = optimize.minimize_scalar(func, bounds=bounds, method='bounded')
    return result.x





def alpha_High(T, gD, v):
    rho_R = np.pi**2 * cs.g_dof * T**4 / 30
    phi_min = bh.phi_min_Veff0(gD, v,)
    return -fs.Veff0(phi_min, gD, v) / rho_R

def alpha_Full(T, gD, v):
    rho_R = np.pi**2 * cs.g_dof * T**4 / 30
    phi_min = bs.phi_min(T, gD, v)
    return -fs.Veff(phi_min, gD, T, v) / rho_R





#######High###########

# def beta_High(T_star, gD, v):
#     try:
#         h = 1e-6 #max(adaptive_h(gD), 1e-4)

#         if T_star - 2*h <= 0:
#             print(f"[WARNING] Invalid T range: T_star={T_star}, h={h}")
#             return np.nan

#         f = lambda T: SE(T, gD, v)

#         S_pp = f(T_star + 2*h)
#         S_p  = f(T_star + h)
#         S_m  = f(T_star - h)
#         S_mm = f(T_star - 2*h)

#         if any(np.isnan(x) or np.isinf(x) for x in [S_pp, S_p, S_m, S_mm]):
#             print(f"[WARNING] SE values invalid near T = {T_star:.3e}, gD = {gD}")
#             return np.nan

#         dS_dT = (-S_pp + 8*S_p - 8*S_m + S_mm) / (12 * h)
#         beta_val = T_star * dS_dT

#         if beta_val <= 0:
#             print(f"[WARNING] Non-physical beta: {beta_val:.2f}")
#             return np.nan

#         if beta_val > 1e6:
#             print(f"[WARNING] Unusually large beta: {beta_val:.2e}")
#             return np.nan

#         return beta_val

#     except Exception as e:
#         print(f"[ERROR] beta_Full failed: {e}")
#         return np.nan





# def beta_High(T_star, gD, v):
#     """
#     Compute β/H using a 5-point stencil with a fixed step size h depending on gD.
#     """
#     try:
#         f = lambda T: SE_HT(T, gD, v)

#         # Choose h based on gD to handle numerical sensitivity at low gD
#         if gD < 0.51:
#             h = 7.8e-8 # very fine resolution for sensitive region 
#         elif gD < 0.7:
#             h = 5e-6 
#         elif gD <= 0.98:
#             h = 1e-4
#         else:
#             h = 5e-4

#         dS_dT =  (f(T_star + h) - f(T_star - h)) / (2 * h) #(-f(T_star + 2*h) + 8*f(T_star + h) - 8*f(T_star - h) + f(T_star - 2*h)) / (12 * h)

#         if np.isnan(dS_dT) or np.isinf(dS_dT):
#             return np.nan

#         return T_star * dS_dT

#     except:
#         return np.nan


# def adaptive_h(gD):
#     gD_clipped = np.clip(gD, 0.5, 1.0)
#     # h = 10^(-a), donde a va de 8 (cuando gD = 0.5) a 4 (cuando gD = 1.0)
#     exponent = 8 + 4 * (gD_clipped - 0.5) / 0.5  # lineal entre 0.5 y 1.0
#     return 10 ** (-exponent)




def beta_Full(T_star, gD, v):
    try:
        h = 5e-5 #max(adaptive_h(gD), 1e-4)

        if T_star - 2*h <= 0:
            print(f"[WARNING] Invalid T range: T_star={T_star}, h={h}")
            return np.nan

        f = lambda T: SE_HT(T, gD, v)

        S_pp = f(T_star + 2*h)
        S_p  = f(T_star + h)
        S_m  = f(T_star - h)
        S_mm = f(T_star - 2*h)

        if any(np.isnan(x) or np.isinf(x) for x in [S_pp, S_p, S_m, S_mm]):
            print(f"[WARNING] SE values invalid near T = {T_star:.3e}, gD = {gD}")
            return np.nan

        dS_dT = (S_p - S_m) /(2 * h) #(-S_pp + 8*S_p - 8*S_m + S_mm) / (12 * h)
        beta_val = T_star * dS_dT

        if beta_val <= 0:
            print(f"[WARNING] Non-physical beta: {beta_val:.2f}")
            return np.nan

        if beta_val > 1e6:
            print(f"[WARNING] Unusually large beta: {beta_val:.2e}")
            return np.nan

        return beta_val

    except Exception as e:
        print(f"[ERROR] beta_Full failed: {e}")
        return np.nan
    


def beta_High(T_star, gD, v):
    """
    Calcula β/H = dS/dT - S/T usando solo 3 evaluaciones de SE (evita recalcular).
    """
    try:

        h = 2.01706e-07


        # Evaluar SE una sola vez por punto
        S_plus  = SE_HT(T_star + h, gD, v)
        S_minus = SE_HT(T_star - h, gD, v)
        S_0     = SE_HT(T_star, gD, v)

        # Derivada y fórmula analítica
        dS_dT = (S_plus - S_minus) / (2 * h)
        beta_H = dS_dT - S_0 / T_star

        # Validación
        if any(np.isnan([dS_dT, S_0, beta_H])) or any(np.isinf([dS_dT, S_0, beta_H])):
            return np.nan

        return beta_H

    except:
        return np.nan


def beta_Full(T_star, gD, v):
    """
    Calcula β/H = dS/dT - S/T para el caso Full, usando solo 3 evaluaciones de SE.
    """
    try:
        # Paso adaptativo según gD
        if gD < 0.51:
            h = 8.e-8
        elif gD < 0.7:
            h = 5e-6
        elif gD <= 0.98:
            h = 1e-4
        else:
            h = 5e-4

        # Evaluar SE solo una vez por punto
        S_plus  = SE(T_star + h, gD, v)
        S_minus = SE(T_star - h, gD, v)
        S_0     = SE(T_star, gD, v)

        # Derivada central y fórmula analítica
        dS_dT = (S_plus - S_minus) / (2 * h)
        beta_H = dS_dT - S_0 / T_star

        # Validación de resultado
        if any(np.isnan([dS_dT, S_0, beta_H])) or any(np.isinf([dS_dT, S_0, beta_H])):
            return np.nan

        return beta_H

    except:
        return np.nan
