# bouncesol.py

import Miselanie.constants as cs
import NoRGEVeff as fs
import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, root_scalar, minimize, root, approx_fprime
from scipy.integrate import quad
import warnings
# Handling complex numbers
Re1 = complex(1., 0.)
Imi = complex(0., 1.)

#==================================================================================================================================================================
# Define a function to find the minimum of Veff0 (Effective potential at T=0) within a given interval, now dependent on gD and v
def phi_min_Veff0(gD, v, bound_min=(0.001, 1e7)):
    return optimize.minimize_scalar(lambda S: fs.Veff0(S, gD, v), bounds=bound_min, method='bounded').x


# Update phi_interval and phi_root_interval to use gD and v
def get_intervals(gD, v):
    phi_min_val = phi_min_Veff0(gD, v)
    upper_bound = min(3, phi_min_val)
    return (1e-7, upper_bound ), (1e-6, upper_bound), (1e-7, 1.0) #(0.01, upper_bound), (0.001, upper_bound), (1e-5, 1.0)

#==================================================================================================================================================================
# def phi_max(T, gD, v, scale_factor=1e13, verbose=False):
#     Veff_scaled = lambda S: np.real(scale_factor * fs.Veff(S, gD, T, v))

#     # Escaneo dinámico adaptado
#     S_start = 1e-5 * T
#     S_stop = 0.05 + 2 * gD  # Rango adaptado: crece con gD
#     S_vals = np.linspace(S_start, S_stop, 5000)
#     V_vals = np.array([Veff_scaled(S) for S in S_vals])

#     i_max = np.argmax(V_vals)
#     phiT = S_vals[i_max]

#     if verbose:
#         print(f"[phi_max] max found at S = {phiT:.4e} with Veff = {fs.Veff(phiT, gD, T, v):.4e}")

#     return phiT
def phi_max(T, gD, v):
    phi_interval, _, _ = get_intervals(gD,v)  # Rango bien definido
    def func(S):
        return -np.real(fs.Veff(S, gD, T, v))
    result = optimize.minimize_scalar(func, bounds=phi_interval, method='bounded')
    return result.x



# Now using get_intervals to obtain phi_interval and phi_root_interval
def phi_min(T, gD, v):
    # Retrieve the intervals based on gD and v
    phi_interval, _, _ = get_intervals(gD, v)
    
    def func(S):
        return fs.Veff(S, gD, T, v)  # Pass gD, T, and v to Veff
    
    result = optimize.minimize_scalar(func, bounds=phi_interval, method='bounded')
    return result.x




def phi_root(T, gD, v, scale_factor=1e0, verbose=False):
    """
    Robust automatic root finder for Veff_HighT, works even when the barrier is shallow/narrow.
    """
    phiT = phi_max(T, gD, v)
    Veff_scaled = lambda S: np.real(scale_factor * fs.Veff(S, gD, T, v))

    # Step 1: scan with very fine steps beyond the barrier
    S_vals = np.linspace(phiT+1e-6 * T, 3.0, 50000)
    V_vals = np.array([Veff_scaled(S) for S in S_vals])

    # Step 2: check sign change
    sign_changes = np.where(np.sign(V_vals[:-1]) != np.sign(V_vals[1:]))[0]
    if len(sign_changes) == 0:
        raise ValueError("No root found beyond the barrier.")

    # Step 3: pick first valid bracket and refine root
    i = sign_changes[0]
    S_low, S_high = S_vals[i], S_vals[i + 1]

    if verbose:
        print(f"[DEBUG] Root bracket found in [{S_low:.4e}, {S_high:.4e}]")
        print(f"[DEBUG] V(S_low) = {Veff_scaled(S_low):.4e}, V(S_high) = {Veff_scaled(S_high):.4e}")

    result = optimize.root_scalar(Veff_scaled, bracket=(S_low, S_high), xtol=1e-14)
    return result.root

#==================================================================================================================================================================
# ConstructVt should also pass gD and v to Veff and its derivative functions
def ConstructVt(T, phi0, gD, v):
    phiT = phi_max(T, gD, v)  # Now pass gD and v to phi_max
    V0 = fs.Veff(phi0, gD, T, v)  # Pass gD, T, and v to Veff

    # Compute the derivative using numerical differentiation
    def dVeff_ds(s, T):
        epsilon = 1e-9
        return (fs.Veff(s + epsilon, gD, T, v) - fs.Veff(s - epsilon, gD, T, v)) / (2 * epsilon)  # Pass gD, T, and v

    dV0 = dVeff_ds(phi0, T)

    # VT is defined in terms of phiT
    VT = fs.Veff(phiT, gD, T, v)  # Pass gD, T, and v to Veff

    # Rest of the function remains the same
    d = 3
    a1 = V0 / phi0
    a2 = ((d - 1.) * phi0 * dV0 - d * V0) / (d * phi0**2)
    a3 = ((d - 1.) * phi0 * dV0 - 2. * d * V0) / (d * phi0**3)

    dVt3T = a1 + a2 * (2. * phiT - phi0) + a3 * (3. * phiT - phi0) * (phiT - phi0)
    d2Vt3T = 2. * a2 + 2. * a3 * (3. * phiT - 2. * phi0)

    phi0T = phi0 - phiT
    c = 4 * phiT**2 * phi0T**2 * phi0**2

    Vt3T = (V0 * phiT / phi0 +
            (2 * phi0 * dV0 - 3 * V0) / (3 * phi0**2) * phiT * (phiT - phi0) +
            (2 * phi0 * dV0 - 6 * V0) / (3 * phi0**3) * phiT * (phiT - phi0)**2)

    a0T = (-4 * (VT - Vt3T) * (phi0**2 - 6 * phiT * phi0T) -
           6 * phiT * (phi0T - phiT) * phi0T * dVt3T +
           2 * phiT**2 * phi0T**2 * d2Vt3T)

    Ut3T = 3 * dVt3T**2 + 4 * (VT - Vt3T) * d2Vt3T

    if a0T**2 - c * Ut3T > 0:
        a4 = 1 / c * (a0T - np.sqrt(a0T**2 - c * Ut3T))
    else:
        a4 = 0

    def Vt(phi):
        return (a1 * phi +
                a2 * phi * (phi - phi0) +
                a3 * phi * (phi - phi0)**2 +
                a4 * phi**2 * (phi - phi0)**2)

    return Vt

#==================================================================================================================================================================
# Euclidean action computation also requires passing gD and v
# def SE0(T, phi0, gD, v):
#     vt = ConstructVt(T, phi0, gD, v)  # Pass gD and v to ConstructVt

#     def integrand(S, T):
#         dvt_ds = optimize.approx_fprime([S], vt, 1e-9)
#         v_eff_diff = Re1 * (fs.Veff(S, gD, T, v) - vt(S))  # Pass gD, T, and v to Veff
#         if np.abs(dvt_ds) < 1e-10:
#             dvt_ds = 1e-10
#         return np.real(v_eff_diff**1.5 / dvt_ds**2)

#     result = quad(lambda S: integrand(S, T), 0, phi0, limit=20000, epsrel=1.49e-9, epsabs=1.49e-9)[0]
#     return (32 * np.pi * np.sqrt(2) / 3) * np.real(result)



# def SE0(T, phi0, gD, v):
#     """
#     Compute the Euclidean action S_E using the full effective potential (not high-T expansion).
    
#     Parameters:
#     - T (float): Temperature
#     - phi0 (float): Field value at the false vacuum
#     - gD (float): Gauge coupling
#     - v (float): Symmetry breaking scale

#     Returns:
#     - S_E (float): The Euclidean action
#     """
#     vt = ConstructVt(T, phi0, gD, v)  # Build the tunneling potential

#     def integrand(S, T):
#         # Central difference for numerical derivative of the tunneling potential
#         h = max(1e-9, abs(S) * 1e-3)
#         dvt_ds = (vt(S + h) - vt(S - h)) / (2 * h)
#         dvt_ds = max(np.abs(dvt_ds), 1e-20)  # Regularization to avoid division by zero

#         # Difference between the full effective potential and the tunneling potential
#         v_eff_diff = fs.Veff(S, gD, T, v) - vt(S)

#         return np.real(v_eff_diff**1.5 / dvt_ds**2)

#     # Numerical integration from 0 to phi0
#     result = quad(lambda S: integrand(S, T), 0, phi0,
#                   limit=1000, epsrel=1.49e-5, epsabs=1.49e-7)[0]

#     # Return the full Euclidean action with prefactor
#     return (32 * np.pi * np.sqrt(2) / 3) * np.real(result)





def SE0(T, phi0, gD, v, debug=False):
    """
    Compute the Euclidean action S_E using the full effective potential (not high-T expansion).
    
    Parameters:
    - T (float): Temperature
    - phi0 (float): Field value at the false vacuum
    - gD (float): Gauge coupling
    - v (float): Symmetry breaking scale
    - debug (bool): If True, plots integrand for diagnostic
    
    Returns:
    - S_E (float): The Euclidean action
    """
    vt = ConstructVt(T, phi0, gD, v)

    def integrand(S):
        try:
            h = max(1e-9, abs(S) * 1e-3)

            vtS_plus = vt(S + h)
            vtS_minus = vt(S - h)
            vtS = vt(S)
            veffS = fs.Veff(S, gD, T, v)

            # Validación de estabilidad numérica
            if not all(np.isfinite(x) for x in [vtS_plus, vtS_minus, vtS, veffS]):
                return 0.0

            dvt_ds = (vtS_plus - vtS_minus) / (2 * h)
            dvt_ds = np.sign(dvt_ds)* np.clip(np.abs(dvt_ds), 1e-65, np.inf)

            v_eff_diff = veffS - vtS
            if v_eff_diff < 0:
                return 0.0

            return v_eff_diff**1.5 / dvt_ds**2

        except Exception:
            return 0.0


    # Evaluación de la integral con supresión de warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = quad(integrand, 0, phi0,
                      limit=20000, epsrel=1e-8, epsabs=1e-8)[0]

    return (32 * np.pi * np.sqrt(2) / 3) * result



# SE computation also needs gD and v
def SE(T, gD, v):
    phi1 = phi_root(T, gD, v)
    phi2 =  phi_min_Veff0(gD, v) 
    result = minimize_scalar(lambda S: SE0(T, S, gD, v), bounds=(phi1, phi2), method='bounded')
    phi_val = result.x
    SE00 = result.fun 
    return SE00, phi_val
