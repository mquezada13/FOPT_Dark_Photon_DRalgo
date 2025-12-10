# Name: Maura E. Ramirez-Quezada
# Date: 11.07.2024
# Description: This code contains the necessary to compute the bounce solution. Import this to the main code.
# File: bouncesol.py
# Version: V1
# Contact: mramirez@uni-mainz.de
#==================================================================================================================================================================
#==================================================================================================================================================================
#==================================================================================================================================================================
# VeffFunc.py

import Miselanie.constants as cs
import numpy as np
from scipy.special import kn
from scipy.interpolate import interp1d
import scipy.optimize as optimize
import cmath

#Handling complex numbers
Re1 = complex(1., 0.)
Imi = complex(0., 1.)

# Tree-level potential

def Vtree(S, v):
    muS2 = cs.mS0
    term1 = -muS2 * S**2 / 2
    term2 = cs.lambdaS0 * (S**4 / 4)
    VtreeComplete_result = term1 + term2
    return VtreeComplete_result


# Function to numerically evaluate the mass of S
def mPhi2(S, v):
    muS2 = cs.mS0
    return -muS2 + 3* cs.lambdaS0 * S**2


def mSigma2(S, v):
    muS2 = cs.mS0
    return -muS2 + cs.lambdaS0 * S**2


def mAp2(S, gD, v):
    return gD**2 * S**2

#Debye masses : For daisy term cut-function is to set limits on the contribution of the dark photon (not yet used)
def PiPhi(gD, T, v):
    return (cs.lambdaS0 / 3 + 0.25 * gD**2) * T**2

def cut(y):
    return y / 2 * kn(2, Re1 * np.sqrt(Re1 * y))

def PiAp(S, gD, T, v):
    return gD**2 / 3 * T**2

#================================================================================================================================================================
#===================================Coleman-Weinberg potential (Written in the MSbar : Vanishing counterterms)===================================================
#================================================================================================================================================================
# Useful functions
def kdelta(a, b):
    if isinstance(a, np.ndarray):
        mask = a == b
        return 1. * mask
    else:
        if a == b:
            return 1.
        else:
            return 0.

def Vcw(S, gD, v):
    mPhi2_value = mPhi2(S, v)
    mSigma2_value = mSigma2(S, v)
    mAp2_value = mAp2(S, gD, v)

    term1 = mPhi2_value**2 * (np.log(Re1 * mPhi2_value / v**2 + kdelta(mPhi2_value / v**2, 0)) - 3/2)
    term2 = mSigma2_value**2 * (np.log(Re1 * mSigma2_value / v**2 + kdelta(mSigma2_value / v**2, 0)) - 3/2)
    term3 = 3 * mAp2_value**2 * (np.log(Re1 * mAp2_value / v**2 + kdelta(mAp2_value / v**2, 0)) - 5/6)

    Vcw_value = 1 / (64 * np.pi**2) * (term1 + term2 + term3)
    return np.real(Vcw_value)



#==================================================================================================================================================================
#=================================================================Daisy potential==================================================================================
#==================================================================================================================================================================
# General function for daisy
def FD(mi, Pi):
    return (Re1 * (mi + Pi))**1.5 - (Re1 * mi)**1.5

# Constructing the daisy potential
def Vdaisy(S, gD, T, v):
    PiPhi_value = PiPhi(gD, T, v)
    PiAp_value = PiAp(S, gD, T, v)

    FD_mPhi = FD(mPhi2(S, v), PiPhi_value)
    FD_mSigma = FD(mSigma2(S, v), PiPhi_value)
    FD_mAp = FD(mAp2(S, gD, v), PiAp_value)

    Vdaisy_value = -(T / (12 * np.pi)) * (FD_mPhi + FD_mSigma + FD_mAp)
    return np.real(Vdaisy_value)



  
#==================================================================================================================================================================
#======================================================================Non-zero temperature potential==============================================================
#==================================================================================================================================================================
# Importing the data for J-factor
IntforVTdata = np.loadtxt('Miselanie/VT_integralNumeric.dat')
# Creation of the interpolating function
Integrand = interp1d(IntforVTdata[:, 0], IntforVTdata[:, 1], kind='linear', fill_value=0, bounds_error=False)

# Definition of the function JB: y= mi^2/T^2
def JB(y):
    return np.where(y < 600, Integrand(y), 0)



#High temperature limit
def JBhighexp(y):
    pi = np.pi
    term1 = -pi**4 / 45
    term2 = (pi**2 / 12) * y
    term3 = -(pi / 6) * np.maximum(0,y)**(3/2) 
    term4 = -(1 / 32) * y**2 * (np.log(np.maximum(1e-10, np.abs(y))) - cs.ab)
    return term1 + term2 + term3 + term4




# Non-zero temperature potential (full)
def VTfull(S, gD, T, v):
    mPhi2_value = mPhi2(S, v) / T**2
    mSigma2_value = mSigma2(S, v) / T**2
    mAp2_value = mAp2(S, gD, v) / T**2

    return (T**4) / (2 * np.pi**2) * (JB(mPhi2_value) + JB(mSigma2_value) + 3 * JB(mAp2_value))

# Non-zero temperature potential (high-T expansion)
def V_highT(S, gD, T, v):
    mPhi2_value = mPhi2(S, v) / T**2
    mSigma2_value = mSigma2(S, v) / T**2
    mAp2_value = mAp2(S, gD, v) / T**2

    return (T**4) / (2 * np.pi**2) * (JBhighexp(mPhi2_value) + JBhighexp(mSigma2_value) + 3 * JBhighexp(mAp2_value))



#






#==================================================================================================================================================================
#===========================================================================EFFECTIVE POTENTIAL====================================================================
#==================================================================================================================================================================
#Effective potential without temperature corrections
def Veff0(S, gD, v):
    Veff_at_zero = Vtree(0, v) + Vcw(0, gD, v)
    return np.real(Vtree(S, v) + Vcw(S, gD, v) - Veff_at_zero)

#Effective potential with temperature corrections
def Veff(S,T,gD, v):
    Veff_at_zero_S = (Vtree(0, v) + Vcw(0, gD, v) + VTfull(0, gD, T, v) + Vdaisy(0, gD, T, v))
    result = np.real((Vtree(S, v) + Vcw(S, gD, v) + VTfull(S, gD, T, v) + Vdaisy(S, gD, T, v)) - Veff_at_zero_S)
    return np.where(T > 0, result, Veff0(S, gD, v))

#Effective potential with temperature corrections (high-T expansion)
def Veff_HighT(S,T, gD, v):
    Veff_at_zero_S = (Vtree(0, v) + Vcw(0, gD, v) + V_highT(0, gD, T, v) + Vdaisy(0, gD, T, v))
    result = np.real((Vtree(S, v) + Vcw(S, gD, v) + V_highT(S, gD, T, v) + Vdaisy(S, gD, T, v)) - Veff_at_zero_S)
    return np.where(T > 0, result, Veff0(S, gD, v))




