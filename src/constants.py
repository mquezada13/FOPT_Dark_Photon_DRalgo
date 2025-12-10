# constants.py
"""
Physical and model constants for FOPT_project.
All values are in natural units unless otherwise noted.
"""

# ==============================================================
# Model parameters
# ==============================================================
GF = 6.7088e-39       # Fermi constant [GeV^-2]
ab = 5.4076           # Numerical constant from thermal functions
g_dof = 10.75 + 4     # Effective relativistic degrees of freedom (g_SM + g_h) [check]
MPL = 2.4e18          # Reduced Planck mass [GeV]

# Derived quantity (cosmological redshift factor)
R = 1.67e-5 * (g_dof / 100) ** (-1 / 3)

# ==============================================================
# Cosmological scales
# ==============================================================
MPC = 3.0857e22       # 1 Mpc in meters
H_HUND = 1e5 / MPC    # Hubble scale factor (100 km/s/Mpc in SI units)
YEAR = 365.25 * 24 * 3600  # seconds per year

# ==============================================================
# Detector parameters (LISA, etc.)
# ==============================================================
T_LISA = 3 * YEAR     # LISA mission time
fmin = 3e-5           # Minimum frequency [Hz]
fmax = 0.5            # Maximum frequency [Hz]
SNR_THR = 10          # Detection threshold

# ==============================================================
# RGE initial conditions
# ==============================================================
gD0 = 0.6
lambdaS0= 1e-10
mS0 = 1e-10

